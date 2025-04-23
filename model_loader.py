import json
import os
import requests
import subprocess
import time
from PyQt6.QtWidgets import QMessageBox


class ModelLoader:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_or_create_config()
        self.model = None

    def load_or_create_config(self):
        default_config = {
            "default_model": {
                "type": "ollama",
                "model_name": "mistral:latest"
            },
            "generation": {
                "temperature": 0.7,
                "max_tokens": 512
            },
            "performance": {
                "backend": "auto",
                "last_benchmark": {
                    "cpu": None,
                    "gpu": None
                }
            }
        }

        if not os.path.exists(self.config_path):
            self.save_config(default_config)
            return default_config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            config.setdefault("default_model", {}).update(default_config["default_model"])
            config.setdefault("generation", {}).update(default_config["generation"])
            config.setdefault("performance", {}).update(default_config["performance"])
            config["performance"].setdefault("backend", "auto")
            config["performance"].setdefault("last_benchmark", {"cpu": None, "gpu": None})

            return config

        except (json.JSONDecodeError, IOError) as e:
            QMessageBox.warning(None, "Config Error",
                f"Failed to load config file:\n{str(e)}\n\nA default config will be created.")
            self.save_config(default_config)
            return default_config

    def save_config(self, config=None):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config or self.config, f, indent=2)

    def choose_best_backend(self):
        times = self.config["performance"].get("last_benchmark", {})
        cpu_time = times.get("cpu")
        gpu_time = times.get("gpu")

        if cpu_time is None or gpu_time is None:
            return "cpu"

        return "gpu" if gpu_time < cpu_time else "cpu"

    def load_model(self):
        backend_pref = self.config["performance"].get("backend", "auto")
        if backend_pref == "auto":
            best_backend = self.choose_best_backend()
            print(f"[Auto-selected backend: {best_backend}]")
            self.config["performance"]["backend"] = best_backend
            self.save_config()
        else:
            best_backend = backend_pref

        model_type = self.config["default_model"]["type"]
        if model_type == "ollama":
            return "ollama"
        elif model_type == "llama.cpp":
            return self._load_llamacpp_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_llamacpp_model(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama_cpp module not installed. Run `pip install llama-cpp-python`.")

        model_path = self.config["default_model"].get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        return Llama(model_path=model_path)

    def get_generation_settings(self):
        return self.config.get("generation", {})

    def is_ollama_running(self):
        try:
            response = requests.get("http://localhost:11434")
            return response.status_code == 200
        except:
            return False

    def generate_with_ollama_stream(self, prompt: str):
        model_name = self.config["default_model"].get("model_name", "mistral")
        settings = self.get_generation_settings()

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": settings.get("temperature", 0.7),
                "stream": True
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                except json.JSONDecodeError:
                    continue

    def generate_single_response(self, prompt: str) -> str:
        backend = self.config["default_model"]["type"]
        if backend == "ollama":
            chunks = []
            for chunk in self.generate_with_ollama_stream(prompt):
                chunks.append(chunk)
            return ''.join(chunks)
        return "[Only Ollama supported]"
    
    def generate_sync(self, prompt: str) -> str:
        """
        Synchronously generate a full response using the configured backend.
        This is used for things like prompt chaining.
        """
        model_name = self.config["default_model"].get("model_name", "mistral")
        settings = self.get_generation_settings()

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": settings.get("temperature", 0.7),
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            return f"[Error in generate_sync: {str(e)}]"


    def list_ollama_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().split("\n")[1:]
            return [line.split()[0] for line in lines if line]
        except Exception:
            return []

    def run_performance_test(self):
        prompt = "Benchmarking system performance with this simple prompt."
        model = self.config["default_model"]["model_name"]
        results = {}

        for device in ["cpu", "gpu"]:
            try:
                start = time.time()
                requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                results[device] = round(time.time() - start, 2)
            except Exception as e:
                print(f"[Benchmark error on {device.upper()}]: {e}")
                results[device] = None

        self.config["performance"]["last_benchmark"] = results
        self.save_config()
        return results