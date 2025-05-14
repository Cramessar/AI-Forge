# plugins/image_gen/plugin.py
from core.plugin_base import AIForgePlugin
import torch
import json
import os
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import importlib.util

# ─── Plugin Class ──────────────────────────────────────────
class Plugin(AIForgePlugin):
    def __init__(self, config={}):
        super().__init__(config)
        self.pipe = None
        self.current_model_id = None

        default_model_id = "stabilityai/stable-diffusion-2-1"
        print(f"🖼️ [ImageGen] Loading default model: {default_model_id}")

        try:
            self.pipe = self.load_pipeline(default_model_id)
            self.current_model_id = default_model_id
        except Exception as e:
            print(f"💥 [ImageGen Error] Failed to load default model: {e}")

    def get_name(self):
        return "Image Generation"

    def plugin_type(self):
        return "image_gen"

    def run(self, input_data: dict):
        print("🎨 [ImageGen] run() called")

        config_path = os.path.join(os.path.dirname(__file__), "image_gen_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                print(f"🔁 [ImageGen] Reloaded config: {self.config}")
            except Exception as e:
                print(f"⚠️ [ImageGen] Failed to reload config: {e}")
        else:
            print("⚠️ [ImageGen] No config found. Using defaults.")

        prompt = input_data.get("original_prompt", "") or input_data.get("text", "")
        if not prompt:
            return {"error": "No prompt provided."}
        if prompt.lower().startswith("image:"):
            prompt = prompt.split("image:", 1)[-1].strip()

        model_id = self.config.get("model_id", "stabilityai/stable-diffusion-2-1")

        if self.current_model_id != model_id:
            print(f"🔄 [ImageGen] Switching to model: {model_id}")
            try:
                self.pipe = self.load_pipeline(model_id)
                self.current_model_id = model_id
            except Exception as e:
                print(f"💥 [ImageGen Error] Failed to load model '{model_id}': {e}")
                return {"error": f"Failed to load model: {e}"}

        guidance_scale = self.config.get("guidance_scale", 7.5)
        num_inference_steps = self.config.get("num_inference_steps", 50)
        width = self.config.get("width", 512)
        height = self.config.get("height", 512)

        print(f"🧪 [ImageGen] Steps: {num_inference_steps}, Scale: {guidance_scale}, Res: {width}x{height}")
        print(f"🎨 [ImageGen] Prompt: {prompt}")

        try:
            output = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width
            )
            print("✅ Pipeline executed successfully.")

            if isinstance(output, list):
                image = output[0]
            elif hasattr(output, "images"):
                image = output.images[0]
            else:
                image = output

            image_path = "generated_image.png"
            image.save(image_path)
            print(f"✅ [ImageGen] Image saved as '{image_path}'")
            return {"image_path": image_path}

        except Exception as e:
            print(f"💥 [ImageGen Error] {e}")
            return {"error": f"Image generation failed: {e}"}

    def load_pipeline(self, model_id):
        if model_id == "Freepik/F-Lite":
            print(f"🧠 Loading custom pipeline for {model_id} from local file")

            try:
                file_path = os.path.join(os.path.dirname(__file__), "custom_pipelines", "f_lite_pipeline.py")
                spec = importlib.util.spec_from_file_location("f_lite_pipeline", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                pipeline_cls = getattr(module, "FLitePipeline")

                pipe = pipeline_cls.from_pretrained(
                    "./models/Freepik/F-Lite",
                    torch_dtype=torch.float16,
                )
                return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            except Exception as e:
                print(f"💥 [ImageGen Error] Failed to import or load custom pipeline:\n{repr(e)}")
                raise RuntimeError(repr(e))

        else:
            print(f"🧠 Loading standard pipeline for {model_id}")
            return StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                revision="fp16" if "1-5" in model_id or "2-1" in model_id else None,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
