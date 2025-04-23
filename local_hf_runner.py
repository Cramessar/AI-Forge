import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

class HFRunner:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"  # Uses Accelerate under the hood
        )
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate(self, prompt, max_new_tokens=256, temperature=0.7):
        response = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        return response[0]["generated_text"]

#Friendly name-to-ID map for Hugging Face models
HF_MODEL_MAP = {
    "Phi-2 (HF)": "microsoft/phi-2",
    "TinyLlama (HF)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # add more as you see fit
}
