import torch
from diffusers import DiffusionPipeline
from typing import List, Optional, Union
import PIL.Image

# 🛠️ Local fallback version for older diffusers
def randn_tensor(shape, generator=None, device=None, dtype=torch.float32):
    if generator is not None:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)

class FLitePipeline(DiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker=None, feature_extractor=None):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ):
        if isinstance(prompt, str):
            prompt = [prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(self.device)
        text_embeddings = self.text_encoder(input_ids)[0]

        if guidance_scale > 1.0:
            uncond_tokens = [""] * len(prompt)
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        latents = randn_tensor(
            (len(prompt), self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=text_embeddings.dtype
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = [PIL.Image.fromarray((img * 255).astype("uint8")) for img in image]

        return {"images": images}

