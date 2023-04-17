import requests
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


class ImageToImageGenerator:
    def __init__(self, model_id_or_path="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)


    def generate_image(self, prompt, init_image, strength=0.75, guidance_scale=7.5):
        images = self.pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
        return images[0]


# Example usage:

image_to_image_transformer = ImageToImageGenerator()
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
initial_image = image_to_image_transformer.fetch_image(url)
transformed_image = image_to_image_transformer.generate_image(prompt="A fantasy landscape, trending on artstation", init_image=initial_image)
# transformed_image.save("transformed_image.png")

