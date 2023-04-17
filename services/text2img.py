#@title Load the Stable Diffusion model
# Load models and create wrapper for stable diffusion
import requests
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

class TextToImageGenerator:
    def __init__(self, model_id_or_path="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)

    def generate_image(self, prompt):
        image = self.pipe(prompt).images[0]
        return image



# Example usage:
text_to_image_generator = TextToImageGenerator()
generated_image = text_to_image_generator.generate_image(prompt="a photo of an astronaut riding a horse on mars")
# generated_image.save("generated_image.png")

image_to_image_transformer = ImageToImageGenerator()
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
initial_image = image_to_image_transformer.fetch_image(url)
transformed_image = image_to_image_transformer.transform_image(prompt="A fantasy landscape, trending on artstation", init_image=initial_image)
# transformed_image.save("transformed_image.png")
