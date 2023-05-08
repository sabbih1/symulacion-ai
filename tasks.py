
import multiprocessing
import torch
import accelerate
import itertools
import os

from config import BASE_OUTPUT_IMG_PATH
from celery import Celery
from celery.result import AsyncResult
from services.dreambooth import TrainingArgs, training_function, text_encoder, vae, unet, load_newly_trained_model, inference_dreambooth
celery_app = Celery("tasks", broker="redis://localhost:6379/0",backend="redis://localhost:6379/1")

celery_app.autodiscover_tasks()  

@celery_app.task
def call_training_function(training_args: dict):
    training_function(text_encoder, vae, unet, training_args)
    '''
    After the training is completed, the code iterates through the parameters of the unet and text_encoder models.
    It deletes the gradient tensors associated with each parameter to free up some GPU memory.
    '''
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()

    return {"status": "success", "model_path": os.path.join(training_args["output_dir"])}


@celery_app.task
def call_inference_model(inference_args: dict):
    model_dir = inference_args["model_dir"]
    # datatype = inference_args["datatype"]
    output_image_dir = inference_args["output_image_dir"]
    prompt = inference_args["prompt"]
    negative_prompt = inference_args["negative_prompt"]
    num_images_per_prompt=inference_args["num_images_per_prompt"]
    num_inference_steps=inference_args["num_inference_steps"]
    guidance_scale=inference_args["guidance_scale"]
    pipeline = load_newly_trained_model(model_dir,)# datatype)
    images = inference_dreambooth(pipeline, prompt, negative_prompt, num_images_per_prompt, num_inference_steps, guidance_scale)
    if not os.path.exists(output_image_dir): os.makedirs(os.path.join(BASE_OUTPUT_IMG_PATH, output_image_dir), exist_ok=True)
    folder_path = os.path.join(BASE_OUTPUT_IMG_PATH, output_image_dir)
    for i, img in enumerate(images):
        img.save(os.path.join(folder_path,f"{i}.jpg"))
    return os.listdir(folder_path)