
# from celery import Celery
# from celery.result import AsyncResult
# celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
import os
import tempfile
import accelerate
import itertools
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException

from pydantic import BaseModel
from typing import Optional
from services.dreambooth import TrainingArgs, training_function, text_encoder, vae, unet
from utils.utils import logger, extract_zip_file, count_valid_images
from config import DATA_EXTRACTION_PATH

app = FastAPI()

async def extract_file_from_zip(file):
    # Save the zip file to disk
    file_path = os.path.join(DATA_EXTRACTION_PATH, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    logger.debug(f"File path: {file_path}")

    # Extract the contents of the zip file
    extracted_folder_path = extract_zip_file(file_path, DATA_EXTRACTION_PATH)

    # Check if the extracted folder contains valid images
    num_images = count_valid_images(extracted_folder_path)

    return {"status": "success", "num_images": num_images, "folder_path": extracted_folder_path}




@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Check that the uploaded file is a zip file
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    return await extract_file_from_zip(file)


@app.post("/train")
async def train_model(training_args: TrainingArgs):
    # Your training code here using the provided training_args
    accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, training_args), num_processes=1)
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()
