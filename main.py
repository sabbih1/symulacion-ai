
# from celery import Celery
# from celery.result import AsyncResult
import os
import tempfile
import accelerate
import itertools
import torch
import redis # Connect to a local redis instance

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from services.dreambooth import TrainingArgs, training_function, text_encoder, vae, unet
from utils.utils import logger, extract_zip_file, count_valid_images
from config import DATA_EXTRACTION_PATH
from celery import Celery, signals

app = FastAPI()
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
r = redis.Redis(host = 'localhost', port = 6379, db = 0)

async def extract_file_from_zip(file):
    # Save the zip file to disk
    file_path = os.path.join(DATA_EXTRACTION_PATH, file.filename)
    complete_folder_path = file_path.split(".zip")[0]
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    logger.debug(f"File path: {file_path}")
    # # Extract the contents of the zip file
    extracted_folder_path = extract_zip_file(file_path, complete_folder_path)

    # # Check if the extracted folder contains valid images
    num_images = count_valid_images(extracted_folder_path)
    event = {"status": "success", "num_images": num_images, "folder_path": extracted_folder_path}
    logger.debug(f"{event}")

    r.xadd("data_upload_messages", event, '*')


class TaskResponse(BaseModel):
    task_id: str


@app.post("/uploadfile/", response_model=TaskResponse)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Check that the uploaded file is a zip file
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    return background_tasks.add_task(extract_file_from_zip, file)
    # task = celery_app.send_task("extract_file_from_zip", args=(file,))


@celery_app.task
def call_accelerate(training_args: TrainingArgs):
    accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, training_args), num_processes=1)
    '''
    After the training is completed, the code iterates through the parameters of the unet and text_encoder models.
    It deletes the gradient tensors associated with each parameter to free up some GPU memory.
    '''
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()


@app.post("/train")
async def train_model(training_args: TrainingArgs):
    # Your training code here using the provided training_args
    task = celery_app.send_task("call_accelerate", args=(training_args,))
