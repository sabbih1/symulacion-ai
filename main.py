import multiprocessing
import os
import redis
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import DATA_EXTRACTION_PATH
from models.text2img.model import InferenceArgs
from services.dreambooth import TrainingArgs
from tasks import celery_app
from utils.utils import logger, extract_zip_file, count_valid_images

print("*******",multiprocessing.get_start_method(allow_none=True))
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")


r = redis.Redis(host = "localhost", port = 6379, db = 0)

app = FastAPI()

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

    r.xadd("data_upload_messages", event, "*")


@app.post("/uploadfile/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Check that the uploaded file is a zip file
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    background_tasks.add_task(extract_file_from_zip, file)
    return {"status": "File upload and processing initiated"}


@app.post("/train")
async def train_model(training_args: TrainingArgs):
    # Your training code here using the provided training_args
    task = celery_app.send_task("tasks.call_training_function", args=(training_args.dict(),))
    return task.id

@app.post("/inference_dreambooth")
async def inference_dreambooth(inference_args: InferenceArgs):
    task = celery_app.send_task("tasks.call_inference_model", args=(inference_args.dict(),))
    return task.id

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
