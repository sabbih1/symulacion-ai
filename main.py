import multiprocessing
import os
import redis
import uvicorn
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from celery.result import AsyncResult
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from config import DATA_EXTRACTION_PATH, BASE_OUTPUT_IMG_PATH
from models.text2img.model import InferenceArgs
from services.dreambooth import TrainingArgs
from tasks import celery_app
from utils.utils import logger, extract_zip_file, count_valid_images
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

print(multiprocessing.get_start_method(allow_none=True))
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")


r = redis.Redis(host = "localhost", port = 6379, db = 0)

app = FastAPI()

class Message(BaseModel):
    message: str

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
    return event


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Check that the uploaded file is a zip file
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    return await extract_file_from_zip(file)

@app.post("/train")
async def train_model(training_args: TrainingArgs):
    # Your training code here using the provided training_args
    task = celery_app.send_task("tasks.call_training_function", args=(training_args.dict(),))
    return task.id

@app.post("/inference_dreambooth")
async def inference_dreambooth(inference_args: InferenceArgs):
    task = celery_app.send_task("tasks.call_inference_model", args=(inference_args.dict(),))
    return task.id

@app.get("/images/", response_model=List[str])
async def get_image_list(output_image_dir: str):
    image_files = os.listdir(os.path.join(BASE_OUTPUT_IMG_PATH, output_image_dir))
    image_urls = [os.path.join(output_image_dir, image) for image in image_files]
    return image_urls

@app.get("/images/{image_dir_name}")
async def get_image(image_dir: str, file_name: str):
    image_path = os.path.join(BASE_OUTPUT_IMG_PATH, image_dir, file_name)
    logger.debug(f"image path recv{ image_path}")
    logger.info(f"image path recv{ image_path}")

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/jpeg")

@app.get("results/{task_id}")
async def get_task_result(task_id):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():
        return {"task_id": str(task_id), "status": "Not ready"}
    result = task.get()
    return {"task_id": str(task_id), "status": str(result)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="trace", reload=True)