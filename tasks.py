from celery import Celery
from celery.result import AsyncResult
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
from utils import logger, extract_zip_file, count_valid_images
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import tempfile


@celery_app.task
async def extract_file_from_zip(file):

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the zip file to disk
        file_path = os.path.join(temp_dir, file.filename)
        logger.debug(f"File path: {file_path}")
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Extract the contents of the zip file
        extracted_folder_path = extract_zip_file(file_path, temp_dir)

        # Check if the extracted folder contains valid images
        num_images = count_valid_images(extracted_folder_path)

    return {"status": "success", "num_images": num_images, "folder_path": extracted_folder_path}


