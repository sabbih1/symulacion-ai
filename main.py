
from tasks import extract_file_from_zip
from celery import Celery
from celery.result import AsyncResult
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
from utils import logger, extract_zip_file, count_valid_images
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import tempfile

app = FastAPI()

@app.post("/uploadfile/")
def upload_file(file: UploadFile = File(...)):
    # Check that the uploaded file is a zip file
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    upload_task = extract_file_from_zip.delay(file)
    return {"task_id": upload_task.id}

@app.get("/task_status/{task_id}")
async def task_status(task_id: str):
    async_result = AsyncResult(task_id, app=celery_app)
    if async_result.state == "PENDING":
        raise HTTPException(status_code=404, detail="Task not found")
    elif async_result.state in ["SUCCESS", "FAILURE"]:
        return {
            "task_id": task_id,
            "status": async_result.state,
            "result": async_result.result,
        }
    else:
        return {"task_id": task_id, "status": async_result.state}