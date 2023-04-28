import zipfile
import os
import logging
import requests

from PIL import Image
from fastapi import HTTPException
from io import BytesIO



# Configure the logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from enum import Enum
import torch

class TorchDtype(Enum):
    FLOAT16 = torch.float16
    FLOAT32 = torch.float32

class ModelID(Enum):
    V1_4 = {"model_path": "CompVis/stable-diffusion-v1-4", "resolution": (512, 512)}
    V1_5 =  {"model_path": "runwayml/stable-diffusion-v1-5", "resolution": (512, 512)}
    V2_BASE =  {"model_path": "stabilityai/stable-diffusion-2-base", "resolution": (512, 512)}
    V2 = {"model_path": "stabilityai/stable-diffusion-2", "resolution": (768, 768)}
    V2_1_BASE = {"model_path": "stabilityai/stable-diffusion-2-1-base", "resolution": (512, 512)}
    V2_1 = {"model_path": "stabilityai/stable-diffusion-2-1", "resolution": (768, 768)}


def extract_zip_file(zip_file_path, temp_dir):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    with zipfile.ZipFile(zip_file_path) as zip_file:
        zip_file.extractall(temp_dir)
        subdirs = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
        print("subdirs", subdirs)
        if len(subdirs) != 1:
            raise HTTPException(status_code=400, detail="Zip file does not contain a single folder")
        extracted_folder_path = subdirs[0]
    return extracted_folder_path


def count_valid_images(extracted_folder_path):
    num_images = 0
    for filename in os.listdir(extracted_folder_path):
        file_path = os.path.join(extracted_folder_path, filename)
        logger.debug(f"File path for image: {file_path}")
        try:
            with Image.open(file_path) as img:
                num_images += 1
        except:
            pass
    if num_images == 0:
        raise HTTPException(status_code=400, detail="No valid images found in the extracted folder")
    return num_images


def fetch_image(url):
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((224, 224))
    return init_image