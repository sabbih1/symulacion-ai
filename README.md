Run `python3 main.py` and `celery -A main.celery_app worker -l INFO` in the main `symulacion-ai` folder

Goto `http://ipaddress:8000/docs`

Step 1. Upload a zip file containing a single folder of images using `/uploadfile/`

|--my_images.zip/
    |--my_images/
        |-- image1.jpg
        |-- image2.jpg
        |-- image3.png


The upload file will return a sucess status and model dir path. 

Step 2. Provide the following required values in the train JSON for `/train` endpoint. An example is shared below.  
```
  "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5", #base training image
  "resolution": 512,
  "instance_data_dir": "/home/fsuser/mi_workspace/symulacion/data/Coco/Coco", # Data directory returned from step 1.
  "instance_prompt": "coco_dog", # A unique prompt for the subject
  "num_class_images": 17, # This int value is also returned in step 1.
  "output_dir": "coco_dog_images", # Directory name for saving the trained model.
  ```

  Step 3. Use the `output_dir` provided in the previous step here to use the newly trained model for inference using `/inference_dreambooth` endpoint.
  ```
  {
  "model_dir": "coco_dog_images",  # provided in step 2
  "output_image_dir": "coco_dog_images_folder", # images generated will be stored here.
  "prompt": "A coco_dog as an astronaut",
  "negative_prompt": "",
  "num_images_per_prompt": 10,
  "num_inference_steps": 10,
  "guidance_scale": 7.5
}
```

Step 4. 

Use the  `output_image_dir` provided in the previous step to get the list of file names generated using `/images/` endpoint.

Step 5. 
Use the `/images/{image_dir_name}` endpoint to get image file generated during the inference. The filenames can be obtained in the previous step.
# Known Issue:

Async endpoints require the celery worker to be run with the `-P solo` flag. Because, there is currently an issue with CUDA initialization in multiprocessing setup.


Ignore `results/{task_id}` endpoint
