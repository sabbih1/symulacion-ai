from pydantic import BaseModel

class TaskResponse(BaseModel):
    task_id: str

class InferenceArgs(BaseModel):
    model_dir: str
    prompt: str
    negative_prompt: str = ""
    num_images_per_prompt: int = 10
    num_inference_steps: int = 10
    guidance_scale: float = 7.5

class TrainingArgs(BaseModel):
    pretrained_model_name_or_path: str
    resolution: int
    center_crop: bool = False
    train_text_encoder: bool = False
    instance_data_dir: str
    instance_prompt: str
    learning_rate: float = 5e-07
    max_train_steps: int = 200
    save_steps: int = 50
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    seed: int = 3434554
    with_prior_preservation: bool
    prior_loss_weight: float
    sample_batch_size: int = 2
    class_data_dir: str
    class_prompt: str
    num_class_images: int
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    output_dir: str = "dreambooth-concept"
