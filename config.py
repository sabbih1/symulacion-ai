from utils.utils import TorchDtype, ModelID

MODEL_ID_OR_PATH = ModelID.V1_5.value["model_path"]
DEVICE = "cuda"
GUIDANE_SCALE = 7.5
STRENGTH = 7.5
DATA_TYPE = TorchDtype.FLOAT16.value
DATA_EXTRACTION_PATH = "/home/fsuser/mi_workspace/symulacion/data"