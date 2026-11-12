import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Indian Traffic Sign Recognition using BanTrafficNet"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    
    MODEL_PATH: Path = ARTIFACTS_DIR / "model" / "bantrafficnet_indian.onnx"
    KERAS_MODEL_PATH: Path = ARTIFACTS_DIR / "model" / "bantrafficnet_indian.keras"
    CLASS_MAPPING_PATH: Path = METADATA_DIR / "class_names.json"
    DATASET_SUMMARY_PATH: Path = METADATA_DIR / "dataset_summary.json"
    METRICS_PATH: Path = ARTIFACTS_DIR / "reports" / "evaluation_metrics.json"
    
    INPUT_SIZE: tuple = (96, 96)
    NUM_CLASSES: int = 56
    
    CORS_ORIGINS: list = ["*"]

settings = Settings()
