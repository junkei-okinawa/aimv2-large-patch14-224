import io
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def model_loader(
        cache_dir_path: str,
        model_file_path: str,
        processor_file_path: str,
    ) -> tuple:
    """
    load for model and processor
    """
    # GPUの設定 cpu or cuda or mps
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    processor = AutoImageProcessor.from_pretrained(model_file_path)
    model = AutoModel.from_pretrained(processor_file_path, trust_remote_code=True).to(device)
    return model, processor, device

def feature_loader(features_file: str) -> dict:
    """
    load for features
    """
    if not os.path.exists(features_file):
        raise FileNotFoundError("Features file not found")

    features_dict = torch.load(features_file, weights_only=True)
    return features_dict


def get_query_features(model, processor, device: str, image_bytes: bytes):
    """
    Extract features from query image
    """
    if len(image_bytes) == 0:
        raise ValueError("Uploaded file size is 0")

    query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    query_inputs = processor(images=query_image, return_tensors="pt").to(device)
    query_outputs = model(**query_inputs)
    query_features = query_outputs.last_hidden_state.mean(dim=1).detach().cpu()
    return {"features": query_features}
