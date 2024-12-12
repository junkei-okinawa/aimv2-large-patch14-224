import os
import tomllib
from typing import Tuple, Dict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

def extract_features(model, processor, image_dir: str) -> Dict[str, torch.Tensor]:
    """
    Extract features from images in a specified directory.

    Args:
        image_dir (str): The directory containing the images to process.

    Returns:
        Dict[str, mx.array]: A dictionary where the keys are image filenames and the values are the extracted features.

    Raises:
        Exception: If there is an error processing an image, it will be caught and printed.
    """
    features_dict = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for img_file in tqdm(image_files, desc="Extracting features", unit="image"):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).detach().cpu()
            features_dict[img_file] = features
        except Exception as e:
            print(f"Error processing {img_file}: {e.__class__.__name__} - {e}")
    return features_dict


with open("config.toml", mode="rb") as f:
    config = tomllib.load(f)

# Load the model and processor
features_path = os.path.join("./features", config["features_file"])

# GPUの設定 cpu or cuda or mps
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

processor = AutoImageProcessor.from_pretrained(config["processor_file"]) #, local_files_only=True, cache_dir="./models"
model = AutoModel.from_pretrained(config["model_file"], trust_remote_code=True).to(device)
features_dict = extract_features(model, processor, config["image_dir"])
torch.save(features_dict, features_path)
model.save_pretrained(config["models_dir_local"])
processor.save_pretrained(config["models_dir_local"])