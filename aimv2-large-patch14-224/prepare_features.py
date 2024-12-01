import os
from typing import Tuple, Dict

import argparse
import mlx.core as mx
import numpy as np
from tqdm import tqdm
from PIL import Image
from aim.v2.utils import load_pretrained
from aim.v1.torch.data import val_transforms

from sklearn.metrics.pairwise import cosine_similarity

# モデルとプロセッサのロード
model = load_pretrained("aimv2-large-patch14-336", backend="mlx")
transform = val_transforms(img_size=336)

# 特徴量の保存またはロード
def extract_features(image_dir: str, features_file: str) -> Dict[str, mx.array]:
    """
    Extracts features from images in a specified directory and saves them to a file.

    Args:
        image_dir (str): The directory containing the images to process.
        features_file (str): The file path where the extracted features will be saved.

    Returns:
        Dict[str, mx.array]: A dictionary mapping image filenames to their extracted features.

    Raises:
        Exception: If an error occurs during image processing, it will be caught and printed.

    Notes:
        - Only processes images with a ".jpg" extension.
        - Converts images to RGB format before processing.
        - Uses a pre-trained model to extract features from the images.
        - Saves the extracted features in the SafeTensors format.
    """
    features_dict: Dict[str, mx.array] = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    img_file: str
    for img_file in tqdm(image_files, desc="Extracting features", unit="image"):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = transform(image).unsqueeze(0)
            inputs = mx.array(inputs.numpy())
            features: mx.array = model(inputs)
            features_dict[img_file] = features
        except Exception as e:
            print(f"Error processing {img_file}: {e.__class__.__name__} - {e}")

    if features_file.endswith(".safetensors"):
        features_file = features_file[:-12]

    mx.save_safetensors(features_file, features_dict)
    print(f"Features saved to {features_file}.safetensors")


# メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str,default="./val2017", help="images directory")
    parser.add_argument("--features_file", type=str, default="coco_features", help="features file name")
    args = parser.parse_args()

    extract_features(args.image_dir, args.features_file)