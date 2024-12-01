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

model = load_pretrained("aimv2-large-patch14-336", backend="mlx")
transform = val_transforms(img_size=336)

image_dir: str = "./val2017"


def extract_features(image_dir) -> Dict[str, mx.array]:
    """
    Extract features from images in a specified directory.

    Args:
        image_dir (str): The directory containing the images to process.

    Returns:
        Dict[str, mx.array]: A dictionary where the keys are image filenames and the values are the extracted features.

    Raises:
        Exception: If there is an error processing an image, it will be caught and printed.
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
    return features_dict


def get_query_features(query_image_path) -> mx.array:
    """
    Extracts features from a query image using a pre-trained model.

    Args:
        query_image_path (str): The file path to the query image.

    Returns:
        mx.array: The extracted features from the query image.
    """
    print("Extracting features from query image...")
    query_image = Image.open(query_image_path).convert("RGB")
    query_inputs = transform(query_image).unsqueeze(0)
    query_inputs = mx.array(query_inputs.numpy())
    query_features: mx.array = model(query_inputs)
    return query_features

def get_features_dict(features_file: str) -> Dict[str, mx.array]:
    """
    Retrieves a dictionary of features from a specified file. If the features file
    exists in the .safetensors format, it loads the features from the file. Otherwise,
    it extracts features from images in a specified directory, saves them to the 
    .safetensors file, and then returns the features dictionary.

    Args:
        features_file (str): The path to the features file (without the .safetensors extension).

    Returns:
        Dict[str, mx.array]: A dictionary where the keys are image identifiers and the values
                            are the corresponding feature arrays.
    """
    if os.path.exists(f"{features_file}.safetensors"):
        print("Loading saved features...")
        # mlx.core による読み込み
        features_dict: Dict[str, mx.array] = mx.load(f"{features_file}.safetensors")
        print(f"Features loaded from {features_file}.safetensors")
    else:
        print("Extracting features from all images...")
        features_dict = extract_features(image_dir)
        mx.save_safetensors(features_file, features_dict)
        print("Features saved to coco_features.safetensors")
    return features_dict

def main(features_dict: Dict[str, mx.array], query_features: np.array) -> Tuple[str, float]:
    """
    Finds the most similar image to the query features from a dictionary of image features.

    Args:
        features_dict (Dict[str, mx.array]): A dictionary where keys are image file names and values are their corresponding feature arrays.
        query_features (np.array): A numpy array representing the features of the query image.

    Returns:
        Tuple[str, float]: A tuple containing the file name of the most similar image and the similarity score.
    """
    best_match_file = None
    best_similarity = -1

    for img_file, img_features in tqdm(features_dict.items(), desc="Finding the most similar image..", unit="image"):
        img_features = np.array(img_features).reshape(1, -1)
        similarity = cosine_similarity(query_features, img_features)[0, 0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_file = img_file

    print(f"Most similar image: {best_match_file}")
    print(f"Similarity score: {best_similarity:.4f}")
    return best_match_file, best_similarity

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--features_file", type=str, default="coco_features")
    argparser.add_argument("--query_file_path", type=str, default="./test_search_image/egg_plant.jpg")
    args = argparser.parse_args()
    query_file_path = args.query_file_path
    features_dict = get_features_dict(args.features_file)
    query_features = get_query_features(query_file_path)
    query_features = np.array(query_features).reshape(1, -1)
    main(features_dict, query_features)
