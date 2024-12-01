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

# 検索対象画像のディレクトリ
image_dir: str = "./val2017"


# クエリ画像の特徴量
def get_query_features(query_image_path: str) -> mx.array:
    print("Extracting features from query image...")
    query_image = Image.open(query_image_path).convert("RGB")
    query_inputs = transform(query_image).unsqueeze(0)
    query_inputs = mx.array(query_inputs.numpy())
    query_features: mx.array = model(query_inputs)
    return query_features

# 類似画像を検索
def find_most_similar_image(features_dict: Dict[str, mx.array], query_features: np.array) -> Tuple[str, float]:
    """
    Find the most similar image based on cosine similarity.

    Args:
        features_dict (Dict[str, mx.array]): A dictionary where keys are image file names and values are their corresponding feature vectors.
        query_features (np.array): The feature vector of the query image.

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

    return best_match_file, best_similarity

# メイン処理
if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser(description="Search for the most similar image")
    parser.add_argument("query_image", type=str, help="Path to the query image")
    parser.add_argument("--features", type=str, default="coco_features", help="Path to the saved features file")
    args = parser.parse_args()

    # クエリ画像と特徴量のロード
    query_image_path = args.query_image
    features_path = args.features

    if features_path.endswith(".safetensors"):
        features_path = features_path[:-12]

    if not os.path.exists(f"{features_path}.safetensors"):
        print(f"Features file {features_path} not found. Run prepare_features.py first.")
        exit(1)

    print(f"Loading features from {features_path}.safetensors...")
    features_dict = mx.load(f"{features_path}.safetensors")

    print(f"Extracting features from query image: {query_image_path}...")
    query_features = get_query_features(query_image_path)
    query_features = np.array(query_features).reshape(1, -1)

    print("Finding the most similar image...")
    best_match_file, similarity_score = find_most_similar_image(features_dict, query_features)
    print(f"Most similar image: {best_match_file}")
    print(f"Similarity score: {similarity_score:.4f}")
    best_image = Image.open(f"./val2017/{best_match_file}")
    query_search_image = Image.open(query_image_path)
    best_image.show()
    query_search_image.show()
