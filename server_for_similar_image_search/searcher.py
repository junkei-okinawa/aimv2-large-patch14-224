import sys
import json
from typing import List, Tuple

import loguru
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import handld_features

logger = loguru.logger
logger.remove()
logger.add(sys.stdout, format="{time} - {level} - ({extra[request_id]}) {message} ", level="DEBUG")

# 類似性計算
def find_top_n_similar_image(
        query_features,
        features_dict: dict,
        top_n: int,
    ) -> List[Tuple[str, float]]:
    list_top_n: List[Tuple[str | None, float]] = [(None, -1) for i in range(top_n)]
    query_features = query_features.numpy()

    for img_file, img_features in features_dict.items():
        manage_no = img_file.split("/")[-1].split(".")[0]
        img_features = img_features.numpy()
        similarity = cosine_similarity(query_features, img_features)[0, 0]
        similarity =  int(similarity * 1000000)

        if len(list_top_n) < top_n:
            list_top_n.append((manage_no, similarity))
            list_top_n.sort(key=lambda x: x[1], reverse=True)
            continue

        if similarity > list_top_n[-1][1]:
            list_top_n[-1] = (manage_no, similarity)
            list_top_n.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"list_top_n: {list_top_n}")
    return list_top_n

def search(
        model,
        processor,
        device: str,
        features_dict: dict,
        image_bytes: bytes,
        top_n: int,
    ) -> List[Tuple[str, float]]:
    query_features = handld_features.get_query_features(model, processor, device, image_bytes)
    query_features = query_features["features"]  # 辞書から取り出す
    return find_top_n_similar_image(query_features, features_dict, top_n)
