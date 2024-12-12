import os
import sys
import json
import uuid
import tomllib
from typing import Optional

import loguru
from PIL import Image
import numpy as np
from fastapi import FastAPI, Request, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import searcher
from handld_features import model_loader, feature_loader

logger = loguru.logger
logger.remove()
logger.add(sys.stdout, format="{time} - {level} - ({extra[request_id]}) {message} ", level="DEBUG")

# FastAPIのインスタンス作成
app = FastAPI(title="Floor Plan Search AI", description="This is sample of Floor Plan Search AI.")
# セキュアな通信を行うための設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("config.toml", mode="rb") as f:
    app.state.config = tomllib.load(f)

def path_generator(config: dict) -> tuple:
    """
    Generate path for model and processor and features
    """

    cache_dir_path = config["cache_dir_local"]
    if not os.path.exists(cache_dir_path):
        cache_dir_path = config["cache_dir_cloud"]

    model_file_path = os.path.join(
        cache_dir_path,
        config["model_file"]
    )

    processor_file_path = os.path.join(
        cache_dir_path,
        config["processor_file"]
    )

    features_file_path = os.path.join(
        config["features_dir_local"],
        config["features_file"]
    )
    if not os.path.exists(features_file_path):
        features_file_path = os.path.join(
            config["features_dir_cloud"],
            config["features_file"]
        )

    return cache_dir_path, model_file_path, processor_file_path, features_file_path

cache_dir_path, model_file_path, processor_file_path, features_file_path = path_generator(app.state.config)

# load for model and processor
app.state.model, app.state.processor, app.state.device = model_loader(cache_dir_path, model_file_path, processor_file_path)
app.state.features_dict = feature_loader(features_file_path)


@app.middleware("http")
async def request_middleware(request, call_next):
    request_id = str(uuid.uuid4())
    with logger.contextualize(request_id=request_id):
        logger.info("Request started")
        request.state.request_id = request_id

        try:
            response = await call_next(request)

        except Exception as ex:
            logger.opt(exception=True).error(f"Request failed: {ex}")
            response = JSONResponse(content={"success": False}, status_code=500)

        finally:
            response.headers["X-Request-ID"] = request_id
            logger.info("Request ended")
            return response


@app.get("/reload-model")
async def reload(request: Request):
    with open("config.toml", mode="rb") as f:
        app.state.config = tomllib.load(f)
    cache_dir_path, model_file_path, processor_file_path, _ = path_generator(app.state.config)
    app.state.model, app.state.processor, app.state.device = model_loader(cache_dir_path, model_file_path, processor_file_path)
    return {"success": True}

@app.get("/reload-features")
async def reload(request: Request):
    with open("config.toml", mode="rb") as f:
        app.state.config = tomllib.load(f)
    _, _, _, features_file_path = path_generator(app.state.config)
    app.state.features_dict = feature_loader(features_file_path)
    return {"success": True}

@app.post("/search")
# async def search(request: Request, upload_file: UploadFile, top_n: Optional[int]=app.state.config["top_n"]):
async def search(request: Request, post_file: bytes = File(...), top_n: Optional[bytes] = File(...)):
    logger.info("Request received")

    top_n = int(top_n.decode())
    logger.info(f"Request top_n: {top_n}")
    logger.info(f"Request len(post_file): {len(post_file)}")

    results = searcher.search(app.state.model, app.state.processor, app.state.device, app.state.features_dict, post_file, top_n)
    return {"success": True, "results": json.dumps(results)}
    return {"success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))