# Apple aimv2 API server for similar image search

## 1. Settings

### 1-1. install [uv](https://docs.astral.sh/uv/getting-started/installation/)
An extremely fast Python package and project manager, written in Rust.
- macOS and linux
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc
```

- Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
Add-Content -Path $PROFILE -Value '(& uv generate-shell-completion powershell) | Out-String | Invoke-Expression'
Add-Content -Path $PROFILE -Value '(& uvx --generate-shell-completion powershell) | Out-String | Invoke-Expression'
```

### 1-2. clone repo
```bash
git clone https://github.com/junkei-okinawa/aimv2-large-patch14-224.git
```

### 1-3. change directory
```bash
cd aimv2-large-patch14-224/server_for_similar_image_search
```

### 1-4. create python environment and install packages
```sh
uv sync
```

## 2. Run dev server

### local server
```bash
uv run server.py
# INFO:     Started server process [xxxxx]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Run Docker server
```bash
docker build -f ./Dockerfile -t image_searcher .
docker run -p 8080:8080 -e PORT=8080 -itd image_searcher
# => => naming to docker.io/library/image_searcher                                                                                            0.0s
# <docker container id>
```

### Post image data
Example
```python
import os
import json
import tempfile
import requests
from PIL import Image

url = "http://127.0.0.1:8080"
str_image_path = "any/image/path"
obj_img = Image.open(str_image_path)
x, y = obj_img.size
if x * y > 336 * 336:
    suffix = os.path.splitext(os.path.basename(str_image_path))[1]
    img_format = obj_img.format
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=suffix) as f:
        obj_img.resize((336,336)).save(f, format=obj_img.format, quality=100)
        obj_img = open(f.name, "rb")
else:
    obj_img = open(str_image_path, "rb")

top_n = 10
files = {"post_file": obj_img, "top_n": top_n}
r = requests.post(f"{url}/search", files=files)
results = json.loads(r.json()["results"])

for res in results:
    print(res[0], ":", res[1])
# output example
# image_file_name_0 : 995860
# image_file_name_1 : 995454
# image_file_name_2 : 995413
# image_file_name_3 : 995409
# image_file_name_4 : 995404
# image_file_name_5 : 995402
# image_file_name_6 : 995402
# image_file_name_7 : 995201
# image_file_name_8 : 994999
# image_file_name_9 : 994990
```
