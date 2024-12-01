# APPLE 公式さGITHUB readme Link 
[APPLE GITHUB README](https://github.com/apple/ml-aim/blob/main/README.md)

## syun88 さんのコードの mlx 版です
refarence: [AppleのAIMv2で画像特徴量抽出しcocodatasetの画像セットで類似画像検索に挑戦](https://qiita.com/syun88/items/50c1d60d1516d5816773)

## setup
uv を使用しています
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc
```

### clone repo
```sh
git clone https://github.com/junkei-okinawa/aimv2-large-patch14-224.git
cd aimv2-large-patch14-224
```

### create env and install packages
```sh
uv sync
```

### dowonload val2017
```sh
./get_coco_files.sh
``` 

### run
create features safetensors
```sh
uv run prepare_features.py
```

image search
```sh
uv run main.py
```

image search and open files
```sh
uv run image_search.py
```