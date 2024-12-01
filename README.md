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

### 以下、随時実装

AppleのAIMv2でマルチモーダル機能を活用編1「画像領域特徴量の抽出とテキストで画像領域の可視化」の起動コマンド
```commandline
python3 aimv2-large-patch14-224-lit/image_search_from_text_and_show.py 
```

# aimv2-project-for-qiita-article
Apple/aimv2 for-qiita-article
以下に、モデルのサイズが小さい順に並べたリストを示します。名前の中にある情報（`large` → `huge` → `1B` → `3B` など）や入力解像度（`224` → `336` → `448`）を基準にしています。

### サイズが小さい順のモデル
1. **`apple/aimv2-large-patch14-224`** <br>
    [Update at 2024/11/26 aimv2-large-patch14-224の場所→](https://github.com/syun88/aimv2-project-for-qiita-article/tree/main/aimv2-large-patch14-224)
2. **`apple/aimv2-large-patch14-224-distilled`**
3. **`apple/aimv2-large-patch14-224-lit`**<br>
    [Update at 2024/11/29 aimv2-large-patch14-224-litの場所→](https://github.com/syun88/aimv2-project-for-qiita-article/tree/main/aimv2-large-patch14-224-lit)
4. **`apple/aimv2-large-patch14-native`**
5. **`apple/aimv2-large-patch14-336`**
6. **`apple/aimv2-large-patch14-336-distilled`**
7. **`apple/aimv2-large-patch14-448`**
8. **`apple/aimv2-huge-patch14-224`**
9. **`apple/aimv2-huge-patch14-336`**
10. **`apple/aimv2-huge-patch14-448`**
11. **`apple/aimv2-1B-patch14-224`**
12. **`apple/aimv2-1B-patch14-336`**
13. **`apple/aimv2-1B-patch14-448`**
14. **`apple/aimv2-3B-patch14-224`**
15. **`apple/aimv2-3B-patch14-336`**
16. **`apple/aimv2-3B-patch14-448`**

### 基準
- 解像度 (`224 < 336 < 448`) が小さいものを優先。
- モデルサイズ (`large < huge < 1B < 3B`) を優先。