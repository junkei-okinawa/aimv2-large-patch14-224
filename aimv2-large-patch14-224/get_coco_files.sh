# This script downloads and extracts the COCO 2017 validation dataset.
# It performs the following steps:
# 1. Downloads the val2017.zip file from the COCO dataset website.
# 2. Extracts the contents of the downloaded ZIP file.
# 3. Deletes the ZIP file after extraction to save space.

curl http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
unzip val2017.zip
rm val2017.zip