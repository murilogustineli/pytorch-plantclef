#!/bin/bash

set -e # exit on error

# define base directory
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ==============================
# Download Dataset
# ==============================
DATASET_URL="https://drive.google.com/uc?id=1bw6OcjNaqMv62zKX2elQvVaAfHLTt3JN"
DATASET_NAME="subset_top5_train"
DATASET_DIR="$BASE_DIR/data/parquet"
DATASET_ZIP="$DATASET_DIR/$DATASET_NAME.zip"

# create dataset directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# check if dataset is already downloaded
if [ -d "$DATASET_DIR" ] && [ "$(ls -A "$DATASET_DIR")" ]; then
    echo "Dataset already exists in $DATASET_DIR. Skipping download."
else
    # download dataset
    echo "Downloading dataset..."
    gdown "$DATASET_URL" -O "$DATASET_ZIP"

    # unzip dataset
    echo "Extracting dataset..."
    unzip -q "$DATASET_ZIP" -d "$DATASET_DIR"

    # remove zip file
    rm "$DATASET_ZIP"

    echo "Dataset path: $DATASET_DIR"
    echo "Dataset download and extraction complete!"
fi

# ==============================
# Download Model
# ==============================
MODEL_URL="https://drive.google.com/uc?id=1R1SUgzVqvEYQgdjvaRc2r3jji6k3oraS"
MODEL_NAME="vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
MODEL_DIR="$BASE_DIR/model/pretrained_models"
MODEL_ZIP="$MODEL_DIR/$MODEL_NAME.zip"

# create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# check if model is already downloaded
if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR")" ]; then
    echo "Model already exists in $MODEL_DIR. Skipping download."
else
    # download model
    echo "Downloading model..."
    gdown "$MODEL_URL" -O "$MODEL_ZIP"

    # unzip model
    echo "Extracting model..."
    unzip -q "$MODEL_ZIP" -d "$MODEL_DIR"

    # remove zip file
    rm "$MODEL_ZIP"

    echo "Model path: $MODEL_DIR"
    echo "Model download and extraction complete!"
fi

echo "All downloads and extractions complete!"
