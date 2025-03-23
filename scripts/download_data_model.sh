#!/bin/bash

set -e # exit on error

# define base directory
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ==============================
# Dataset URLs and Paths
# ==============================
TRAIN_DATA_URL="https://drive.google.com/uc?id=1550hDVupH7H6C5D4bm8LEe1S6U6ofmHM"
TEST_DATA_URL="https://drive.google.com/uc?id=1D4QheXF2yagLYrYvCG3aLz4YDFJsSph5"
TRAIN_DATASET_NAME="train_pytorch_webinar_filtered"
TEST_DATASET_NAME="test_2025_pytorch_webinar"
DATA_DIR="$BASE_DIR/data/parquet"

TRAIN_DATA_ZIP="$DATA_DIR/$TRAIN_DATASET_NAME.zip"
TEST_DATA_ZIP="$DATA_DIR/$TEST_DATASET_NAME.zip"

TRAIN_DATA_DIR="$DATA_DIR/$TRAIN_DATASET_NAME"
TEST_DATA_DIR="$DATA_DIR/$TEST_DATASET_NAME"

# Create dataset directory if it doesn't exist
mkdir -p "$DATA_DIR"

# ==============================
# Download and Extract Train Dataset
# ==============================
if [ -d "$TRAIN_DATA_DIR" ] && [ "$(ls -A "$TRAIN_DATA_DIR")" ]; then
    echo "Train dataset already exists in $TRAIN_DATA_DIR. Skipping download."
else
    echo "Downloading Train dataset..."
    gdown "$TRAIN_DATA_URL" -O "$TRAIN_DATA_ZIP"

    echo "Extracting Train dataset..."
    unzip -q "$TRAIN_DATA_ZIP" -d "$DATA_DIR"

    echo "Removing Train dataset zip..."
    rm "$TRAIN_DATA_ZIP"

    echo "Train dataset is ready at: $TRAIN_DATA_DIR"
fi

# ==============================
# Download and Extract Test Dataset
# ==============================
if [ -d "$TEST_DATA_DIR" ] && [ "$(ls -A "$TEST_DATA_DIR")" ]; then
    echo "Test dataset already exists in $TEST_DATA_DIR. Skipping download."
else
    echo "Downloading Test dataset..."
    gdown "$TEST_DATA_URL" -O "$TEST_DATA_ZIP"

    echo "Extracting Test dataset..."
    unzip -q "$TEST_DATA_ZIP" -d "$DATA_DIR"

    echo "Removing Test dataset zip..."
    rm "$TEST_DATA_ZIP"

    echo "Test dataset is ready at: $TEST_DATA_DIR"
fi

echo "All datasets are ready!"

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
