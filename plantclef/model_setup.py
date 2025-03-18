import os
import requests
from pathlib import Path


def get_model_dir() -> str:
    """
    Get the model directory in the plantclef shared project for the current user on PACE
    """
    # get root directory
    root_dir = Path(__file__).resolve().parent.parent
    # check if model directory exists, create if not
    model_dir = os.path.join(root_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # return model directory
    return model_dir


def setup_fine_tuned_model() -> str:
    """
    Downloads and unzips a model from PACE and returns the path to the specified model file.
    Checks if the model already exists and skips download and extraction if it does.

    :return: Absolute path to the model file.
    """
    model_base_path = get_model_dir()
    tar_filename = "model_best.pth.tar"
    pretrained_model = (
        "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
    )
    relative_model_path = f"pretrained_models/{pretrained_model}/{tar_filename}"
    full_model_path = os.path.join(model_base_path, relative_model_path)

    # Check if the model file exists
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at: {full_model_path}")

    # Return the path to the model file
    return full_model_path


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # ensure the download was successful
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print(f"Downloaded {url} to {dest_path}")


if __name__ == "__main__":
    # get model directory
    model_dir = get_model_dir()
    print("Model directory:", model_dir)

    # get model
    dino_model_path = setup_fine_tuned_model()
    print("Model path:", dino_model_path)
