import os
import requests
from pathlib import Path


def get_model_dir() -> str:
    """
    Get the model directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/models"


def get_scratch_model_dir() -> str:
    """
    Get the model directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/scratch/plantclef/models"


def setup_fine_tuned_model(
    scratch_model: bool = True,
    use_only_classifier: bool = False,
    ensure_model_exists: bool = False,
):
    """
    Downloads and unzips a model from PACE and returns the path to the specified model file.
    Checks if the model already exists and skips download and extraction if it does.

    :return: Absolute path to the model file.
    """
    # get directory for model
    if scratch_model:
        model_base_path = get_scratch_model_dir()
    else:
        model_base_path = get_model_dir()
    tar_filename = "model_best.pth.tar"
    pretrained_model = (
        "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
    )
    if use_only_classifier:
        pretrained_model = "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier"
    relative_model_path = f"pretrained_models/{pretrained_model}/{tar_filename}"
    full_model_path = os.path.join(model_base_path, relative_model_path)

    # Check if the model file exists
    if not os.path.exists(full_model_path) and ensure_model_exists:
        raise FileNotFoundError(f"Model file not found at: {full_model_path}")

    # Return the path to the model file
    return full_model_path


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the download was successful
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print(f"Downloaded {url} to {dest_path}")


if __name__ == "__main__":
    # Get model
    dino_model_path = setup_fine_tuned_model()
    print("Model path:", dino_model_path)
