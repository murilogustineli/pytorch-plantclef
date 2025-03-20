import torch
from pathlib import Path


def get_torch_version():
    return torch.__version__


def get_device():
    """Get the device (CPU, GPU, or XPU) available for PyTorch."""
    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_base_dir() -> str:
    """Get the base project directory."""
    return Path(__file__).resolve().parent.parent


def get_class_mappings_file() -> str:
    """Get the directory containing the class mappings for the DINOv2 model."""
    base_dir = get_base_dir()
    return f"{base_dir}/plantclef/class_mapping.txt"


if __name__ == "__main__":
    # get root directory
    base_dir = get_base_dir()
    print("Base directory:", base_dir)

    # get class mappings file
    class_mappings_file = get_class_mappings_file()
    print("Class mappings file:", class_mappings_file)
