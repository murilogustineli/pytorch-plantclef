import torch


def get_torch_version():
    return torch.__version__


def get_device():
    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device
