import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__}; device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))
    return device
