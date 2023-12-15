import torch
import numpy as np

def mixup_data(x, y, lam, device):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x
