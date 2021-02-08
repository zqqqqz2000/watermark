import torch
from typing import *
import random
import torch.nn.functional as F


def rand_cut(img: torch.Tensor) -> torch.Tensor:
    b, c, h, w = img.shape
    rand_y_start = random.randint(0, h // 4)
    rand_y_end = random.randint(h // 1.5, h)
    rand_x_start = random.randint(0, w // 4)
    rand_x_end = random.randint(w // 1.5, w)
    cut = img[:, :, rand_y_start: rand_y_end, rand_x_start: rand_x_end]
    resized_tensor = F.interpolate(cut, size=h)
    return resized_tensor


def transition(img: torch.Tensor) -> List[torch.Tensor]:
    flip_x = img.flip(2)
    flip_y = img.flip(3)
    rand_cut_img = rand_cut(img)
    return [img, flip_x, flip_y, rand_cut_img]
