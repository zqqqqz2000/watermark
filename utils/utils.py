import torch


def denormalize(img: torch.Tensor):
    img_GT = img * 0.5 + 0.5
    return img_GT.detach().cpu().numpy()[::-1] * 255
