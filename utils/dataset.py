import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np


class Mp4DataSet(Dataset):
    def __init__(self, mp4_name: str, img_resize=64):
        self.mp4_name = mp4_name
        self.data = []
        self.transform = transforms.Compose(
            [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cap = cv2.VideoCapture(mp4_name)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            resized_img: np.ndarray = cv2.resize(frame, (img_resize, img_resize))
            torch_img = self.transform(resized_img)
            self.data.append(torch_img)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
