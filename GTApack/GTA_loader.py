import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt



class GTA_loader(Dataset):
    """
    This data-loader reads images and targets. In this format neither the target
    nor the input are onehot encoded.
    """
    def __init__(self, path: str, width: int, height: int, device):
        """
        Args:
            path (str): path to the images and targets.
            width (int): size given to the output.
            height (int): size given to the output.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.

        Return:
            tulpe: containing the input and target in tensor format
        """

        self.device = device
        self.imgs_path = path
        self.data = []
        for i in range(1, 2501):
            self.data.append([self.imgs_path + f"Input ({i}).png",
                        self.imgs_path + f"Target ({i}).png"])

        self.img_dim = (width, height)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_in_path, img_tar_path = self.data[idx] # Path for target + input

        img_in = cv2.imread(img_in_path)[:, :, ::-1]
        img_in = cv2.resize(img_in, self.img_dim)
        img_input = torch.from_numpy(img_in)
        img_input = img_input.permute(2, 0, 1)/255
        img_input = img_input.type(torch.float).to(self.device)

        img_tar = cv2.imread(img_tar_path)[:, :, ::-1]
        img_tar = cv2.resize(img_tar, self.img_dim)
        img_target = torch.from_numpy(img_tar)
        img_target = img_target.permute(2, 0, 1)/255
        img_target = img_target.type(torch.float).to(self.device)

        return img_input, img_target
