import cv2
import torch
import numpy as np


def GTA_antihot(onehot_img, colors: dict, width: int, height: int):
    """
    This function takes a onehot encoded input and turns it into a image with
    three channels.

    Args:
        onehot_img (tensor): onehot encoded image.
        col (dict): a mapping of the differnt layers to different colors.
        width (int): width of the given image.
        height (int): height of the given image.

    Return:
        Tensor: An image with 3 channels.
    """
    inv_colors = {v: k for k, v in colors.items()}
    mat = onehot_img.numpy()

    map = torch.empty((3, height, width), dtype=torch.uint8)
    for i in range(0, len(set(colors.values()))):
        col = inv_colors.get(i)
        indicies = np.where(mat[i] == 1)
        for r, c in zip(*indicies):
            map[0][r][c] = col[0]
            map[1][r][c] = col[1]
            map[2][r][c] = col[2]

    return map
