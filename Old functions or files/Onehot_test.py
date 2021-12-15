import cv2
import torch
import numpy as np
from GTApack.GTA_imageplot import GTA_imageplot

en = torch.load("C:/Users/Marc/Desktop/Billeder/onehot/one (10).pt")



colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6}

inv_colors = {v: k for k, v in colors.items()}

mat = en.numpy()

map = torch.empty((3, 300, 400), dtype=torch.uint8)
for i in range(0, len(colors)):
    col = inv_colors.get(i)
    indicies = np.where(mat[i] == 1)
    for r, c in zip(*indicies):
        map[0][r][c] = col[0]
        map[1][r][c] = col[1]
        map[2][r][c] = col[2]

print(map)
GTA_imageplot(map, map)
