import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt

# Imports you will need
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F


# Data from:
# https://download.visinf.tu-darmstadt.de/data/from_games/

class GTA_loader(Dataset):
    def __init__(self):
        self.imgs_path = "C:/Users/Marc/Desktop/Billeder/images/"
        self.data = []
        for i in range(1, 2501):
            self.data.append([self.imgs_path + f"Input ({i}).png",
                        self.imgs_path + f"Target ({i}).png"])

        self.img_dim = (640, 640)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_in_path, img_tar_path = self.data[idx] # Path for target + input

        img_in = cv2.imread(img_in_path)[:, :, ::-1]
        img_in = cv2.resize(img_in, self.img_dim)
        img_input = torch.from_numpy(img_in)
        img_input = img_input.permute(2, 0, 1)/255
        img_input = img_input.type(torch.float).to(device)

        img_tar = cv2.imread(img_tar_path)[:, :, ::-1]
        img_tar = cv2.resize(img_tar, self.img_dim)
        img_target = torch.from_numpy(img_tar)
        img_target = img_target.permute(2, 0, 1)/255
        img_target = img_target.type(torch.float).to(device)

        return img_input, img_target



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#8
batch_size = 1


# Set up the dataset.
dataset = GTA_loader()



# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
img1, lab1 = dataiter.next()

print(img1)

# Blue : sky [ 70 130 180]
# Dark blue : other vichels  (0, 0, 142)
# Very dark blue : big vicheles 0, 0, 70
# Motor cycle : 0, 0, 230
# cyan : busses 0, 60, 100
# Train + tram blue: 0, 80, 100



# Red : humans (220, 20, 60)  -
# another red: motorcyclelists (255, 0, 0)

# Black : own vichele (0, 0, 0)
# Brown : Trash-cans/fire-hydrants/big signs


# Orange : trafic-lights
# Yellow : Signs
# Gray : Lightposts + electrical things + poles [ 70  70  70]
# Darg gray : buildings
# Purple : road [128, 64, 128]
# Pink : curb + sidewalk/pavement (244, 35, 232)
# Light green : grass
# Dark green : BUSHES


# 608, 909, 910 might be deleted



# https://gist.github.com/mstankie/71e49f628beac320953e0460b8ee78c2

colors = ((128, 64, 128), (244, 35, 232), (0, 0, 142), (0, 0, 70), (0, 0, 0))

#for images, labels in zip(img, lab):
#    f = plt.figure()
#    f.add_subplot(1,2, 1)
#    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
#    f.add_subplot(1,2, 2)
#    plt.imshow(np.transpose(labels.numpy(), (1, 2, 0)))
#    plt.show(block=True)
#
#    new = np.transpose(labels.numpy(), (1, 2, 0))
#
#    col = set()
#
#    for i in range(len(new)):
#        for j in range(len(new[i])):
#
#            #if not(set(new[i][j]) in col):
#            #    col.add(tuple(new[i][j]))
#
#            #if all(new[i][j] == np.array([70, 70, 70])):
#            #    new[i][j] = [70, 130, 180]
#
#            if not(tuple(new[i][j]) in colors):
#                new[i][j] = [70, 130, 180]
#
#
#
#    print(col)
#
#    f = plt.figure()
#    f.add_subplot(1,2, 1)
#    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
#    f.add_subplot(1,2, 2)
#    plt.imshow(new)
#    plt.show(block=True)
#
#    print(np.transpose(labels.numpy(), (1, 2, 0)).shape)
