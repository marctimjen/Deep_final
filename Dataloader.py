import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt



# Data from:
# https://download.visinf.tu-darmstadt.de/data/from_games/


class GTA_loader(Dataset):
    def __init__(self):
        self.imgs_path = "C:/Users/Bruger/Desktop/Billeder/images/"
        self.data = []
        for i in range(1, 2501):
            self.data.append([self.imgs_path + f"Input ({i}).png",
                        self.imgs_path + f"Target ({i}).png"])

        self.img_dim = (900, 800)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_in_path, img_tar_path = self.data[idx] # Path for target + input
        img_in = cv2.imread(img_in_path)[:, :, ::-1]
        img_in = cv2.resize(img_in, self.img_dim)
        img_input = torch.from_numpy(img_in)
        img_input = img_input.permute(2, 0, 1)


        img_tar = cv2.imread(img_tar_path)[:, :, ::-1]
        img_tar = cv2.resize(img_tar, self.img_dim)
        img_target = torch.from_numpy(img_tar)
        img_target = img_target.permute(2, 0, 1)

        return img_input, img_target




batch_size = 4


# Set up the dataset.
dataset = GTA_loader()



# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
img, lab = dataiter.next()


# Blue : sky
# Dark blue : other vichels
# Very dark blue : big vicheles

# Red : humans

# Black : own vichele
# Brown : Trash-cans/fire-hydrants/big signs


# Orange : trafic-lights
# Yellow : Signs
# Gray : Lightposts + electrical things + poles
# Darg gray : buildings
# Purple : road
# Pink : curb + sidewalk/pavement
# Light green : grass
# Dark green : BUSHES



# https://gist.github.com/mstankie/71e49f628beac320953e0460b8ee78c2


for images, labels in zip(img, lab):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
    f.add_subplot(1,2, 2)
    plt.imshow(np.transpose(labels.numpy(), (1, 2, 0)))
    plt.show(block=True)
