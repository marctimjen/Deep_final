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
        #img_input = img_input.type(torch.float).to(device)

        img_tar = cv2.imread(img_tar_path)[:, :, ::-1]
        img_tar = cv2.resize(img_tar, self.img_dim)
        img_target = torch.from_numpy(img_tar)
        #img_target = img_target.permute(2, 0, 1)
        #img_target = img_target.type(torch.float).to(device)

        return img_input, img_target



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


#%%

print(lab1.shape)

#%%
print(img1[0].shape)

print(lab1[0][2].shape)

print(lab1[0][:,0])


#%%
a=torch.empty((3, 800, 900), dtype=torch.int32, device = 'cpu')
print(a.shape)

print(a)

#%%

invcol = {0:"everything else", 1:"curb", 2:"road"}

colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2}


print(colors)

print(len(invcol))
print(colors.get((128, 64, 128), 0))

#%%

print(tuple([1, 2, 3]))

print(1500 % 900)
print(1500 // 800)
#%%

invcol = {0:"everything else", 1:"curb", 2:"road", 3:"cars", 4:"trucks", 5:"motorcycle", 6:"humans"}

colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6}

def onehot(image, dim, col):
    map = torch.empty((len(col), dim[1], dim[0]), dtype=torch.uint8)
    for r in range(dim[1]):
        for j in range(dim[0]):
            num = col.get(tuple(image[r][j].numpy()), 0)
            map[num][r][j] = 1

    return map

val = onehot(lab1, (900, 800), colors)





plt.imshow(val[0])


print(lab1[0].shape)

#%% Onehot the files


colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6}

img_dim = (900, 800)

for i in range(3, 30):
    path = f"C:/Users/Bruger/Desktop/Billeder/images/Target ({i}).png"
    img_tar = cv2.imread(path)[:, :, ::-1]
    img_tar = cv2.resize(img_tar, img_dim)
    img_target = torch.from_numpy(img_tar)

    one = onehot(img_target, (900, 800), colors)
    torch.save(one, f'C:/Users/Bruger/Desktop/Billeder/onehot/one ({i}).pt')



#%% load

onehotaf = torch.load(f'C:/Users/Bruger/Desktop/Billeder/onehot/one (2).pt')
plt.imshow(onehotaf[6])
