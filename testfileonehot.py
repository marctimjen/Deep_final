from GTApack.GTA_loader import GTA_loader
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_imageplot import GTA_imageplot
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_Unetpadding import GTA_Unetpadding
from GTApack.GTA_Unet import GTA_Unet
from GTApack.GTA_prop_to_hot import GTA_prop_to_hot

from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import time

import matplotlib.pyplot as plt


device = "cpu"
print(device)

model = GTA_Unet(n_channels = 3, n_classes = 7)
model.load_state_dict(torch.load("C:/Users/Marc/Desktop/Billeder/params/onehot/GTA_Unet_params.pt"))

val_set, test_set, train_set = torch.utils.data.random_split(
                                [i for i in range(1, 251)], [25, 50, 175],
                                generator=torch.Generator().manual_seed(42))

testload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = test_set, device = device)


batch_size = 1

testloader = torch.utils.data.DataLoader(testload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

# Test accuarcy:
test_acc_per_map = []

count = 0

for img, lab in testloader:
    count += 1
    print(count)
    with torch.no_grad():
        y_pred = model(img)

        for i in range(lab.shape[0]):
            test_acc_per_map.append(np.mean((GTA_prop_to_hot(y_pred.cpu().detach(), 7, 400, 300) == lab[i].cpu()).numpy()))


print(test_acc_per_map)
print(np.mean(test_acc_per_map))
