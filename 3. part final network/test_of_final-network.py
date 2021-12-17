from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_Unet import GTA_Unet
from GTApack.GTA_tester import GTA_tester

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
import time

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = GTA_Unet(n_channels = 3, n_classes = 9)
model.load_state_dict(torch.load("C:/Users/Marc/Desktop/Billeder/params/final/network.pt"))
model.to(device)

# Set up the datasets
np.random.seed(42)

test_val_set = np.random.randint(low = 1, high = 858, size = 200)

testload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/test-val/",
                        width = 400, height = 300, ind = test_val_set,
                        device = device)

batch_size = 1

testloader = torch.utils.data.DataLoader(testload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


test_acc_per_pic = GTA_tester(model, testloader)
print(np.mean(test_acc_per_pic))
