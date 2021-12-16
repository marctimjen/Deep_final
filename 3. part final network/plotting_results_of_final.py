from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_Unet import GTA_Unet
from GTApack.GTA_prop_to_hot import GTA_prop_to_hot
from GTApack.GTA_tester import GTA_tester
from GTApack.GTA_imageplot import GTA_imageplot

from torchvision import datasets, transforms
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (ReduceLROnPlateau, CyclicLR,
                                     CosineAnnealingLR)
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import time
from neptune.new.types import File
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

np.random.seed(23)
test_set = np.random.randint(low = 1, high = 2255, size = 1)

dataset = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/test/",
    width = 400, height = 300, ind = test_set, device = device)


batch_size = 1

testloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

model = GTA_Unet(n_channels = 3, n_classes = 9)
model.load_state_dict(torch.load("C:/Users/Marc/Desktop/Billeder/params/final/network.pt"))
model.to(device)

dataiter = iter(testloader)
images, targets = dataiter.next()

colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6, (255, 0, 0):6,
            (0, 60, 100):7, (0, 80, 100):8}

width = 400
height = 300

pred = model(images)
pred = GTA_antihot(GTA_prop_to_hot(pred, 9, width, height), colors, width, height)
targets = GTA_antihot(targets[0].cpu(), colors, 400, 300)
GTA_imageplot(images[0], targets, pred.detach().cpu())
