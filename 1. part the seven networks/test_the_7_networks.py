import neptune.new as neptune
import os
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_Unet import GTA_Unet
from GTApack.GTA_prop_to_hot import GTA_prop_to_hot
from GTApack.GTA_tester import GTA_tester

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


token = os.getenv('Neptune_api')

np.random.seed(42)

run = neptune.init(
    project="Deep-Learning-test/Deep-Learning-Test",
    api_token=token,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_set = np.random.randint(low = 1, high = 1621, size = 200)

dataset = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/test-val/",
                        width = 400, height = 300, ind = test_set,
                        device = device)

batch_size = 1

testloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


for i in range(1, 8):
    print(f"network {i}")
    #net = run[f'network{i}/network_weights'].download()

    model = GTA_Unet(n_channels = 3, n_classes = 9)
    x = torch.load(f"C:/Users/Marc/Desktop/Billeder/params/The seven/network{i}.pt")
    model.load_state_dict(x)
    model.to(device)

    test_acc_per_pic = GTA_tester(model, testloader)
    print(np.mean(test_acc_per_pic))

    run[f"network{i}/test_accuracy_per_pic"].log(test_acc_per_pic)
    run[f"network{i}/mean_test_accuracy"].log(np.mean(test_acc_per_pic))


run.stop()
