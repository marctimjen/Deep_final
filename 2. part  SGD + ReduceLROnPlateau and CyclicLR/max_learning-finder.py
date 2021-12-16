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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set up the datasets


np.random.seed(42)

val_set, train_set = torch.utils.data.random_split(
                            np.random.randint(low = 1, high = 4962, size = 500),
                            [60, 440],
                            generator=torch.Generator().manual_seed(42))

test_val_set = np.random.randint(low = 1, high = 856, size = 200)


valload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/train/",
                        width = 400, height = 300, ind = val_set,
                        device = device)

trainload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/train/",
                        width = 400, height = 300, ind = train_set,
                        device = device)

testload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/test-val/",
                        width = 400, height = 300, ind = test_val_set,
                        device = device)

batch_size = 1

# Set up the dataloaders:
valloader = torch.utils.data.DataLoader(valload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

trainloader = torch.utils.data.DataLoader(trainload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

testloader = torch.utils.data.DataLoader(testload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

token = os.getenv('Neptune_api')

run = neptune.init(
    project="Deep-Learning-test/Deep-Learning-Test",
    api_token=token,
)


nEpoch = 41


params = {"optimizer":"SGD", "optimizer_momentum": 0.9,
          "optimizer_learning_rate": 1, "loss_function":"MSEloss",
          "model":"GTA_Unet", "scheduler":"CyclicLR",
          "scheduler_base_lr":1, "scheduler_max_lr":40,
          "scheduler_step_size_up":40}

run[f"network/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = SGD(model.parameters(), lr=1, momentum=0.9)
scheduler = CyclicLR(optimizer, base_lr=1, max_lr=40, step_size_up=40)

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    val_acc_per_pic = np.mean(GTA_tester(model, valloader, p = False))
    run[f"network/validation_mean_acc"].log(val_acc_per_pic)

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network/validation_loss"].log(w)

    scheduler.step()

    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/find_learning/network2.pt")
run[f"network/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/find_learning/network2.pt"))


test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network/mean_test_accuracy"].log(np.mean(test_acc_per_pic))

run.stop()
