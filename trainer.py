import neptune.new as neptune
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_Unetpadding import GTA_Unetpadding
from GTApack.GTA_Unet import GTA_Unet
from GTApack.GTA_prop_to_hot import GTA_prop_to_hot
from GTApack.GTA_tester import GTA_tester

from torchvision import datasets, transforms
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import time

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set up the datasets

#val_set, train_set = torch.utils.data.random_split(
#                                [i for i in range(1, 61)], [10, 50],
#                                generator=torch.Generator().manual_seed(42))

val_set, train_set = torch.utils.data.random_split(
                                [i for i in range(1, 2498)], [250, 2247],
                                generator=torch.Generator().manual_seed(42))


#test_set = np.random.randint(low = 1, high = 2478, size = 10)
test_set = np.random.randint(low = 1, high = 2478, size = 500)


valload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/train/",
                        width = 400, height = 300, ind = val_set,
                        device = device)

trainload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/train/",
                        width = 400, height = 300, ind = train_set,
                        device = device)

testload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/test/",
                        width = 400, height = 300, ind = test_set,
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



token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjQ1MjNhYi0zN2YzLTRlZDAtOWExYy1jMjEyMjYxMjhhMmMifQ=="

run = neptune.init(
    project="Deep-Learning-test/Deep-Learning-Test",
    api_token=token,
)  # your credentials


nEpoch = 10





# Network 1

#params = {"learning_rate":0.01, "momentum": 0.9, "optimizer":"SGD",
#          "loss_function":"MSEloss", "model":"GTA_Unet"}
#
#run[f"network1/parameters"] = params
#
#lossFunc = nn.MSELoss()
#model = GTA_Unet(n_channels = 3, n_classes = 10).to(device)
#optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#
#valid_loss, train_loss = [], []
#
#avg_train_loss, avg_valid_loss = [], []
#
#
#for iEpoch in range(nEpoch):
#    print(f"Training epoch {iEpoch}")
#    for img, lab in trainloader:
#        y_pred = model(img)
#        model.zero_grad()
#        loss = lossFunc(y_pred, lab)
#        loss.backward()
#        optimizer.step()
#        train_loss.append(loss.item())
#
#    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
#    run[f"network1/train_loss"].log(w)
#    train_loss = []
#
#    for img, lab in valloader:
#        y_pred = model(img)
#        loss = lossFunc(y_pred, lab)
#        valid_loss.append(loss.item())
#
#    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
#    run[f"network1/validation_loss"].log(w)
#    valid_loss = []
#
#run[f"network1/network_weights"].log(model.state_dict())
#
#
#
#test_acc_per_pic = GTA_tester(model, testloader, 10)
#
#print(np.mean(test_acc_per_pic))
#
#run[f"network1/test_accuracy_per_pic"].log(test_acc_per_pic)
#run[f"network1/mean_test_accuracy"].log(np.mean(test_acc_per_pic))




# Network 2:


params = {"momentum": 0.9, "optimizer":"SGD",
          "loss_function":"MSEloss", "model":"GTA_Unet",
          "scheduler":"ReduceLROnPlateau"}

run[f"network2/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 10).to(device)

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min')

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network2/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network2/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network2/validation_loss"].log(w)

    scheduler.step(w)

    valid_loss = []

run[f"network2/network_weights"].log(model.state_dict())



test_acc_per_pic = GTA_tester(model, testloader, 10)

print(np.mean(test_acc_per_pic))

run[f"network2/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network2/mean_test_accuracy"].log(np.mean(test_acc_per_pic))




# Network 3


#params = {"optimizer":"Adam",
#          "loss_function":"MSEloss", "model":"GTA_Unet",
#          "scheduler":"ReduceLROnPlateau"}
#
#run[f"network3/parameters"] = params
#
#lossFunc = nn.MSELoss()
#model = GTA_Unet(n_channels = 3, n_classes = 10).to(device)
#
#optimizer = Adam(model.parameters(), lr=0.1)
#scheduler = ReduceLROnPlateau(optimizer, 'min')
#
#valid_loss, train_loss = [], []
#
#avg_train_loss, avg_valid_loss = [], []
#
#
#for iEpoch in range(nEpoch):
#    print(f"Training epoch {iEpoch}")
#
#    run[f"network3/learning_rate"].log(optimizer.param_groups[0]['lr'])
#
#    for img, lab in trainloader:
#        y_pred = model(img)
#        model.zero_grad()
#        loss = lossFunc(y_pred, lab)
#        loss.backward()
#        optimizer.step()
#        train_loss.append(loss.item())
#
#    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
#    run[f"network3/train_loss"].log(w)
#    train_loss = []
#
#    for img, lab in valloader:
#        y_pred = model(img)
#        loss = lossFunc(y_pred, lab)
#        valid_loss.append(loss.item())
#
#    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
#    run[f"network3/validation_loss"].log(w)
#
#    scheduler.step(w)
#
#    valid_loss = []
#
#run[f"network3/network_weights"].log(model.state_dict())
#
#
#
#test_acc_per_pic = GTA_tester(model, testloader, 10)
#
#print(np.mean(test_acc_per_pic))
#
#run[f"network3/test_accuracy_per_pic"].log(test_acc_per_pic)
#run[f"network3/mean_test_accuracy"].log(np.mean(test_acc_per_pic))


run.stop()
