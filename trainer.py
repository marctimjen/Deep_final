import neptune.new as neptune
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_Unetpadding import GTA_Unetpadding
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

#val_set, train_set = torch.utils.data.random_split(
#                                [i for i in range(1, 7)], [1, 5],
#                                generator=torch.Generator().manual_seed(42))

#val_set, train_set = torch.utils.data.random_split(
#                                [i for i in range(1, 2498)], [250, 2247],
#                                generator=torch.Generator().manual_seed(42))

np.random.seed(42)

val_set, train_set = torch.utils.data.random_split(
                            np.random.randint(low = 1, high = 4962, size = 500),
                            [60, 440],
                            generator=torch.Generator().manual_seed(42))

test_set = np.random.randint(low = 1, high = 2478, size = 100)

#test_set = np.random.randint(low = 1, high = 2478, size = 500)

#test_set = np.random.randint(low = 1, high = 2478, size = 100)


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


nEpoch = 50




# Network 1

params = {"optimizer":"SGD", "optimizer_learning_rate":0.01,
          "optimizer_momentum": 0.9, "loss_function":"MSEloss",
          "model":"GTA_Unet"}

run[f"network1/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)


valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")
    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network1/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network1/validation_loss"].log(w)
    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network1.pt")
run[f"network1/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network1.pt"))

test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network1/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network1/mean_test_accuracy"].log(np.mean(test_acc_per_pic))




# Network 2:


params = {"optimizer":"SGD", "optimizer_momentum": 0.9,
          "optimizer_learning_rate": 0.1, "loss_function":"MSEloss",
          "model":"GTA_Unet", "scheduler":"ReduceLROnPlateau",
          "scheduler_patience":3, "scheduler_threshold":0.01}

run[f"network2/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=0.01)

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

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network2.pt")
run[f"network2/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network2.pt"))


test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network2/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network2/mean_test_accuracy"].log(np.mean(test_acc_per_pic))




# Network 3


params = {"optimizer":"Adam", "optimizer_learning_rate": 0.1,
          "loss_function":"MSEloss", "model":"GTA_Unet",
          "scheduler":"ReduceLROnPlateau", "scheduler_patience":3,
          "scheduler_threshold":0.01}

run[f"network3/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=0.01)

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network3/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network3/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network3/validation_loss"].log(w)

    scheduler.step(w)

    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network3.pt")
run[f"network3/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network3.pt"))

test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network3/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network3/mean_test_accuracy"].log(np.mean(test_acc_per_pic))




# Network 4

params = {"optimizer":"SGD", "optimizer_momentum": 0.9,
          "optimizer_learning_rate": 0.1, "loss_function":"MSEloss",
          "model":"GTA_Unet", "scheduler":"CyclicLR",
          "scheduler_base_lr":0.01, "scheduler_max_lr":0.1,
          "scheduler_step_size_up":10}

run[f"network4/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=10)

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network4/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network4/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network4/validation_loss"].log(w)

    scheduler.step()

    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network4.pt")
run[f"network4/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network4.pt"))


test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network4/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network4/mean_test_accuracy"].log(np.mean(test_acc_per_pic))



# network 5

params = {"optimizer":"Adam", "optimizer_learning_rate": 0.1,
          "loss_function":"MSEloss", "model":"GTA_Unet", "scheduler":"CyclicLR",
          "scheduler_base_lr":0.01, "scheduler_max_lr":0.1,
          "scheduler_step_size_up":10, "scheduler_cycle_momentum":"False"}

run[f"network5/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, cycle_momentum = False)

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network5/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network5/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network5/validation_loss"].log(w)

    scheduler.step()

    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network5.pt")
run[f"network5/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network5.pt"))


test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network5/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network5/mean_test_accuracy"].log(np.mean(test_acc_per_pic))



# network 6


params = {"optimizer":"SGD", "optimizer_learning_rate": 0.1,
          "loss_function":"MSEloss", "model":"GTA_Unet", "scheduler":"CyclicLR",
          "scheduler_base_lr":0.01, "scheduler_max_lr":0.1,
          "scheduler_step_size_up":1, "scheduler_step_size_down":4}

run[f"network6/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = SGD(model.parameters(), 0.1)
scheduler = CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=1, step_size_down=4)

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network6/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network6/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network6/validation_loss"].log(w)

    scheduler.step()

    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network6.pt")
run[f"network6/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network6.pt"))


test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network6/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network6/mean_test_accuracy"].log(np.mean(test_acc_per_pic))





# Network 7


params = {"optimizer1":"SGD", "optimizer1_learning_rate": 0.5,
          "loss_function":"MSEloss", "model":"GTA_Unet",
          "scheduler1":"CosineAnnealingLR",
          "scheduler1_T_max":10, "scheduler1_eta_min":0.1,
          "optimizer2":"SGD", "optimizer2_learning_rate": 0.1,
          "scheduler2":"CosineAnnealingLR",
          "scheduler2_T_max":10, "scheduler2_eta_min":0.01,}

run[f"network7/parameters"] = params

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 9).to(device)

optimizer = SGD(model.parameters(), 0.5)
scheduler = CosineAnnealingLR(optimizer, T_max = 10, eta_min = 0.1)

valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    run[f"network7/learning_rate"].log(optimizer.param_groups[0]['lr'])

    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network7/train_loss"].log(w)
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network7/validation_loss"].log(w)

    if iEpoch == 30:
        optimizer = SGD(model.parameters(), 0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max = 10, eta_min = 0.01)


    scheduler.step()

    valid_loss = []

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/network7.pt")
run[f"network7/network_weights"].upload(File("C:/Users/Marc/Desktop/Billeder/params/network7.pt"))


test_acc_per_pic = GTA_tester(model, testloader)

print(np.mean(test_acc_per_pic))

run[f"network7/test_accuracy_per_pic"].log(test_acc_per_pic)
run[f"network7/mean_test_accuracy"].log(np.mean(test_acc_per_pic))


run.stop()
