from GTApack.GTA_loader import GTA_loader
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_imageplot import GTA_imageplot
from GTApack.GTA_antihot import GTA_antihot
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set up the datasets
val_set, test_set, train_set = torch.utils.data.random_split(
                                [i for i in range(1, 251)], [25, 50, 175],
                                generator=torch.Generator().manual_seed(42))


valload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = val_set, device = device)

testload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = test_set, device = device)

trainload = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = train_set, device = device)



batch_size = 1

# Set up the dataloaders:
valloader = torch.utils.data.DataLoader(valload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

testloader = torch.utils.data.DataLoader(testload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

trainloader = torch.utils.data.DataLoader(trainload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)



valid_loss, train_loss = [], []

avg_train_loss, avg_valid_loss = [], []


lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 7).to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)


start = time.time()

i = 0
nEpoch = 2
for iEpoch in range(nEpoch):
    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        #run["testnet/loss"].log(loss.item())
        i += 1
        print(i)

    avg_train_loss.append(np.mean(np.array(train_loss)))
    train_loss = []

    for img, lab in valloader:
        y_pred = model(img)
        loss = lossFunc(y_pred, lab)
        valid_loss.append(loss.item())

    avg_valid_loss.append(np.mean(np.array(valid_loss)))
    valid_loss = []

end = time.time()

print(end - start)


torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/onehot/GTA_Unet_params.pt")


plt.plot([i for i in range(0, len(avg_train_loss))], avg_train_loss)
plt.show()


plt.plot([i for i in range(0, len(avg_valid_loss))], avg_valid_loss)
plt.show()

#print(y_pred[0][0][0][0])
#print(y_pred[0][1][0][0])
#print(y_pred[0][2][0][0])
#print(y_pred[0][3][0][0])
#print(y_pred[0][4][0][0])
#print(y_pred[0][5][0][0])
#print(y_pred[0][6][0][0])
#print(sum(y_pred[0][i][0][0] for i in range(0, 7)))
#print("----")
#print(lab)

colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6}


dataiter = iter(trainloader)
images, targets = dataiter.next()

pred = model(images)


GTA_imageplot(images[0].cpu(), GTA_antihot(targets.cpu()[0], colors, 400, 300),
              GTA_antihot(GTA_prop_to_hot(pred.cpu().detach(), 7, 400, 300), colors, 400, 300))
