from GTApack.GTA_loader import GTA_loader
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_imageplot import GTA_imageplot
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_Unetpadding import GTA_Unetpadding
from GTApack.GTA_Unet import GTA_Unet

from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set up the datasets
val_set, test_set, train_set = torch.utils.data.random_split(
                                [i for i in range(1, 2501)], [500, 500, 1500],
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


train_acc, train_loss = [], []
valid_acc, valid_loss = [], []


lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 3, n_classes = 7).to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)


i = 0
nEpoch = 1
for iEpoch in range(nEpoch):
    for img, lab in trainloader:
        y_pred = model(img)
        model.zero_grad()
        loss = lossFunc(y_pred, lab)
        loss.backward()
        optimizer.step()
        #run["testnet/loss"].log(loss.item())
        i += 1
        print(i)
        print(y_pred)
        print("----")
        print(lab)
        if i == 1:
            break


colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6}


dataiter = iter(trainloader)
images, targets = dataiter.next()

pred = model(images)

GTA_imageplot(GTA_antihot(targets[0].cpu(), colors, 400, 300),
              GTA_antihot(pred[0].cpu().detach(), colors, 400, 300))
