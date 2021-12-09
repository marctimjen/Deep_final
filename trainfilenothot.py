from GTApack.GTA_loader import GTA_loader
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


valload = GTA_loader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = val_set, device = device)

testload = GTA_loader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = test_set, device = device)

trainload = GTA_loader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400,
                        height = 300, ind = train_set, device = device)





batch_size = 3

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
model = GTA_Unet(n_channels = 3, n_classes = 3).to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)


i = 0
nEpoch = 2
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

torch.save(model.state_dict(), "C:/Users/Marc/Desktop/Billeder/params/GTA_Unet_params.pt")

dataiter = iter(trainloader)
images, targets = dataiter.next()

pred = model(images)

GTA_imageplot(targets[0].cpu(), pred[0].cpu().detach())
