#%% For the GTA_loader

from GTApack.GTA_loader import GTA_loader
from GTApack.GTA_imageplot import GTA_imageplot
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set up the dataset.
dataset = GTA_loader(path = "C:/Users/Marc/Desktop/Billeder/images/", width = 320, height = 320, device = device)



batch_size = 1
# Set up the dataloader.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
images, targets = dataiter.next()

#GTA_imageplot(images[0], targets[0], targets[0])



#%% For the onehot encoder


#test!
from GTApack.GTA_Unetpadding import GTA_Unetpadding

from GTApack.GTA_Unet import GTA_Unet
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn


lossFunc = nn.MSELoss()
model = GTA_Unet(3, 3).to(device)
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
        if i == 250:
            break


pred = model(images)

GTA_imageplot(images[0], targets[0], pred[0].detach())
