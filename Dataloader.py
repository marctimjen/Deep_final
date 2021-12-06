import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt

# Imports you will need
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F


# Data from:
# https://download.visinf.tu-darmstadt.de/data/from_games/

class GTA_loader(Dataset):
    def __init__(self):
        self.imgs_path = "C:/Users/Bruger/Desktop/Billeder/images/"
        self.data = []
        for i in range(1, 2501):
            self.data.append([self.imgs_path + f"Input ({i}).png",
                        self.imgs_path + f"Target ({i}).png"])

        self.img_dim = (640, 640)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_in_path, img_tar_path = self.data[idx] # Path for target + input

        img_in = cv2.imread(img_in_path)[:, :, ::-1]
        img_in = cv2.resize(img_in, self.img_dim)
        img_input = torch.from_numpy(img_in)
        img_input = img_input.permute(2, 0, 1)/255
        img_input = img_input.type(torch.float).to(device)

        img_tar = cv2.imread(img_tar_path)[:, :, ::-1]
        img_tar = cv2.resize(img_tar, self.img_dim)
        img_target = torch.from_numpy(img_tar)
        img_target = img_target.permute(2, 0, 1)/255
        img_target = img_target.type(torch.float).to(device)

        return img_input, img_target



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#8
batch_size = 1


# Set up the dataset.
dataset = GTA_loader()



# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
img1, lab1 = dataiter.next()



# Blue : sky [ 70 130 180]
# Dark blue : other vichels  (0, 0, 142)
# Very dark blue : big vicheles 0, 0, 70
# Motor cycle : 0, 0, 230

# Red : humans (220, 20, 60)

# Black : own vichele (0, 0, 0)
# Brown : Trash-cans/fire-hydrants/big signs


# Orange : trafic-lights
# Yellow : Signs
# Gray : Lightposts + electrical things + poles [ 70  70  70]
# Darg gray : buildings
# Purple : road [128, 64, 128]
# Pink : curb + sidewalk/pavement (244, 35, 232)
# Light green : grass
# Dark green : BUSHES




# https://gist.github.com/mstankie/71e49f628beac320953e0460b8ee78c2

colors = ((128, 64, 128), (244, 35, 232), (0, 0, 142), (0, 0, 70), (0, 0, 0))

#for images, labels in zip(img, lab):
#    f = plt.figure()
#    f.add_subplot(1,2, 1)
#    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
#    f.add_subplot(1,2, 2)
#    plt.imshow(np.transpose(labels.numpy(), (1, 2, 0)))
#    plt.show(block=True)
#
#    new = np.transpose(labels.numpy(), (1, 2, 0))
#
#    col = set()
#
#    for i in range(len(new)):
#        for j in range(len(new[i])):
#
#            #if not(set(new[i][j]) in col):
#            #    col.add(tuple(new[i][j]))
#
#            #if all(new[i][j] == np.array([70, 70, 70])):
#            #    new[i][j] = [70, 130, 180]
#
#            if not(tuple(new[i][j]) in colors):
#                new[i][j] = [70, 130, 180]
#
#
#
#    print(col)
#
#    f = plt.figure()
#    f.add_subplot(1,2, 1)
#    plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
#    f.add_subplot(1,2, 2)
#    plt.imshow(new)
#    plt.show(block=True)
#
#    print(np.transpose(labels.numpy(), (1, 2, 0)).shape)


#%% Netork


#import neptune.new as neptune
#
#token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYWVhMmY0Mi05ZGMyLTQ3OGUtYTU5Mi04NzM4YTE0MWI0NTAifQ=="
#
#run = neptune.init(
#    project="andersschultz12/test-project",
#    api_token=token,
#)  # your credentials



class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 3, stride = 1),
            nn.Tanh()
            #nn.AvgPool2d(kernel_size = 2)
        )

        self.decoder_conv = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 6, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(6, 3, kernel_size = 3, stride = 1),
            nn.Tanh()
        )



    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.decoder_conv(x)
        return x


# 800 x 900 -> 798 x 898 -> 399 x 449 -> 397 x 447 ->


#%%


class Double_Convolution(nn.Module): # Blue arrow
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)


class Down_Scale(nn.Module): # red + double_conv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Double_Convolution(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)



#class Up_Convolution(nn.Module): # green arrow
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.double_conv = nn.Sequential(
#            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU()
#        )
#    def forward(self, x):
#        return self.double_conv(x)



class Up_Scale(nn.Module): # Green arrow
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='nearest'),
        self.up_conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        #self.up_conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.doub = Double_Convolution(in_channels, out_channels)

    def forward(self, x, y):
        print(x.shape)
        print(y.shape)
        x = self.up_conv1(x)
        #x = self.up_conv2(x)
        x = self.up(x)
        print(x.shape)
        ind = torch.cat((y, x), 1)
        print(ind.shape)
        out = self.doub(ind)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Double_Convolution(n_channels, 64)
        self.down1 = Down_Scale(64, 128)
        self.down2 = Down_Scale(128, 256)
        self.down3 = Down_Scale(256, 512)
        self.down4 = Down_Scale(512, 1024)
        self.up1 = Up_Scale(1024, 512)
        self.up2 = Up_Scale(512, 256)
        self.up3 = Up_Scale(256, 128)
        self.up4 = Up_Scale(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



lossFunc = nn.MSELoss()
model = UNet(3, 3).to(device)
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
        if i == 10:
            break




y = model(img1)
y = y.to("cpu")
img1 = img1.to("cpu")

f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(np.transpose(img1[0].numpy(), (1, 2, 0)))
f.add_subplot(1,2, 2)
plt.imshow(np.transpose(y[0].detach().numpy(), (1, 2, 0)))
plt.show(block=True)

#run.stop()
