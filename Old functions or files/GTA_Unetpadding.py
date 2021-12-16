import torch.nn as nn
import torch

# We have taken inspiration from:
# https://github.com/milesial/Pytorch-UNet

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



class Up_Convolution(nn.Module): # green arrow
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)



class Up_Scale(nn.Module): # Green arrow
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='nearest'),
        self.up_conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        #self.up_conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.doub = Double_Convolution(in_channels, out_channels)

    def forward(self, x, y):
        x = self.up_conv1(x)
        #x = self.up_conv2(x)
        x = self.up(x)
        ind = torch.cat((y, x), 1)
        out = self.doub(ind)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GTA_Unetpadding(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(GTA_Unetpadding, self).__init__()

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
