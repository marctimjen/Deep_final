#%% For the GTA_loader

from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_imageplot import GTA_imageplot
from GTApack.GTA_antihot import GTA_antihot
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set up the dataset.
#dataset = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/images/", width = 400, height = 300, device = device)

val_set, train_set = torch.utils.data.random_split([i for i in range(0, 1200)], [120, 1080],
                                    generator=torch.Generator().manual_seed(42))


dataset = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/", width = 400, height = 300, ind = train_set, device = device)


#print([i for i in val_set])

batch_size = 1
# Set up the dataloader.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
images, targets = dataiter.next()


#GTA_imageplot(images[0], targets[0], targets[0])


colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6}


targets = GTA_antihot(targets[0].cpu(), colors, 400, 300)

print(targets.shape)

GTA_imageplot(images[0], targets)

#%% For the onehot encoder








#test!

#rom GTApack.GTA_Unetpadding import GTA_Unetpadding
#
#from GTApack.GTA_Unet import GTA_Unet
#from torchvision import datasets, transforms
#from torch.optim import SGD
#from torch.utils.data import DataLoader, random_split
#import torch.nn.functional as F
#import torch.nn as nn
#
#
#lossFunc = nn.MSELoss()
#model = GTA_Unet(3, 3).to(device)
#optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#i = 0
#nEpoch = 1
#for iEpoch in range(nEpoch):
#    for img, lab in trainloader:
#        y_pred = model(img)
#        model.zero_grad()
#        loss = lossFunc(y_pred, lab)
#        loss.backward()
#        optimizer.step()
#        #run["testnet/loss"].log(loss.item())
#        i += 1
#        print(i)
#        if i == 250:
#            break
#
#
#pred = model(images)
#
#GTA_imageplot(images[0], targets[0], pred[0].detach())
