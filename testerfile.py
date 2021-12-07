#%% For the GTA_loader

from GTApack.GTA_loader import GTA_loader
from GTApack.GTA_imageplot import GTA_imageplot
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print(type(device))

# Set up the dataset.
dataset = GTA_loader(path = "C:/Users/Marc/Desktop/Billeder/images/", width = 900, height = 800, device = device)



batch_size = 1
# Set up the dataloader.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
images, targets = dataiter.next()



GTA_imageplot(images[0], targets[0], targets[0])



#%% For the onehot encoder
