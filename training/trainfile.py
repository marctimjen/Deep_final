from GTApack.GTA_loader import GTA_loader
from GTApack.GTA_imageplot import GTA_imageplot
import torch

# Set the device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The batc_size:
batch_size = 1

# Set up the dataloader.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


dataiter = iter(trainloader)
images, targets = dataiter.next()


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
