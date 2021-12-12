import neptune.new as neptune
from GTApack.GTA_Unet import GTA_Unet
from GTApack.GTA_hotloader import GTA_hotloader
from GTApack.GTA_imageplot import GTA_imageplot
from GTApack.GTA_antihot import GTA_antihot
from GTApack.GTA_prop_to_hot import GTA_prop_to_hot
import torch
import numpy as np

#from collections import OrderedDict
#import json

token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjQ1MjNhYi0zN2YzLTRlZDAtOWExYy1jMjEyMjYxMjhhMmMifQ=="
run = neptune.init(
    project="Deep-Learning-test/Deep-Learning-Test",
    api_token=token,
    run="DEEP-12"
    )

net = run['network2/network_weights'].download()
print(net)
run.stop()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


test_set = np.random.randint(low = 1, high = 2478, size = 10)

dataset = GTA_hotloader(path = "C:/Users/Marc/Desktop/Billeder/test/", width = 400, height = 300, ind = test_set, device = device)


batch_size = 1
# Set up the dataloader.
testloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

dataiter = iter(testloader)
images, targets = dataiter.next()


model = GTA_Unet(n_channels = 3, n_classes = 9)

model.load_state_dict(torch.load("network_weights.pt"))

model.to(device)

pred = model(images)


colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6, (255, 0, 0):6,
            (0, 60, 100):7, (0, 80, 100):8}



GTA_imageplot(images[0].cpu(), GTA_antihot(targets.cpu()[0], colors, 400, 300),
              GTA_antihot(GTA_prop_to_hot(pred.cpu().detach(), 9, 400, 300), colors, 400, 300))





#optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
#lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.1)
#lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.3, step_size_up=1, step_size_down=3)
