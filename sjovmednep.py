import numpy as np
from PIL import ImageGrab
import cv2
import time
from GTApack.GTA_Unet import GTA_Unet
import torch
import neptune.new as neptune
from GTApack.GTA_prop_to_hot import GTA_prop_to_hot
from GTApack.GTA_antihot import GTA_antihot


token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjQ1MjNhYi0zN2YzLTRlZDAtOWExYy1jMjEyMjYxMjhhMmMifQ=="
run = neptune.init(
    project="Deep-Learning-test/Deep-Learning-Test",
    api_token=token,
    run="DEEP-35"
    )

net = run['network2/network_weights'].download()
run.stop()

model = GTA_Unet(n_channels = 3, n_classes = 9)
model.load_state_dict(torch.load("network_weights.pt"))
width = 400
height = 300
img_dim = (width, height)


colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6, (255, 0, 0):6,
            (0, 60, 100):7, (0, 80, 100):8}

# code from:
# https://github.com/Sentdex/pygta5/blob/master/Tutorial%20Codes/Part%201-7/part-1-Getting-Visuals.py

def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode for GTA 5, at the top left position of your main screen.
        # 40 px accounts for title bar.
        printscreen =  np.array(ImageGrab.grab(bbox=(0,25,800,625)))
        print(f'loop took {(w := time.time()) - last_time} seconds')
        last_time = w
        img_resize = np.transpose(cv2.resize(printscreen, img_dim), (2, 0, 1))
        val = torch.from_numpy(np.array([img_resize])).type(torch.float)
        pred = model(val)
        pred = GTA_antihot(GTA_prop_to_hot(pred, 9, width, height), colors, width, height)
        cv2.imshow('window',cv2.cvtColor(np.transpose(pred.detach().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()

# torch.Size([3, 3, 300, 400])
