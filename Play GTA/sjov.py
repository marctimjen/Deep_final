import numpy as np
from PIL import ImageGrab
import cv2
import time
from GTApack.GTA_Unet import GTA_Unet
import torch


model = GTA_Unet(n_channels = 3, n_classes = 3)
model.load_state_dict(torch.load("C:/Users/Marc/Desktop/Billeder/params/GTA_Unet_params.pt"))
img_dim = (400, 300)


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
        cv2.imshow('window',cv2.cvtColor(np.transpose(pred[0].detach().numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


screen_record()

# torch.Size([3, 3, 300, 400])
