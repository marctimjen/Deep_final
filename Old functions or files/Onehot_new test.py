import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt



img = torch.load("C:/Users/Marc/Desktop/Billeder/onehot/one (6).pt").numpy()

invcol = {0:"everything else", 1:"curb", 2:"road", 3:"cars", 4:"trucks", 5:"motorcycle", 6:"humans"}

plt.imshow(img[1])
plt.show()
