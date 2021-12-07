import torch
import matplotlib.pyplot as plt
import numpy as np

def GTA_imageplot(image, target, output = None):
    """
    This function makes a plot of the input and the target (can also be output
    from the network). Note that the input and target can be in tensor format,
    but can't be in a minibatch.

    Args:
        image (tensor): the input image.
        target (tensor): the target in image format.
        output (tensor): the output from the network. Will not be plottet if not
        given.

    Plots:
        Makes a plot with the input on the left and the target and output on the
        right.
    """

    image = image.to("cpu")
    target = target.to("cpu")
    num = 2

    if not(output == None):
        output = output.to("cpu")
        num = 3

    f = plt.figure()
    f.add_subplot(1, num, 1)
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    f.add_subplot(1, num, 2)
    plt.imshow(np.transpose(target.numpy(), (1, 2, 0)))

    if not(output == None):
        f.add_subplot(1, num, 3)
        plt.imshow(np.transpose(target.numpy(), (1, 2, 0)))

    plt.show(block=True)
