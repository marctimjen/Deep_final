import torch
import numpy as np

def GTA_tester(model, testloader, p = True):
    """
    This function calculates the accuarcy of a model (network) given an input
    and target. Note that this testor only works for the networks with onehot
    encoding.

    Args:
        model: The model (network) we want to test the accuarcy for.
        testloader (generator): The testloader is a generator object that
        deliver the input and target.
        p (bool): This value tells the function if it should print the progress.

    Returns:
        list: With the means of the pixel-accuarcy per images.
    """
    count = 0
    test_acc_per_pix = []

    for img, lab in testloader:
        if p:
            count += 1
            print("test count:", count)

        with torch.no_grad():
            y_pred = model(img)

            for i in range(lab.shape[0]):
                test_acc_per_pix.append(np.mean(
                (torch.argmax(y_pred[i].cpu().detach(), dim = 0) ==
                torch.argmax(lab[i].cpu(), dim = 0)).numpy()))

    return test_acc_per_pix
