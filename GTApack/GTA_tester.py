from GTApack.GTA_prop_to_hot import GTA_prop_to_hot
import torch
import numpy as np

def GTA_tester(model, testloader, n_classes):
    count = 0
    test_acc_per_pic = []

    for img, lab in testloader:
        count += 1
        print("test count:", count)
        with torch.no_grad():
            y_pred = model(img)

            for i in range(lab.shape[0]):
                test_acc_per_pic.append(np.mean(
                (torch.argmax(
                GTA_prop_to_hot(y_pred.cpu().detach(), n_classes, 400, 300), dim = 1) ==
                torch.argmax(lab[i].cpu(), dim = 1)).numpy()))

    return test_acc_per_pic
