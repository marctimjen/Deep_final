import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class GTA_hotloader(Dataset):
    """
    This dataloader loads input images and onehot encoded targets.
    """
    def __init__(self, path: str, width: int, height: int, ind: list, device):
        """
        Args:
            path (str): path to the image- and onehot encoded folder.
            width (int): size given to the output.
            height (int): size given to the output.
            ind (list): list of indices for which pictures to load.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = device
        self.imgs_path = path
        self.data = []
        for i in ind:
            self.data.append([self.imgs_path + f"images/Input ({i}).png",
                        self.imgs_path + f"onehot/onehot ({i}).pt"])

        self.img_dim = (width, height)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_in_path, img_tar_path = self.data[idx] # Path for target + input
        img_in = cv2.imread(img_in_path)[:, :, ::-1]
        img_in = cv2.resize(img_in, self.img_dim)
        img_input = torch.from_numpy(img_in)
        img_input = img_input.permute(2, 0, 1)/255
        img_input = img_input.type(torch.float).to(self.device)

        img_target = torch.load(img_tar_path)
        img_target = img_target.type(torch.float).to(self.device)

        return img_input, img_target
