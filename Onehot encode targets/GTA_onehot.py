import torch
import cv2




invcol = {0:"everything else", 1:"curb", 2:"road", 3:"cars", 4:"trucks",
            5:"motorcycle", 6:"humans", 7:"busses", 8:"tram/train"}

colors = {(0, 0, 0):0, (244, 35, 232):1, (128, 64, 128):2, (0, 0, 142):3,
            (0, 0, 70):4, (0, 0, 230):5, (220, 20, 60):6, (255, 0, 0):6,
            (0, 60, 100):7, (0, 80, 100):8}


def onehot(image, width: int, height: int, col: dict):
    """
    This function takes the target pictures and onehot encode them. The amount
    of classes are defined by the amount of different classes in the col
    dictionary.

    Args:
        image (tensor): target image meant to be onehot encoded.
        width (int): width of the given image.
        height (int): height of the given image.
        col (dict): a mapping of the different colors to differnt layers.

    Return:
        Tensor: A onehot encoded version of the target image.
    """
    map = torch.zeros((len(set(col.values())), height, width), dtype=torch.uint8)
    # make empty tensor

    for r in range(height):
        for j in range(width):
            num = col.get(tuple(image[r][j].numpy()), 0)
            map[num][r][j] = 1

    return map


# Now we can onehot encode some targets:
img_dim = (400, 300)

for i in range(1, 856):
    path = f"C:/Users/Marc/Desktop/Billeder/test-val/images/Target ({i}).png"
    img_tar = cv2.imread(path)[:, :, ::-1]
    img_tar = cv2.resize(img_tar, img_dim)
    img_target = torch.from_numpy(img_tar)

    one = onehot(image = img_target, width = 400, height = 300, col = colors)

    torch.save(one, f'C:/Users/Marc/Desktop/Billeder/test-val/onehot/onehot ({i}).pt')

    print(i)
