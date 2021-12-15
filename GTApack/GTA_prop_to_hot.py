import torch

def GTA_prop_to_hot(img, n_classes: int, width: int, height: int):
    """
    This function turns the output of the network (given in probability format)
    into the most likely onehot encoded output.

    Args:
        img (tensor): The tensor with probabilities.
        n_classes (int): Amount of classes in the onehot encoded format.
        width (int): Size width of the given tensor.
        height (int): Size height of the given tensor.

    Returns:
        Tensor: In onehot encoded format.
    """
    mat = torch.argmax(img, dim=1)
    map = torch.zeros((n_classes, height, width), dtype=torch.uint8)

    for r in range(height):
        for c in range(width):
            map[mat[0][r][c]][r][c] = 1

    return map
