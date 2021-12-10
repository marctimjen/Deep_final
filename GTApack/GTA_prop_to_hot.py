import torch

def GTA_prop_to_hot(img, n_classes, width, height):
    mat = torch.argmax(img, dim=1)
    map = torch.zeros((n_classes, height, width), dtype=torch.uint8)

    for r in range(height):
        for c in range(width):
            map[mat[0][r][c]][r][c] = 1

    return map
