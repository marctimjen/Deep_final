import cv2
import numpy as np

invcol = {0:"everything else", 1:"curb", 2:"road", 3:"cars", 4:"trucks",
            5:"motorcycle", 6:"humans"}




colors = [(0, 0, 0), (244, 35, 232), (128, 64, 128), (0, 0, 142),
            (0, 0, 70), (0, 0, 230), (220, 20, 60)]

def samecolor(image, width: int, height: int, col: dict):
    map = np.zeros((height, width, 3), dtype=int)
    # make empty tensor

    for r in range(height):
        for j in range(width):
            if not((tuple(image[r][j]) in col)):
                image[r][j] = [0, 0, 0]

    return image


# Now we can onehot encode some targets:
img_dim = (400, 300)

for i in range(1001, 2501):
    path = f"C:/Users/Marc/Desktop/Billeder/images/Target ({i}).png"
    img_tar = cv2.imread(path)[:, :, ::-1]
    img_tar = cv2.resize(img_tar, img_dim)

    img = samecolor(image = img_tar, width = 400, height = 300, col = colors)
    cv2.imwrite(f'C:/Users/Marc/Desktop/Billeder/samecolor/same ({i}).png',
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    #https://stackoverflow.com/questions/42406338/why-cv2-imwrite-changes-the-color-of-pics

    print(i)
