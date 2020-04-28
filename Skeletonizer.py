import cv2
import numpy as np
from skimage.morphology import skeletonize

# -----------------------------


class Skeletonizer:

    def skeletonize(self, binImg):
        """ Skeletonization
        Input - a binary image
        Output - a skeleton image """

        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(opening, (1, 1), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th[th == 255] = 1
        skel = skeletonize(th)
        skeletonImg = np.ones(skel.shape, np.uint8)
        width, height = skel.shape
        for i in range(width):
            for j in range(height):
                if skel[i][j]:
                    skeletonImg[i][j] = 0
                else:
                    skeletonImg[i][j] = 255
        return skeletonImg


# -----------------------------

if __name__ == "__main__":
    pass

# -----------------------------
