import numpy as np

# -----------------------------


class Binarizer:

    def binarize(self, fpImg):
        """ Binarization
        Input - a fingerprint image (gray-scale)
        Output - a binary image """

        blockSize = 32
        binImg = np.copy(fpImg)
        height, width = binImg.shape
        for y in range(0, height, blockSize):
            for x in range(0, width, blockSize):
                block = binImg[y:y + blockSize, x:x + blockSize]
                threshold = np.average(block)
                binImg[y:y + blockSize, x:x + blockSize] = np.where(
                    block >= threshold, 1.0, 0.0)
        return binImg


# -----------------------------

if __name__ == "__main__":
    pass

# -----------------------------
