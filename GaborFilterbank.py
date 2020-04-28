from GaborFilter import *
import Helper
import numpy as np

# -----------------------------


class GaborFilterbank:

    def __init__(self):
        self.blockSize = 32

    def filter(self, fpImg, of, mskImg):
        """ Gabor filtering
        Input - a fingerprint image (gray-scale)
        Input - an orientation field (from OfDetector)
        Input - a mask image (region-of-interest)
        Output - a filtered image """

        frequencies = np.where(mskImg == 1.0,
                               GaborFilter.findFrequency(fpImg, of), -1.0)
        result = np.empty(fpImg.shape)
        height, width = fpImg.shape
        for y in range(0, height - self.blockSize, self.blockSize):
            for x in range(0, width - self.blockSize, self.blockSize):
                orientation = of[y + self.blockSize // 2, x +
                                 self.blockSize // 2]
                frequency = Helper.averageFrequency(
                    frequencies[y:y + self.blockSize, x:x + self.blockSize])

                if frequency < 0.0:
                    result[y:y + self.blockSize, x:x +
                           self.blockSize] = fpImg[y:y + self.blockSize, x:x +
                                                   self.blockSize]
                    continue

                result[y:y + self.blockSize, x:x +
                       self.blockSize] = Helper.convolve(
                           fpImg,
                           GaborFilter.gaborKernel(16, orientation, frequency),
                           (y, x), (self.blockSize, self.blockSize))

        filteredImg = Helper.normalize(result)
        return filteredImg


# -----------------------------
if __name__ == "__main__":
    pass

# -----------------------------
