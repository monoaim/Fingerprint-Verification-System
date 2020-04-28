import Helper
import numpy as np

# -----------------------------


class FpSegmentator:

    def __init__(self, bs=32):
        self.blockSize = bs
        self.threshold = 0.1

    def segment(self, fpImg):
        """ Fingerprint segmentation
        input - a fingerprint image
        Output - a segmented image
        Output - a mask image (region-of-interest) """

        segmentedImg = Helper.normalize(fpImg)
        maskImg = np.empty(segmentedImg.shape)
        height, width = segmentedImg.shape
        for y in range(0, height, self.blockSize):
            for x in range(0, width, self.blockSize):
                block = segmentedImg[y:y + self.blockSize, x:x + self.blockSize]
                standardDeviation = np.std(block)
                if standardDeviation < self.threshold or block.shape != (
                        self.blockSize, self.blockSize):
                    maskImg[y:y + self.blockSize, x:x + self.blockSize] = 0.0
                else:
                    maskImg[y:y + self.blockSize, x:x + self.blockSize] = 1.0

        return segmentedImg, maskImg


# -----------------------------
