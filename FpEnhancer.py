import Helper
import numpy as np
from OfDetector import OfDetector
from GaborFilterbank import GaborFilterbank

# -----------------------------


class FpEnhancer:

    def enhance(self, fpImg, mskImg):
        """ Fingerprint Enhancement
        Input - a fingerprint image (gray-scale)
        Input - a mask image (region-of-interest)
        Output - an enhanced image """

        img = np.where(mskImg == 1.0, Helper.localNormalize(fpImg), fpImg)

        # get an orientation field
        orientationField = OfDetector().detect(img, mskImg)

        # filter an image with orientation field
        enhImg = GaborFilterbank().filter(img, orientationField, mskImg)
        return enhImg


# -----------------------------
if __name__ == "__main__":
    pass

# -----------------------------
