from MnSet import MnSet
from Binarizer import Binarizer
from Skeletonizer import Skeletonizer
import cv2
import imageio

# -----------------------------


class MnExtractor:

    def __init__(self):
        self.binarizer = Binarizer()
        self.skeletonizer = Skeletonizer()

    def extract(self, enhancedImg):
        """ Minutia Extraction
        Input - an enhanced fingerprint image
        Output - a set of minutiae """

        # Binarize
        binImg = self.binarizer.binarize(enhancedImg)

        # save binImg
        imageio.imsave('stepImg/Binarized.png', binImg)
        binImg = cv2.imread("stepImg/Binarized.png", cv2.IMREAD_GRAYSCALE)

        # Skeletonize
        skeletonImg = self.skeletonizer.skeletonize(binImg)

        # save skeletonImg
        imageio.imsave('stepImg/Skeletonized.png', skeletonImg)

        # get set of minutiae
        mnSet = MnSet._extract(skeletonImg)
        return mnSet


# -----------------------------

if __name__ == "__main__":
    pass

# -----------------------------
