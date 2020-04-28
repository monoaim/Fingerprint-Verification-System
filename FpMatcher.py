import cv2
import numpy as np
import FpSegmentator
import FpEnhancer
import MnExtractor
import MnMatcher


# -----------------------------
class FpMatcher:

    def __init__(self):
        self.segmentator = FpSegmentator.FpSegmentator(32)
        self.enhancer = FpEnhancer.FpEnhancer()
        self.extractor = MnExtractor.MnExtractor()
        self.matcher = MnMatcher.MnMatcher()

    def getMnSet(self, fpFile):
        # 1) Fingerprint segmentation
        segmentedImg, maskImg = self.segmentator.segment(fpFile)

        # 2) Fingerprint enhancement
        enhImg = self.enhancer.enhance(segmentedImg, maskImg)

        # 3) Minutia extraction
        mnSet = self.extractor.extract(enhImg)

        return mnSet

    def match(self, fpImg1, fpImg2):
        """ Stub - Fingerprint Matching
        Input - a fingerprint image (template)
        Input - a fingerprint image (input)
        Output - similarity score")  #stub """

        mnSet1 = self.getMnSet(fpImg1.astype("float64"))
        mnSet2 = self.getMnSet(fpImg2.astype("float64"))

        lenDists = [len(mnSet1), len(mnSet2)]
        ratio = (min(lenDists) / max(lenDists)) * 100
        if ratio < 70:
            return 0

        # 4) Minutia matching
        # return 0 <= Similarity Score <= 1
        similarity = self.matcher.match(mnSet1, mnSet2)

        # For 0 <= Similarity Score <= 100
        # similarity *= 100
        return similarity


# -----------------------------

if __name__ == "__main__":

    # fingerprint filename
    fpFile1 = '1_1'
    fpFile2 = '1_2'

    # read fingerprint image 1
    fpImg1 = cv2.imread("FP_DB/{}.bmp".format(fpFile1), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("fp1", fpImg1)

    # read fingerprint image 2
    fpImg2 = cv2.imread("FP_DB//{}.bmp".format(fpFile2), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("fp2", fpImg2)

    # match two fingerprint images
    fpMatcher = FpMatcher()
    similarity = fpMatcher.match(fpImg1, fpImg2)
    print("Similary = ", similarity)

    cv2.waitKey()
    cv2.destroyAllWindows()
# -----------------------------
