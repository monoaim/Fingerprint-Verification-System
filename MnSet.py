from Minutiae import *


class MnSet:

    @staticmethod
    def getNeighbours(x, y, img):
        # Return 8-neighbours of image point P1(x,y) in a clockwise order
        x_1 = x - 1
        y_1 = y - 1
        x1 = x + 1
        y1 = y + 1
        return [
            img[x_1][y_1], img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
            img[x1][y], img[x1][y_1], img[x][y_1]
        ]  # P1, P2, P3, P4, P5, P6, P7, P8

    @staticmethod
    def getCrossingNumber1(neighbours):
        n = neighbours + neighbours[0:1]
        return sum((n1, n2) == (255, 0) for n1, n2 in zip(n, n[1:])
                  )  # (P1,P2), (P2,P3), ... , (P7,P8), (P8,P1)

    @staticmethod
    def getCrossingNumber2(neighbours):
        n = neighbours + neighbours[0:1]
        return 0.5 * sum(
            abs(n1 // 255 - n2 // 255) for n1, n2 in zip(n, n[1:])
        )  # (P1,P2), (P2,P3), ... , (P7,P8), (P8,P1)

    @staticmethod
    def _extract(skeletonedImg):
        mnSet = []
        imgTemp = skeletonedImg.copy()
        rows, cols = imgTemp.shape
        margin = 15
        for x in range(margin,
                       rows - margin):  # for each pixels (except the borders)
            for y in range(margin, cols - margin):
                if skeletonedImg[x, y] == 0:
                    neighbours = MnSet.getNeighbours(
                        x, y, skeletonedImg)  # get neighbours
                    cn = MnSet.getCrossingNumber1(neighbours)
                    cn2 = MnSet.getCrossingNumber2(neighbours)
                    if cn == 1 and cn2 == 1.0:
                        mnSet.append(Minutiae(y, x, M_TYPE_ENDPOINT))
                    elif cn >= 3 and cn2 >= 3.0:
                        mnSet.append(Minutiae(y, x, M_TYPE_BIFURCATION))
        return mnSet
