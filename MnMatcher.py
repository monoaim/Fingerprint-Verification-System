import math
import numpy as np

# -----------------------------


class PointPair:

    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.distance = None
        self.angle = None

    def calculateEuclideanDistance(self):
        self.distance = math.hypot(self.q.getX() - self.p.getX(),
                                   self.q.getY() - self.p.getY())
        return self.distance

    def calculateAngle(self):
        ang1 = np.arctan2(self.q.getY(), self.q.getX())
        ang2 = np.arctan2(self.p.getY(), self.p.getX())
        self.angle = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        return self.angle

    def __eq__(self, o):
        return self.distance == o.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __str__(self):
        return "({}, {})\t\t{:.6f}\t\t{:.6f}".format(self.p, self.q,
                                                     self.distance, self.angle)


class MnMatcher:

    def getEuclideanDistanceForSet(self, forset):
        d = []
        n = len(forset)
        for i in range(n):
            for j in range(i + 1, n):
                pair = PointPair(forset[i], forset[j])
                pair.calculateEuclideanDistance()
                pair.calculateAngle()
                d.append(pair)
        return d

    def _calculateEuclideanDistance(self, mnSet1, mnSet2):
        # calculate the Euclidean distance
        distSet1 = self.getEuclideanDistanceForSet(mnSet1)
        distSet2 = self.getEuclideanDistanceForSet(mnSet2)

        # sort by distance (ascending order)
        distSet1 = sorted(distSet1)
        distSet2 = sorted(distSet2)

        return distSet1, distSet2

    def _recordMatchPoint(self, d1, d2):
        i = j = 1
        matchedSet = []
        m, n = len(d1), len(d2)
        while True:
            if i == m or j == n:
                break
            if d1[i] == d2[j]:
                matchedSet.append((d1[i], d2[j]))
                i += 1
            elif d1[i] > d2[j]:
                j += 1
            elif d1[i] < d2[j]:
                i += 1
        return matchedSet

    def angleBetween(self, p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return (ang1 - ang2) % (2 * np.pi)

    def match(self, mnSet1, mnSet2):
        """ Minutia Matching
        Input - a set of minutiae (template)
        Input - a set of minutiae (input)
        Output - similarity score """

        # 1. calculate the Euclidean distance
        distSet1, distSet2 = self._calculateEuclideanDistance(mnSet1, mnSet2)

        # 2. record the matched points
        matchedSet = self._recordMatchPoint(distSet1, distSet2)

        # 3. select a pair of matched points
        c = 0
        for m in matchedSet:
            # A
            a = PointPair(m[0].p, m[1].p)
            aDist = a.calculateEuclideanDistance()
            aAngle = a.calculateAngle()

            # B
            b = PointPair(m[0].q, m[1].q)
            bDist = b.calculateEuclideanDistance()
            bAngle = b.calculateAngle()

            if aDist == 0 and bDist == 0:
                distRatio = 1
            else:
                dists = [aDist, bDist]
                distRatio = min(dists) / max(dists)
            if distRatio > 0.6:
                c += 1

        lenDists = [len(distSet1), len(distSet2)]
        matches = [len(matchedSet), c]
        matchRatio = min(matches) / max(matches)
        sizeRatio = min(lenDists) / max(lenDists)

        if sizeRatio > 0.6:
            multiplier = 1.0
        else:
            multiplier = 0.3
        similarity = matchRatio * multiplier
        return similarity


# -----------------------------

if __name__ == "__main__":
    pass

# -----------------------------
