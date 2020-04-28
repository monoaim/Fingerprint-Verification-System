import Helper
import numpy as np
import scipy.ndimage as ndimage

# -----------------------------


class OfDetector:

    def __init__(self):
        self.sobelKernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobelKernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def detect(self, fpImg, mskImg):
        """ Orientation Field Detection: estimate orientations of lines or ridges in an image
        :param fpImg: a fingerprint image (gray-scale) to estimate orientations in
        :param mskImg: a mask image (region-of-interest)
        :returns: An ndarray the same shape as the image, filled with an orientation
                  angles in radians. """

        size = 16  # size
        height, width = fpImg.shape

        # First we smooth the whole image with a Gaussian filter, to make the
        # individual pixel gradients less spurious.
        image = ndimage.filters.gaussian_filter(fpImg, 2.0)

        # Compute the gradients of both at each pixel
        gradientX = Helper.convolve(image, self.sobelKernelX)
        gradientY = Helper.convolve(image, self.sobelKernelY)

        # Estimate the local orientation of each block
        yblocks = height // size
        xblocks = width // size
        o = np.empty((yblocks, xblocks))
        for j in range(yblocks):
            for i in range(xblocks):
                v_x = v_y = 0
                for v in range(size):
                    for u in range(size):
                        v_x += 2 * gradientX[j * size + v, i * size +
                                             u] * gradientY[j * size +
                                                            v, i * size + u]
                        v_y += gradientX[j * size + v, i * size +
                                         u]**2 - gradientY[j * size +
                                                           v, i * size + u]**2

                o[j, i] = 0.5 * np.arctan2(v_x, v_y)

        # Rotate the orientations so that they point along the ridges, and wrap
        # them into only half of the circle (all should be less than 180 degrees).
        o = (o + 0.5 * np.pi) % np.pi

        # Smooth the orientation field
        o_p = np.empty(o.shape)
        o = np.pad(o, 2, mode="edge")
        for y in range(yblocks):
            for x in range(xblocks):
                surrounding = o[y:y + 5, x:x + 5]
                orientation, deviation = Helper.averageOrientation(
                    surrounding, deviation=True)
                if deviation > 0.5:
                    orientation = o[y + 2, x + 2]
                o_p[y, x] = orientation
        o = o_p

        # Make an orientation field the same shape as the input image, and fill it
        # with values interpolated from the preliminary orientation field.
        orientations = np.full(image.shape, -1.0)
        halfsize = size // 2
        for y in range(yblocks - 1):
            for x in range(xblocks - 1):
                for iy in range(size):
                    for ix in range(size):
                        orientations[y * size + halfsize + iy, x * size +
                                     halfsize + ix] = Helper.averageOrientation(
                                         [
                                             o[y, x], o[y + 1, x], o[y, x + 1],
                                             o[y + 1, x + 1]
                                         ], [
                                             iy + ix, size - iy + ix,
                                             iy + size - ix,
                                             size - iy + size - ix
                                         ])
        of = np.where(mskImg == 1.0, orientations, -1.0)
        return of


# -----------------------------
if __name__ == "__main__":
    pass

# -----------------------------
