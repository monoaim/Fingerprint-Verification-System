import Helper
import numpy as np
import scipy.signal as signal
from warnings import simplefilter

# -----------------------------


class GaborFilter:

    @staticmethod
    def gaborKernel(size, angle, frequency):
        angle += 0.5 * np.pi
        cos = np.cos(angle)
        sin = -np.sin(angle)

        yangle = lambda x, y: x * cos + y * sin
        xangle = lambda x, y: -x * sin + y * cos

        xsigma = ysigma = 4

        kernel = Helper.kernelFromFunction(size, lambda x, y: np.exp(-((xangle(x, y) ** 2) / (xsigma ** 2) + \
                                                                       (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) * \
                                                              np.cos(2 * np.pi * frequency * xangle(x, y)))
        return kernel

    @staticmethod
    def findFrequency(img, orientationfield, blocksize=32):
        """ Estimate ridge or line frequencies in an image, given an orientation field.
        :param img: The image to estimate orientations in.
        :param orientationfield: An orientation field such as one returned from the
                             estimateOrientations() function.
        :param blocksize: The block size.
        :returns: An ndarray the same shape as the image, filled with frequencies. """

        height, width = img.shape
        yblocks, xblocks = height // blocksize, width // blocksize
        F = np.empty((yblocks, xblocks))
        for y in range(yblocks):
            for x in range(xblocks):
                halfsize = blocksize // 2
                orientation = orientationfield[y * blocksize +
                                               halfsize, x * blocksize +
                                               halfsize]
                block = img[y * blocksize:(y + 1) * blocksize, x *
                            blocksize:(x + 1) * blocksize]
                block = Helper.rotateAndCrop(block, np.pi * 0.5 + orientation)
                if block.size == 0:
                    F[y, x] = -1
                    continue
                columns = np.sum(block, (0,))
                columns = Helper.normalize(columns)

                simplefilter("ignore",
                             FutureWarning)  # just ignore FutureWarning
                peaks = signal.find_peaks_cwt(columns, np.array([3]))

                if len(peaks) < 2:
                    F[y, x] = -1
                else:
                    f = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                    if f < 5 or f > 15:
                        F[y, x] = -1
                    else:
                        F[y, x] = 1 / f

        frequencies = np.full(img.shape, -1.0)
        F = np.pad(F, 1, mode="edge")
        for y in range(yblocks):
            for x in range(xblocks):
                surrounding = F[y:y + 3, x:x + 3]
                surrounding = surrounding[np.where(surrounding >= 0.0)]
                if surrounding.size == 0:
                    frequencies[y * blocksize:(y + 1) * blocksize, x *
                                blocksize:(x + 1) * blocksize] = -1
                else:
                    frequencies[y * blocksize:(y + 1) * blocksize, x *
                                blocksize:(x + 1) *
                                blocksize] = np.median(surrounding)

        return frequencies


# -----------------------------
if __name__ == "__main__":
    pass

# -----------------------------
