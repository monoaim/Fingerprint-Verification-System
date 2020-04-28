import numpy as np
import scipy.ndimage as ndimage


def normalize(image):
    image = np.copy(image)
    image -= np.min(image)
    m = np.max(image)
    if m > 0.0:
        image *= 1.0 / m
    return image


def localNormalize(image, w=32):
    image = np.copy(image)
    height, width = image.shape
    for y in range(0, height, w):
        for x in range(0, width, w):
            image[y:y + w, x:x + w] = normalize(image[y:y + w, x:x + w])
    return image


def kernelFromFunction(size, f):
    """
    Creates a kernel of the given size, populated with values obtained by
    calling the given function.

    :param size: The desired size of the kernel.
    :param f:    The function.
    :returns:    The created kernel.
    """

    kernel = np.empty((size, size))
    for i in range(0, size):
        for j in range(0, size):
            kernel[i, j] = f(i - size / 2, j - size / 2)
    return kernel


def convolve(image, kernel, origin=None, shape=None, pad=True):
    """
    Apply a kernel to an image or to a part of an image.

    :param image:   The source image.
    :param kernel:  The kernel (an ndarray of black and white, or grayvalues).
    :param origin:  The origin of the part of the image to be convolved.
                    Defaults to (0, 0).
    :param shape:   The shape of the part of the image that is to be convolved.
                    Defaults to the shape of the image.
    :param pad:     Whether the image should be padded before applying the
                    kernel. Passing False here will cause indexing errors if
                    the kernel is applied at the edge of the image.
    :returns:       The resulting image.
    """
    if not origin:
        origin = (0, 0)

    if not shape:
        shape = (image.shape[0] - origin[0], image.shape[1] - origin[1])

    result = np.empty(shape)

    if callable(kernel):
        k = kernel(0, 0)
    else:
        k = kernel

    kernelOrigin = (-k.shape[0] // 2, -k.shape[1] // 2)
    kernelShape = k.shape

    topPadding = 0
    leftPadding = 0

    if pad:
        topPadding = max(0, -(origin[0] + kernelOrigin[0]))
        leftPadding = max(0, -(origin[1] + kernelOrigin[1]))
        bottomPadding = max(
            0, (origin[0] + shape[0] + kernelOrigin[0] + kernelShape[0]) -
            image.shape[0])
        rightPadding = max(
            0, (origin[1] + shape[1] + kernelOrigin[1] + kernelShape[1]) -
            image.shape[1])

        padding = (topPadding, bottomPadding), (leftPadding, rightPadding)

        if np.max(padding) > 0.0:
            image = np.pad(image, padding, mode="edge")

    for y in range(shape[0]):
        for x in range(shape[1]):
            iy = topPadding + origin[0] + y + kernelOrigin[0]
            ix = leftPadding + origin[1] + x + kernelOrigin[1]

            block = image[iy:iy + kernelShape[0], ix:ix + kernelShape[1]]
            if callable(kernel):
                result[y, x] = np.sum(block * kernel(y, x))
            else:
                result[y, x] = np.sum(block * kernel)

    return result


def averageOrientation(orientations, weights=None, deviation=False):
    """
    Calculate the average orientation in an orientation field.
    """

    orientations = np.asarray(orientations).flatten()
    o = orientations[0]

    aligned = np.where(
        np.absolute(orientations - o) > np.pi * 0.5,
        np.where(orientations > o, orientations - np.pi, orientations + np.pi),
        orientations)
    if deviation:
        return np.average(aligned, weights=weights) % np.pi, np.std(aligned)
    else:
        return np.average(aligned, weights=weights) % np.pi


def averageFrequency(frequencies):
    """
    Calculate the average frequency in a frequency field.
    """

    frequencies = frequencies[np.where(frequencies >= 0.0)]
    if frequencies.size == 0:
        return -1
    return np.average(frequencies)


def rotateAndCrop(image, angle):
    """
    Rotate an image and crop the result so that there are no black borders.

    :param image: The image to rotate.
    :param angle: The angle in gradians.
    :returns: The rotated and cropped image.
    """
    h, w = image.shape

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a,
                                                                 x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (
            h * cos_a - w * sin_a) / cos_2a

    image = ndimage.interpolation.rotate(
        image, np.degrees(angle), reshape=False)

    hr, wr = int(hr), int(wr)
    y, x = (h - hr) // 2, (w - wr) // 2
    return image[y:y + hr, x:x + wr]
