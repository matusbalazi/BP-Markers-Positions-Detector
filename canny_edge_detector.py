from math import sqrt, atan2, pi
import numpy as np


class CannyEdgeDetector:
    # Initialization
    def __init__(self, pImage):
        self.image = pImage

    # Canny Edge Detector algorithm cleans the image
    # and only keeps the strongest edges
    def applyCannyEdgeDetector(self):
        inputPixels = self.image.load()
        width = self.image.width
        height = self.image.height

        # Image is converted to grayscale
        grayscaledImg = self.convertImageToGrayscale(inputPixels, width, height)

        # Image is blurred to remove noise
        blurredImg = self.blurImage(grayscaledImg, width, height)

        # Gradient and its direction is calculated
        gradient, direction = self.calculateGradient(blurredImg, width, height)

        # Non-maximum suppression is applicated
        self.nonMaximumSuppression(gradient, direction, width, height)

        # Some edges, which not suited requirements are filtered out
        keepEdges = self.applyThresholdToFilterEdges(gradient, width, height, 20, 25)

        return keepEdges

    # Transforms the image to grayscale
    def convertImageToGrayscale(self, pInputPixels, pWidth, pHeight):
        grayscale = np.empty((pWidth, pHeight))
        for x in range(pWidth):
            for y in range(pHeight):
                pixel = pInputPixels[x, y]
                grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
        return grayscale

    # Reduces noise by blurring and smoothing
    # the image using a Gaussian filter
    def blurImage(self, pInputPixels, pWidth, pHeight):

        # Keeps coordinates inside the image
        clip = lambda x, l, u: l if x < l else u if x > u else x

        # Gaussian filter
        kernel = np.array([
            [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256]
        ])

        # Middle of the Gaussian filter
        offset = len(kernel) // 2

        # Blurs the image to remove some unwanted noise
        blurred = np.empty((pWidth, pHeight))
        for x in range(pWidth):
            for y in range(pHeight):
                acc = 0
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = clip(x + a - offset, 0, pWidth - 1)
                        yn = clip(y + b - offset, 0, pHeight - 1)
                        acc += pInputPixels[xn, yn] * kernel[a, b]
                blurred[x, y] = int(acc)
        return blurred

    # Calculates image gradient and its direction to identify the edges
    def calculateGradient(self, pInputPixels, pWidth, pHeight):
        gradient = np.zeros((pWidth, pHeight))
        direction = np.zeros((pWidth, pHeight))
        for x in range(pWidth):
            for y in range(pHeight):
                if 0 < x < pWidth - 1 and 0 < y < pHeight - 1:
                    magx = pInputPixels[x + 1, y] - pInputPixels[x - 1, y]
                    magy = pInputPixels[x, y + 1] - pInputPixels[x, y - 1]
                    gradient[x, y] = sqrt(magx ** 2 + magy ** 2)
                    direction[x, y] = atan2(magy, magx)
        return gradient, direction

    # Keeps only the pixels that have the maximum intensity among
    # their neighbors in the direction of gradient, as a result
    # are thinner and more accurate edges
    def nonMaximumSuppression(self, pGradient, pDirection, pWidth, pHeight):
        for x in range(1, pWidth - 1):
            for y in range(1, pHeight - 1):
                angle = pDirection[x, y] if pDirection[x, y] >= 0 else pDirection[x, y] + pi
                rangle = round(angle / (pi / 4))
                mag = pGradient[x, y]
                if ((rangle == 0 or rangle == 4) and (pGradient[x - 1, y] > mag or pGradient[x + 1, y] > mag)
                        or (rangle == 1 and (pGradient[x - 1, y - 1] > mag or pGradient[x + 1, y + 1] > mag))
                        or (rangle == 2 and (pGradient[x, y - 1] > mag or pGradient[x, y + 1] > mag))
                        or (rangle == 3 and (pGradient[x + 1, y - 1] > mag or pGradient[x - 1, y + 1] > mag))):
                    pGradient[x, y] = 0

    # Edge determination by threshold pixel detection,
    # strong pixels are retained and some weak pixels
    # are transformed into strong ones
    def applyThresholdToFilterEdges(self, pGradient, pWidth, pHeight, pLow, pHigh):

        # Keeping strong pixels
        keep = set()
        for x in range(pWidth):
            for y in range(pHeight):
                if pGradient[x, y] > pHigh:
                    keep.add((x, y))

        # Transforms a weak pixel to a strong one, but only if there
        # is at least one strong pixel in its immediate vicinity
        lastiter = keep
        while lastiter:
            newkeep = set()
            for x, y in lastiter:
                for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if pGradient[x + a, y + b] > pLow and (x + a, y + b) not in keep:
                        newkeep.add((x + a, y + b))
            keep.update(newkeep)
            lastiter = newkeep

        return list(keep)
