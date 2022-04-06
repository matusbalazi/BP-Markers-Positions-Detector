from math import sqrt, atan2, pi
import numpy as np

class CannyEdgeDetector:
    def __init__(self, pImage):
        self.image = pImage

    def canny_edge_detector(self):
        input_pixels = self.image.load()
        width = self.image.width
        height = self.image.height

        grayscaled = self.compute_grayscale(input_pixels, width, height)

        blurred = self.compute_blur(grayscaled, width, height)

        gradient, direction = self.compute_gradient(blurred, width, height)

        self.filter_out_non_maximum(gradient, direction, width, height)

        keep = self.filter_strong_edges(gradient, width, height, 20, 25)

        return keep

    # transformacia obrazka do odtienov sivej
    def compute_grayscale(self, pInputPixels, pWidth, pHeight):
        grayscale = np.empty((pWidth, pHeight))
        for x in range(pWidth):
            for y in range(pHeight):
                pixel = pInputPixels[x, y]
                grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
        return grayscale

    # redukcia sumu rozostrenim obrazka pomocou Gaussovho filtra
    def compute_blur(self, pInputPixels, pWidth, pHeight):
        # ponechanie mierky povodneho obrazka
        clip = lambda x, l, u: l if x < l else u if x > u else x

        # Gausovsky filter
        kernel = np.array([
            [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
        ])

        # stred Gausovskeho filtra
        offset = len(kernel) // 2

        # rozostrenie obrazka
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

    # vypocet gradientu a jeho smeru
    def compute_gradient(self, pInputPixels, pWidth, pHeight):
        gradient = np.zeros((pWidth, pHeight))
        direction = np.zeros((pWidth, pHeight))
        for x in range(pWidth):
            for y in range(pHeight):
                if 0 < x < pWidth - 1 and 0 < y < pHeight - 1:
                    magx = pInputPixels[x + 1, y] - pInputPixels[x - 1, y]
                    magy = pInputPixels[x, y + 1] - pInputPixels[x, y - 1]
                    gradient[x, y] = sqrt(magx**2 + magy**2)
                    direction[x, y] = atan2(magy, magx)
        return gradient, direction

    # metoda potlacenia pixelov s mensou hodnotou intenzity, ako maju pixely v smere gradientu
    def filter_out_non_maximum(self, pGradient, pDirection, pWidth, pHeight):
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

    # urcenie hran pomocou detekcie prahovych pixelov a transformaciiou slabych pixelov na silne
    def filter_strong_edges(self, pGradient, pWidth, pHeight, pLow, pHigh):
        # ponechanie silnych pixelov
        keep = set()
        for x in range(pWidth):
            for y in range(pHeight):
                if pGradient[x, y] > pHigh:
                    keep.add((x, y))

        # transformacia slabeho pixelu na silny, ale iba vtedy,
        # ak sa v jeho najblizsom okoli nachadza aspon jeden silny pixel
        lastiter = keep
        while lastiter:
            newkeep = set()
            for x, y in lastiter:
                for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if pGradient[x + a, y + b] > pLow and (x+a, y+b) not in keep:
                        newkeep.add((x+a, y+b))
            keep.update(newkeep)
            lastiter = newkeep

        return list(keep)