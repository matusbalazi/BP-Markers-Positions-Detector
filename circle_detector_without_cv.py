import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
import math
import matplotlib
import canny_edge_detector
import distance_calculator
from collections import defaultdict

class CircleDetectorWithoutCV:
    def __init__(self, pImageFilename, pMinRadius, pMaxRadius, pObjSize):
        self.image = Image.open(pImageFilename)
        self.minRadius = pMinRadius
        self.maxRadius = pMaxRadius
        self.objSize = pObjSize
        self.outputImage = Image.new("RGB", self.image.size)
        self.outputImage.paste(self.image)
        self.drawResult = ImageDraw.Draw(self.outputImage)
        self.steps = 100
        self.threshold = 0.4
        self.canny = canny_edge_detector.CannyEdgeDetector(self.image)
        self.distance = distance_calculator.DistanceCalculator(self.objSize)

    def detectCircles(self):
        points = []
        for r in range(self.minRadius, self.maxRadius + 1):
            for t in range(self.steps):
                points.append((r, int(r * cos(2 * pi * t / self.steps)), int(r * sin(2 * pi * t / self.steps))))

        acc = defaultdict(int)
        for x, y in self.canny.canny_edge_detector():
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1

        plt.imshow(self.image)
        plt.show()

        circles = []
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / self.steps >= self.threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                print(v / self.steps, x, y, r)
                circles.append((x, y, r))

        return circles

    def findAllCircles(self):
        listOfCircles = self.detectCircles()
        listOfCoordsX = []
        listOfCoordsY = []
        listOfRadii = []
        i = 0
        for x, y, r in listOfCircles:
            i += 1
            self.drawResult.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))
            self.drawResult.text((x, y), "{:d}.".format(i), (0, 0, 0))
            listOfCoordsX.append(x)
            listOfCoordsY.append(y)
            listOfRadii.append(r)


        self.distance.findAllDistances(listOfCoordsX, listOfCoordsY, listOfRadii, self.drawResult, 2)

        nameOfImage = "result.png"
        self.outputImage.save(nameOfImage)

        return nameOfImage