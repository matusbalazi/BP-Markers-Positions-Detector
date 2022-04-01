import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from colorama import Fore, Style
import math
import matplotlib
import canny_edge_detector
import distance_calculator
from collections import defaultdict

class CircleDetectorWithoutCV:
    def __init__(self, pImageFilename, pMinRadius, pMaxRadius, pObjSize, pNumOfExpectedCircles):
        self.image = Image.open(pImageFilename)
        self.filename = pImageFilename
        self.minRadius = pMinRadius
        self.maxRadius = pMaxRadius
        self.objSize = pObjSize
        self.numOfExpectedCircles = pNumOfExpectedCircles
        self.outputImage = Image.new("RGB", self.image.size)
        self.outputImage.paste(self.image)
        self.drawResult = ImageDraw.Draw(self.outputImage)
        self.steps = 100
        self.threshold = 0.4
        self.canny = canny_edge_detector.CannyEdgeDetector(self.image)
        self.distance = distance_calculator.DistanceCalculator(self.objSize)

    def detectCircles(self, pMinRadius, pMaxRadius, pThreshold):
        points = []
        for r in range(pMinRadius, pMaxRadius + 1):
            for t in range(self.steps):
                points.append((r, int(r * cos(2 * pi * t / self.steps)), int(r * sin(2 * pi * t / self.steps))))

        acc = defaultdict(int)
        for x, y in self.canny.canny_edge_detector():
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1

        circles = []
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / self.steps >= pThreshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                print(v / self.steps, x, y, r)
                circles.append((x, y, r))

        print(circles)
        return circles

    def sortCircles(self, pListOfCircles):
        pListOfCircles[0], pListOfCircles[4] = pListOfCircles[4], pListOfCircles[0]
        pListOfCircles[1], pListOfCircles[5] = pListOfCircles[5], pListOfCircles[1]
        pListOfCircles[2], pListOfCircles[3] = pListOfCircles[3], pListOfCircles[2]

    def findAllCircles(self, pWasSuccess):
        listOfCircles = []

        if pWasSuccess == 0:
            numOfCircles = 0
            i = 0
            while numOfCircles != self.numOfExpectedCircles:
                if numOfCircles < self.numOfExpectedCircles:
                    listOfCircles = self.detectCircles(self.minRadius - i, self.maxRadius + i, self.threshold)
                elif numOfCircles > self.numOfExpectedCircles:
                    listOfCircles = self.detectCircles(self.minRadius + i, self.maxRadius - i, self.threshold)
                else:
                    listOfCircles = self.detectCircles(self.minRadius, self.maxRadius, self.threshold)

                if listOfCircles is not None:
                    numOfCircles = int(len(listOfCircles))

                i += 1
        else:
            listOfCircles = self.detectCircles(self.minRadius, self.maxRadius, self.threshold)
            numOfCircles = int(len(listOfCircles))

        listOfCoordsX = []
        listOfCoordsY = []
        listOfRadii = []

        if listOfCircles:
            if numOfCircles == self.numOfExpectedCircles:
                self.sortCircles(listOfCircles)

            if numOfCircles < self.numOfExpectedCircles:
                print(Fore.RED + "\nNebol detegovany pozadovany pocet kruhov!")
                print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
                print(Style.RESET_ALL)

                return 0, self.image
            else:
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

                return 1, nameOfImage

        else:
            print(Fore.RED + "\nNeboli detegovane ziadne kruhy!")
            print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
            print(Style.RESET_ALL)

            return 0, self.filename