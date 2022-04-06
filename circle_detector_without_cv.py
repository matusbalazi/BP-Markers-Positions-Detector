from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from colorama import Fore, Style
import canny_edge_detector
import distance_calculator
from collections import defaultdict
from operator import itemgetter

class CircleDetectorWithoutCV:
    def __init__(self, pImageFilename, pMinRadius, pMaxRadius, pObjSize, pNumOfExpectedCircles, pTypeOfImage):
        self.image = Image.open(pImageFilename)
        if (self.image.size[0] > 1500):
            newWidth = int(self.image.size[0] / 2)
            newHeight = int(self.image.size[1] / 2)
            self.image = self.image.resize((newWidth, newHeight))
        self.filename = pImageFilename
        self.minRadius = pMinRadius
        self.maxRadius = pMaxRadius
        self.objSize = pObjSize
        self.numOfExpectedCircles = pNumOfExpectedCircles
        self.typeOfImage = pTypeOfImage
        self.outputImage = Image.new("RGB", self.image.size)
        self.outputImage.paste(self.image)
        self.drawResult = ImageDraw.Draw(self.outputImage)
        self.steps = 100
        self.threshold = 0.4
        self.canny = canny_edge_detector.CannyEdgeDetector(self.image)
        self.distance = distance_calculator.DistanceCalculator(self.objSize)

    # detekcia kruhov v obrazku na zaklade minimalneho a maximalneho polomeru
    # threshold predstavuje prahovu hodnotu, od ktorej mozeme kruh povazovat
    # za doveryhodny a splnajuci zadane kriteria
    def detectCircles(self, pMinRadius, pMaxRadius, pThreshold):
        points = []
        for r in range(pMinRadius, pMaxRadius + 1):
            for t in range(self.steps):
                points.append((r, int(r * cos(2 * pi * t / self.steps)), int(r * sin(2 * pi * t / self.steps))))

        # Cannyho algoritmus detekcie hran
        acc = defaultdict(int)
        for x, y in self.canny.canny_edge_detector():
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1

        # Houghova kruhova transformacia
        circles = []
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / self.steps >= pThreshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
                print(v / self.steps, x, y, r)
                circles.append((x, y, r))

        print(circles)
        return circles

    def cmp(self, pCoord1, pCoord2):
        marker1 = None
        marker2 = None
        if pCoord1[0] >= pCoord2[0] and pCoord1[1] <= pCoord2[1]:
            marker1 = pCoord1
            marker2 = pCoord2
        else:
            marker1 = pCoord2
            marker2 = pCoord1
        return marker1, marker2

    # utriedenie najdenych kruhov
    def sortCircles(self, pListOfCircles, pTypeOfImage):
        markersOrder = []
        auxList = sorted(pListOfCircles, key=itemgetter(1), reverse=True)
        print("Sorted list:")
        print(auxList)
        if pTypeOfImage == 1:
            markersOrder.append(auxList[3])
            marker2, marker1 = self.cmp(auxList[1], auxList[2])
            markersOrder.append(marker1)
            marker2, marker5 = self.cmp(auxList[5], marker2)
            markersOrder.append(marker2)
            markersOrder.append(auxList[6])
            markersOrder.append(auxList[4])
            markersOrder.append(marker5)
            markersOrder.append(auxList[0])

        else:
            markersOrder.append(auxList[3])
            marker2, marker1 = self.cmp(auxList[1], auxList[2])
            markersOrder.append(marker1)
            marker2, marker5 = self.cmp(auxList[5], marker2)
            markersOrder.append(marker2)
            markersOrder.append(auxList[6])
            markersOrder.append(auxList[4])
            markersOrder.append(marker5)
            markersOrder.append(auxList[0])

        return markersOrder

    # najdenie kruhov a vypocitanie vzdialenosti medzi nimi
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
            print("Num of circles: ", numOfCircles)
            if numOfCircles == self.numOfExpectedCircles:
                listOfCircles = self.sortCircles(listOfCircles, self.typeOfImage)

            if numOfCircles < self.numOfExpectedCircles:
                print(Fore.RED + "\nNebol detegovany pozadovany pocet kruhov!")
                print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
                print(Style.RESET_ALL)

                return 0, self.filename
            else:
                i = 0
                print("List of circles:")
                print(listOfCircles)
                for x, y, r in listOfCircles:
                    self.drawResult.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))
                    self.drawResult.text((x, y), "{:d}.".format(i), (0, 0, 0))
                    listOfCoordsX.append(x)
                    listOfCoordsY.append(y)
                    listOfRadii.append(r)
                    i += 1

                self.distance.findAllDistances(listOfCoordsX, listOfCoordsY, listOfRadii, self.drawResult, 2)

                nameOfImage = ""
                if self.typeOfImage == 1:
                    nameOfImage = "output_images/detectedCirclesOriginal_withoutOpenCV.jpg"
                elif self.typeOfImage == 2:
                    nameOfImage = "output_images/detectedCirclesTransformed_withoutOpenCV.jpg"

                self.outputImage.save(nameOfImage)

                return 1, nameOfImage

        else:
            print(Fore.RED + "\nNeboli detegovane ziadne kruhy!")
            print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
            print(Style.RESET_ALL)

            return 0, self.filename