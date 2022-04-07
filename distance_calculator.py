import cv2
import math
import random
import numpy as np


class DistanceCalculator:
    # Initialization
    def __init__(self, pObjSize: int):
        self.objSize = pObjSize
        self.averageDiameter = 0

    # Calculation of the average circle diameter
    def setAverageDiameter(self, pRadii: list):
        for x in range(len(pRadii)):
            self.averageDiameter = self.averageDiameter + 2 * pRadii[x]

        self.averageDiameter = self.averageDiameter / len(pRadii)

    # Returns average circle diameter
    def getAverageDiameter(self) -> float:
        return self.averageDiameter

    # Calculation of the distance between the centers of two circles
    # based on the Euclidean distance and the size ratio of the actual
    # circle (in mm) and the detected circle (in px)
    def calculateDistance(self, pPointA: tuple, pPointB: tuple) -> float:
        x1, y1 = pPointA
        x2, y2 = pPointB

        if x1 < x2:
            width = x2 - x1
        elif x1 == x2:
            width = 0
        else:
            width = x1 - x2

        if y1 < y2:
            height = y2 - y1
        elif y1 == y2:
            height = 0
        else:
            height = y1 - y2

        if height == 0:
            return width
        elif width == 0:
            return height

        sizeOfOnePixel = self.objSize / self.getAverageDiameter()

        return math.sqrt(((width * sizeOfOnePixel) ** 2) + ((height * sizeOfOnePixel) ** 2))

    # Find the center of the circle
    def midPoint(self, pPointA: tuple, pPointB: tuple) -> tuple:
        return ((pPointA[0] + pPointB[0]) * 0.5, (pPointA[1] + pPointB[1]) * 0.5)

    # Determine the distance between the centers of all detected circles,
    # write these distances to a file and draw in an image
    def findAllDistances(self, pCoordsX: list, pCoordsY: list, pRadii: list, pImage: np.ndarray,
                         pOption: int) -> np.ndarray:
        self.setAverageDiameter(pRadii)
        i = 0
        j = 0
        file = open("result.txt", "w")

        while i < len(pCoordsX):
            pointA = (pCoordsX[i], pCoordsY[i])

            while j < len(pCoordsY):
                if j > i:
                    pointB = (pCoordsX[j], pCoordsY[j])
                    color = (random.randint(0, 255), random.randint(0, 255),
                             random.randint(0, 255))
                    if pOption == 1:
                        cv2.line(pImage, (pCoordsX[i], pCoordsY[i]), (pCoordsX[j], pCoordsY[j]),
                                 color, 2)
                    elif pOption == 2:
                        pImage.line((pCoordsX[i], pCoordsY[i], pCoordsX[j], pCoordsY[j]),
                                    (random.randint(0, 255), random.randint(0, 255),
                                     random.randint(0, 255)))

                    distance = self.calculateDistance(pointA, pointB)
                    (mX, mY) = self.midPoint((pCoordsX[i], pCoordsY[i]), (pCoordsX[j], pCoordsY[j]))

                    if pOption == 1:
                        cv2.putText(pImage, "{:.2f}mm".format(distance), (int(mX + 10), int(mY + 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    elif pOption == 2:
                        pImage.text((int(mX), int(mY + 20)), "{:.2f}mm".format(distance), (83, 34, 171))

                    file.write(str(i) + " -> " + str(j) + " = " + str(round(distance, 2)) + "\n")
                    print(i + 1, "->", j + 1, "=", distance, "mm")

                j += 1

            j = 0
            i += 1

        file.write("\n")
        file.close()

        return pImage
