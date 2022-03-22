import cv2
import sys
import math
import random

class DistanceCalculator:
    def __init__(self, pObjSize):
        #if type(midPointsX) == list and type(midPointsY) == list and type(radii) == list:
        #    self.midPointsX = midPointsX
        #    self.midPointsY = midPointsY
        #    self.radii = radii
        #else:
        #    sys.exit("CHYBA: Nepodporovany parameter!")
        self.objSize = pObjSize
        self.averageDiameter = 0

    def setAverageDiameter(self, pRadii):
        for x in range(len(pRadii)):
            self.averageDiameter = self.averageDiameter + 2 * pRadii[x]

        self.averageDiameter = self.averageDiameter / len(pRadii)

    def getAverageDiameter(self):
        return self.averageDiameter

    def calculateDistance(self, pPointA, pPointB):
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

    def midPoint(self, pPointA, pPointB):
        return ((pPointA[0] + pPointB[0]) * 0.5, (pPointA[1] + pPointB[1]) * 0.5)

    def findAllDistances(self, pCoordsX, pCoordsY, pRadii, pImage, pOption):
        self.setAverageDiameter(pRadii)
        i = 0
        j = 0
        file = open("result.txt", "w")

        while i < len(pCoordsX):
            pointA = (pCoordsX[i], pCoordsY[i])

            while j < len(pCoordsY):
                if j > i:
                    pointB = (pCoordsX[j], pCoordsY[j])
                    if pOption == 1:
                        cv2.line(pImage, (pCoordsX[i], pCoordsY[i]), (pCoordsX[j], pCoordsY[j]),
                                (random.randint(0, 255), random.randint(0, 255),
                                random.randint(0, 255)), 2)
                    elif pOption == 2:
                        pImage.line((pCoordsX[i], pCoordsY[i], pCoordsX[j], pCoordsY[j]), (random.randint(0, 255), random.randint(0, 255),
                                                                                             random.randint(0, 255)))

                    distance = self.calculateDistance(pointA, pointB)
                    (mX, mY) = self.midPoint((pCoordsX[i], pCoordsY[i]), (pCoordsX[j], pCoordsY[j]))

                    if pOption == 1:
                        cv2.putText(pImage, "{:.2f}mm".format(distance), (int(mX + 10), int(mY + 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (83, 34, 171), 2)
                    elif pOption == 2:
                        pImage.text((int(mX), int(mY + 20)), "{:.2f}mm".format(distance), (83, 34, 171))

                    file.write(str((i + 1)) + " -> " + str((j + 1)) + " = " + str(round(distance, 2)) + "mm" + "\n")
                    print(i + 1, "->", j + 1, "=", distance, "mm")

                j += 1

            j = 0
            i += 1

        file.close()

        return pImage