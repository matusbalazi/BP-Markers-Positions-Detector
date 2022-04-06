from filecmp import cmp
import numpy as np
import distance_calculator
import corner_detector
from colorama import Fore, Style
from operator import itemgetter
import cv2

class CircleDetectorWithCV:
    def __init__(self, pImageFilename, pMinRadius, pMaxRadius, pObjSize, pNumOfExpectedCircles, pTypeOfImage):
        self.image = cv2.imread(pImageFilename)
        dimensions = self.image.shape
        if (dimensions[1] > 1500):
            self.image = cv2.resize(self.image, (0, 0), fx=0.5, fy=0.5)
        self.minRadius = pMinRadius
        self.maxRadius = pMaxRadius
        self.objSize = pObjSize
        self.numOfExpectedCircles = pNumOfExpectedCircles
        self.typeOfImage = pTypeOfImage
        self.decision = "N"
        self.imageCopy = 0
        self.distance = distance_calculator.DistanceCalculator(self.objSize)
        self.corners = corner_detector.CornerDetector(self.image, self.numOfExpectedCircles, self.typeOfImage)

    def detectCircles(self, pImage, pMinRadius, pMaxRadius):
        self.imageCopy = pImage.copy()
        pImage = cv2.cvtColor(pImage, cv2.COLOR_BGR2RGB)
        pImage = cv2.cvtColor(pImage, cv2.COLOR_BGR2GRAY)
        pImage = cv2.medianBlur(pImage, 5)
        edgedImage = cv2.Canny(pImage, 50, 100)
        edgedImage = cv2.dilate(edgedImage, None, iterations=1)
        edgedImage = cv2.erode(edgedImage, None, iterations=1)

        circles = cv2.HoughCircles(image=edgedImage, method=cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=50, param1=100, param2=30, minRadius=pMinRadius,
                                   maxRadius=pMaxRadius)
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

    def sortCircles(self, pListOfCircles, pTypeOfImage):
        markersOrder = []
        auxList = sorted(pListOfCircles[0], key=itemgetter(1), reverse=True)
        if pTypeOfImage == 1:
            markersOrder.append(auxList[3])
            marker1, marker2 = self.cmp(auxList[2], auxList[1])
            markersOrder.append(marker1)
            marker2, marker5 = self.cmp(auxList[5], marker2)
            markersOrder.append(marker2)
            markersOrder.append(auxList[6])
            markersOrder.append(auxList[4])
            markersOrder.append(marker5)
            markersOrder.append(auxList[0])
        else:
            markersOrder.append(auxList[3])
            markersOrder.append(auxList[1])
            markersOrder.append(auxList[2])
            markersOrder.append(auxList[5])
            markersOrder.append(auxList[6])
            markersOrder.append(auxList[4])
            markersOrder.append(auxList[0])

        sortedList = np.array(markersOrder)

        return sortedList

    def findAllCircles(self, pOption, pWasSuccess):
        #TODO: Ukladat kazdy obrazok do priecinku images

        listOfCircles = []

        if pWasSuccess == 0:
            numOfCircles = 0
            i = 0
            while numOfCircles != self.numOfExpectedCircles:
                if numOfCircles < self.numOfExpectedCircles:
                    listOfCircles = self.detectCircles(self.image, self.minRadius - i, self.maxRadius + i)
                elif numOfCircles > self.numOfExpectedCircles:
                    listOfCircles = self.detectCircles(self.image, self.minRadius + i, self.maxRadius - i)
                else:
                    listOfCircles = self.detectCircles(self.image, self.minRadius, self.maxRadius)

                if listOfCircles is not None:
                    numOfCircles = int(sum(map(len, listOfCircles)))

                i += 1
        else:
            listOfCircles = self.detectCircles(self.image, self.minRadius, self.maxRadius)

        listOfCoordsX = []
        listOfCoordsY = []
        listOfRadii = []

        if listOfCircles is not None:
            aux = None
            if int(sum(map(len, listOfCircles))) == self.numOfExpectedCircles:
                sortedListOfCircles = self.sortCircles(listOfCircles, self.typeOfImage)
                listOfCircles = sortedListOfCircles
                aux = listOfCircles[:]
            else:
                aux = listOfCircles[0, :]

            for co, i in enumerate(aux, start=1):
                if pOption == 1:
                    cv2.putText(self.imageCopy, "{:d}.".format(co - 1), (int(i[0] + 20), int(i[1] + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.circle(self.imageCopy, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                    cv2.circle(self.imageCopy, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
                listOfCoordsX.append(int(i[0]))
                listOfCoordsY.append(int(i[1]))
                listOfRadii.append(int(i[2]))
                print(co, "X:", i[0], "| Y:", i[1], "| R:", i[2])

            numOfCircles = int(sum(map(len, listOfCircles)))

            if pOption == 1:
                if numOfCircles < self.numOfExpectedCircles:
                    print(Fore.RED + "\nNebol detegovany pozadovany pocet kruhov!")
                    print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
                    print(Style.RESET_ALL)
                    return 0, self.distance.findAllDistances(listOfCoordsX, listOfCoordsY, listOfRadii, self.imageCopy, 1)
                else:
                    return 1, self.distance.findAllDistances(listOfCoordsX, listOfCoordsY, listOfRadii, self.imageCopy, 1)
            elif pOption == 2:
                list1, list2 = self.corners.findMidpointsOfCircles(listOfCoordsX, listOfCoordsY, listOfRadii)
                print(list1)
                for i in range(len(list1)):
                    cv2.putText(self.imageCopy, "{:d}.".format(i), (int(list1[i] + 20), int(list2[i] + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.circle(self.imageCopy, (int(list1[i]), int(list2[i])),
                               int(listOfRadii[i]), (0, 255, 0), 2)
                    cv2.circle(self.imageCopy, (int(list1[i]), int(list2[i])), 2, (255, 0, 0), 3)

                listOfRadii.reverse()

                if numOfCircles < self.numOfExpectedCircles:
                    print(Fore.RED + "\nNebol detegovany pozadovany pocet kruhov!")
                    print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
                    print(Style.RESET_ALL)
                    return 0, self.distance.findAllDistances(list1, list2, listOfRadii, self.imageCopy, 1)
                else:
                    return 1, self.distance.findAllDistances(list1, list2, listOfRadii, self.imageCopy, 1)
        else:
            print(Fore.RED + "\nNeboli detegovane ziadne kruhy!")
            print(Fore.YELLOW + "Skuste zmenit interval medzi najmensim a najvacsim hladanym polomerom.")
            print(Style.RESET_ALL)

            return 0, self.image
