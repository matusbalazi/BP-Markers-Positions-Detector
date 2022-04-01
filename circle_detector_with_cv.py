import distance_calculator
import corner_detector
from colorama import Fore, Style
import cv2

class CircleDetectorWithCV:
    def __init__(self, pImageFilename, pMinRadius, pMaxRadius, pObjSize, pNumOfExpectedCircles):
        self.image = cv2.imread(pImageFilename)
        self.minRadius = pMinRadius
        self.maxRadius = pMaxRadius
        self.objSize = pObjSize
        self.numOfExpectedCircles = pNumOfExpectedCircles
        self.decision = "N"
        self.imageCopy = 0
        self.distance = distance_calculator.DistanceCalculator(self.objSize)
        self.corners = corner_detector.CornerDetector(self.image)

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

    def findAllCircles(self, pOption, pWasSuccess):
        #TODO: Dorobit, aby sa oba polomery menili v urcitom pomere a iba ak bude pouzivatel chciet
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
            for co, i in enumerate(listOfCircles[0, :], start=1):
                if pOption == 1:
                    cv2.putText(self.imageCopy, "{:d}.".format(co), (int(i[0] + 20), int(i[1] + 20)),
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
                for i in range(len(list1)):
                    cv2.putText(self.imageCopy, "{:d}.".format(i + 1), (int(list1[i] + 20), int(list2[i] + 20)),
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
