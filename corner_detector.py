import cv2
import numpy as np

class CornerDetector:
    def __init__(self, pImage, pNumOfExpectedCircles, pTypeOfImage):
        self.image = pImage
        self.numOfExpectedCircles = pNumOfExpectedCircles
        self.typeOfImage = pTypeOfImage
        self.imageCopy = 0

    # detekcia vsetkych rohov na obrazku
    def detectCorners(self):
        self.imageCopy = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.imageCopy = np.float32(self.imageCopy)
        dst = cv2.cornerHarris(self.imageCopy, 5, 3, 0.04)
        ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(self.imageCopy, np.float32(centroids), (5, 5), (-1, -1), criteria)
        return corners

    def bubbleSort(self, pList1, pList2):
        for i in range(0, len(pList2) - 1):
            for j in range(len(pList2) - 1):
                if(pList2[j] > pList2[j + 1]):
                    temp1 = pList2[j]
                    pList2[j] = pList2[j + 1]
                    pList2[j + 1] = temp1

                    temp2 = pList1[j]
                    pList1[j] = pList1[j + 1]
                    pList1[j + 1] = temp2

        pList1.reverse()
        pList2.reverse()

    def sortMidpoints(self, pListOfMidpointsX, pListOfMidpointsY, pTypeOfImage):
        listX = []
        listY = []
        if pTypeOfImage == 1:
            listX.append(pListOfMidpointsX[3])
            listX.append(pListOfMidpointsX[1])
            listX.append(pListOfMidpointsX[5])
            listX.append(pListOfMidpointsX[6])
            listX.append(pListOfMidpointsX[4])
            listX.append(pListOfMidpointsX[2])
            listX.append(pListOfMidpointsX[0])

            listY.append(pListOfMidpointsY[3])
            listY.append(pListOfMidpointsY[1])
            listY.append(pListOfMidpointsY[5])
            listY.append(pListOfMidpointsY[6])
            listY.append(pListOfMidpointsY[4])
            listY.append(pListOfMidpointsY[2])
            listY.append(pListOfMidpointsY[0])
        else:
            listX.append(pListOfMidpointsX[3])
            listX.append(pListOfMidpointsX[1])
            listX.append(pListOfMidpointsX[2])
            listX.append(pListOfMidpointsX[5])
            listX.append(pListOfMidpointsX[6])
            listX.append(pListOfMidpointsX[4])
            listX.append(pListOfMidpointsX[0])

            listY.append(pListOfMidpointsY[3])
            listY.append(pListOfMidpointsY[1])
            listY.append(pListOfMidpointsY[2])
            listY.append(pListOfMidpointsY[5])
            listY.append(pListOfMidpointsY[6])
            listY.append(pListOfMidpointsY[4])
            listY.append(pListOfMidpointsY[0])

        return listX, listY

   # upravena Houghova kruhova transformacia na zaklade Harrisovej metody detekcie rohov
    def findMidpointsOfCircles(self, pListOfCoordsX, pListOfCoordsY, pListOfRadii):
        corners = self.detectCorners()
        listOfCorners = []
        listOfCornersX = []
        listOfCornersY = []

        # odstranenie vsetkych rohov, ktore sa nenachadzaju v niektorom z kruhov
        for i in range(1, len(corners)):
            for j in range(len(corners[i])):
                listOfCorners.append(corners[i][j])

        for i in range(len(listOfCorners)):
            if i % 2 == 0:
                listOfCornersX.append(listOfCorners[i])
            else:
                listOfCornersY.append(listOfCorners[i])

        listOfMidpointsX = []
        listOfMidpointsY = []

        # roh nachadzajuci si najblizsie k stredu doposial detegovaneho kruhu sa stane novym stredom kruhu
        for i in range(len(listOfCornersX)):
            for j in range(len(pListOfCoordsX)):
                if (listOfCornersX[i] >= (pListOfCoordsX[j] - pListOfRadii[j]) and
                        listOfCornersX[i] <= (pListOfCoordsX[j] + pListOfRadii[j])):
                    if (listOfCornersY[i] >= (pListOfCoordsY[j] - pListOfRadii[j]) and
                            listOfCornersY[i] <= (pListOfCoordsY[j] + pListOfRadii[j])):
                        pListOfCoordsX.remove(pListOfCoordsX[j])
                        pListOfCoordsY.remove(pListOfCoordsY[j])
                        listOfMidpointsX.append(int(listOfCornersX[i]))
                        listOfMidpointsY.append(int(listOfCornersY[i]))
                        break

        self.bubbleSort(listOfMidpointsX, listOfMidpointsY)

        if len(listOfMidpointsX) == self.numOfExpectedCircles:
            markersCorrectOrderX, markersCorrectOrderY = self.sortMidpoints(listOfMidpointsX, listOfMidpointsY, self.typeOfImage)
        else:
            markersCorrectOrderX, markersCorrectOrderY = listOfMidpointsX, listOfMidpointsY

        return (markersCorrectOrderX, markersCorrectOrderY)