import cv2
import numpy as np


class CornerDetector:
    # Initialization
    def __init__(self, pImage, pNumOfExpectedCircles, pTypeOfImage):
        self.image = pImage
        self.numOfExpectedCircles = pNumOfExpectedCircles
        self.typeOfImage = pTypeOfImage
        self.imageCopy = 0

    # Uses Harris Corner Detection method used to
    # detect all the corners in the image
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

    # Sorting algorithm used to sort midpoints of circles according to their coordinates
    def bubbleSort(self, pList1, pList2):
        for i in range(0, len(pList2) - 1):
            for j in range(len(pList2) - 1):
                if (pList2[j] > pList2[j + 1]):
                    temp1 = pList2[j]
                    pList2[j] = pList2[j + 1]
                    pList2[j + 1] = temp1

                    temp2 = pList1[j]
                    pList1[j] = pList1[j + 1]
                    pList1[j + 1] = temp2

        pList1.reverse()
        pList2.reverse()

    # Sorts midpoints of circles according to anchor positions (anchors A, B and C)
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

    # Modified Hough Circle Transform based on Harris corner detection method
    # defines new midpoints of circles in the image
    def findMidpointsOfCircles(self, pListOfCoordsX, pListOfCoordsY, pListOfRadii):
        corners = self.detectCorners()
        listOfCorners = []
        listOfCornersX = []
        listOfCornersY = []

        # Filtering out all corners that are not in any of the circles
        for i in range(1, len(corners)):
            for j in range(len(corners[i])):
                listOfCorners.append(corners[i][j])

        # Dividing the coordinates of the corners that are
        # in one of the circles into X coord and Y coord
        for i in range(len(listOfCorners)):
            if i % 2 == 0:
                listOfCornersX.append(listOfCorners[i])
            else:
                listOfCornersY.append(listOfCorners[i])

        listOfMidpointsX = []
        listOfMidpointsY = []

        # The corner that is closest to the center of the circle
        # detected so far becomes the new center of the circle
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

        # Sorts midpoints of final circles from the largest
        # coordinates to the smallest coordinates
        self.bubbleSort(listOfMidpointsX, listOfMidpointsY)

        # Sorts previously sorted midpoints of circles to
        # the correct order according to anchors
        if len(listOfMidpointsX) == self.numOfExpectedCircles:
            markersCorrectOrderX, markersCorrectOrderY = self.sortMidpoints(listOfMidpointsX, listOfMidpointsY,
                                                                            self.typeOfImage)
        else:
            markersCorrectOrderX, markersCorrectOrderY = listOfMidpointsX, listOfMidpointsY

        return (markersCorrectOrderX, markersCorrectOrderY)
