import cv2
import numpy as np

class CornerDetector:
    def __init__(self, pImage):
        self.image = pImage
        self.imageCopy = 0

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
        #for i in range(1, len(corners)):
        #    print(corners[i])

    def findMidpointsOfCircles(self, pListOfCoordsX, pListOfCoordsY, pListOfRadii):
        corners = self.detectCorners()
        listOfCorners = []
        listOfCornersX = []
        listOfCornersY = []
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

        return (listOfMidpointsX, listOfMidpointsY)