import distance_calculator
import corner_detector
import cv2

class CircleDetectorWithCV:
    def __init__(self, pImageFilename, pMinRadius, pMaxRadius, pObjSize):
        self.image = cv2.imread(pImageFilename)
        #self.image = pImage
        self.minRadius = pMinRadius
        self.maxRadius = pMaxRadius
        self.objSize = pObjSize
        self.imageCopy = 0
        self.distance = distance_calculator.DistanceCalculator(self.objSize)
        self.corners = corner_detector.CornerDetector(self.image)

    def detectCircles(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.imageCopy = self.image.copy()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image, 5)
        edgedImage = cv2.Canny(self.image, 50, 100)
        edgedImage = cv2.dilate(edgedImage, None, iterations=1)
        edgedImage = cv2.erode(edgedImage, None, iterations=1)
        rows = self.image.shape[0]
        circles = cv2.HoughCircles(image=edgedImage, method=cv2.HOUGH_GRADIENT, dp=1.0,
                                   minDist=rows/8, param1=100, param2=30, minRadius=self.minRadius,
                                   maxRadius=self.maxRadius)
        return circles

    def findAllCircles(self, pOption):
        listOfCircles = self.detectCircles()
        listOfCoordsX = []
        listOfCoordsY = []
        listOfRadii = []

        for co, i in enumerate(listOfCircles[0, :], start=1):
            if pOption == 1:
                cv2.putText(self.imageCopy, "{:d}.".format(co), (int(i[0] + 20), int(i[1] + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.circle(self.imageCopy, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                cv2.circle(self.imageCopy, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
            listOfCoordsX.append(int(i[0]))
            listOfCoordsY.append(int(i[1]))
            # pozor na int!
            listOfRadii.append(int(i[2]))

        if pOption == 1:
            return self.distance.findAllDistances(listOfCoordsX, listOfCoordsY, listOfRadii, self.imageCopy, 1)
        elif pOption == 2:
            list1, list2 = self.corners.findMidpointsOfCircles(listOfCoordsX, listOfCoordsY, listOfRadii)
            for i in range(len(list1)):
                cv2.circle(self.imageCopy, (int(list1[i]), int(list2[i])),
                           int(listOfRadii[i]), (0, 255, 0), 2)
                cv2.circle(self.imageCopy, (int(list1[i]), int(list2[i])), 2, (255, 0, 0), 3)
            return self.distance.findAllDistances(list1, list2, listOfRadii, self.imageCopy, 1)
