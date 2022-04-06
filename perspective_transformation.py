import cv2
import numpy as np

class PerspectiveTransformation:
    def __init__(self, pImageFilename, pWidth, pHeight, pPoints):
        self.image = cv2.imread(pImageFilename)
        originalDimensions = self.image.shape
        if (originalDimensions[1] > 1500):
            self.image = cv2.resize(self.image, (0, 0), fx=0.5, fy=0.5)
            self.dimensions = self.image.shape
        self.width = pWidth
        self.height = pHeight
        self.points = pPoints

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getPoints(self):
        return self.points

    def showImage(self):
        cv2.imshow("Image", self.image)
        cv2.waitKey(0)

    # vypocita a aplikuje perspektivnu transformaciu na fotku
    def doTransformation(self):
        # suradnice bodov v povodnej fotke, ktore maju byt transformovane
        pts1 = np.float32(self.points)

        # suradnice bodov na transformovanej fotke (velkost transformovanej fotky)
        #pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
        pts2 = np.float32([[0, 0], [self.dimensions[1], 0], [0, self.dimensions[0]], [self.dimensions[1], self.dimensions[0]]])

        # vytvorenie transformacnej matice
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # aplikovanie perspektivnej transformacie na zdrojovu fotku
        transformedImage = cv2.warpPerspective(self.image, matrix, (int(self.dimensions[1]), int(self.dimensions[0])))

        for x in range(0, 4):
            cv2.circle(self.image, (int(pts1[x][0]), int(pts1[x][1])), 5, (255, 0, 0), cv2.FILLED)
        self.showImage()
        return transformedImage


