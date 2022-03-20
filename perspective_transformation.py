import cv2
import numpy as np

class PerspectiveTransformation:
    def __init__(self, pImage, pWidth, pHeight, pPoints):
        self.image = pImage
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
        pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])

        # vytvorenie transformacnej matice
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # aplikovanie perspektivnej transformacie na zdrojovu fotku
        transformedImage = cv2.warpPerspective(self.image, matrix, (int(self.width), int(self.height)))

        for x in range(0, 4):
            cv2.circle(self.image, (int(pts1[x][0]), int(pts1[x][1])), 5, (255, 0, 0), cv2.FILLED)
        self.showImage()
        return transformedImage


