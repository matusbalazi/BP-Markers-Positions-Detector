import cv2
import numpy as np


class PerspectiveTransformation:
    # Initialization
    def __init__(self, pImageFilename: str, pWidth: int, pHeight: int, pPoints: np.ndarray):
        self.image = cv2.imread(pImageFilename)
        self.width = pWidth
        self.height = pHeight
        self.points = pPoints

        self.wasResize = False
        originalDimensions = self.image.shape

        # Resize input image if the image is too big
        if (originalDimensions[1] > 2000):
            self.image = cv2.resize(self.image, (0, 0), fx=0.5, fy=0.5)
            self.dimensions = self.image.shape
            self.wasResize = True

    # Returns width of the output image
    def getWidth(self):
        return self.width

    # Returns height of the output image
    def getHeight(self):
        return self.height

    # Returns points to be transformed
    def getPoints(self):
        return self.points

    # Shows image to user
    def showImage(self):
        cv2.imshow("Original Image", self.image)
        cv2.waitKey(0)

    # Calculates and applies a perspective transformation to an image
    def doTransformation(self) -> np.ndarray:
        # The coordinates of the points in the original image to be transformed
        pts1 = np.float32(self.points)

        # The coordinates of the points in the transformed image (size of the transformed image)
        if self.wasResize == False:
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
        else:
            pts2 = np.float32(
                [[0, 0], [self.dimensions[1], 0], [0, self.dimensions[0]], [self.dimensions[1], self.dimensions[0]]])

        # Creating a transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # Applying a perspective transformation to the source image
        transformedImage = cv2.warpPerspective(self.image, matrix, (int(self.dimensions[1]), int(self.dimensions[0])))

        for x in range(0, 4):
            cv2.circle(self.image, (int(pts1[x][0]), int(pts1[x][1])), 5, (255, 0, 0), cv2.FILLED)

        self.showImage()

        return transformedImage
