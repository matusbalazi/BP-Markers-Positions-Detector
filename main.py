import perspective_transformation
import circle_detector_with_cv
import argparse
import sys
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="path to the input image")
    ap.add_argument("-wi", "--width", required=False,
                    help="width of the output image")
    ap.add_argument("-he", "--height", required=False,
                    help="height of the output image")
    ap.add_argument("-c", "--coords", required=False,
                    help="comma seperated list of source points")
    ap.add_argument("-s", "--size", type=float, required=False,
                    help="size of the known object in the real world (in milimeters)")
    args = vars(ap.parse_args())

    print(len(sys.argv))

    if len(sys.argv) > 1:
        image = args["image"]
        width = args["width"]
        height = args["height"]
        coords = args["coords"]
    else:
        image = str(input("Zadajte cestu k fotke: "))
        width = str(input("Zadajte sirku fotky (v pixeloch) po transformovani: "))
        height = str(input("Zadajte vysku fotky (v pixeloch) po transformovani: "))
        coords = str(input("Zadajte suradnice rohov objektu urceneho na transformovanie (napr. [(327,303), (1128, 307), ...]): "))

    points = np.array(eval(coords), dtype="float32")
    newImage = "after_perspective_transformation.jpg"

    perspective = perspective_transformation.PerspectiveTransformation(image, width, height, points)
    transformedImage = perspective.doTransformation()
    cv2.imshow("Image", transformedImage)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    status = cv2.imwrite(newImage, transformedImage)

    cv2.waitKey(1000000)

    print("Obrazok bol ulozeny na disk: ", status)


    minRadius = str(input("Zadajte najmensi polomer hladaneho kruhu: "))
    maxRadius = str(input("Zadajte najvacsi polomer hladaneho kruhu: "))
    objSize = str(input("Zadajte priemer kruhu v skutocnosti: "))

    circleDetector = circle_detector_with_cv.CircleDetectorWithCV(newImage, int(minRadius), int(maxRadius), int(objSize))

    cv2.imshow("Image", circleDetector.findAllCircles(1))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()


