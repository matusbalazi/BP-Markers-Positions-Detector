import perspective_transformation
import circle_detector_with_cv
import circle_detector_without_cv
import argparse
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
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

    # nieco = input("Stlacte klavesu: ")

    img = args["image"]

    # image = "../images/image13.png"

    # width = args["width"]
    # height = args["height"]
    # points = np.array(eval(args["coords"]), dtype="float32")

    circleDetector = circle_detector_with_cv.CircleDetectorWithCV(img, 90, 105, 100)
    circleDetector2 = circle_detector_without_cv.CircleDetectorWithoutCV(img, 90, 105, 100)
    circleDetector2.findAllCircles()
    # perspective = perspective_transformation.PerspectiveTransformation(img, width, height, points)

    cv2.imshow("Image", circleDetector.findAllCircles(1))
    cv2.waitKey(0)
    # cv2.imshow("Image", perspective.doTransformation())
    # cv2.imshow("Image", perspective.doTransformation())
    # cv2.waitKey(0)
    # print(perspective.getCoords())
    # perspective.showImage()
    # print(perspective.getWidth())
    # print(perspective.getHeight())

if __name__ == "__main__":
    main()


