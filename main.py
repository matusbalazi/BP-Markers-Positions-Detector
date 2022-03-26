import perspective_transformation
import circle_detector_with_cv
import circle_detector_without_cv
from colorama import Fore, Style
import argparse
import sys
import cv2
from PIL import Image
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

    wasEnd = False
    newImage = "after_perspective_transformation.jpg"

    image = None

    while not wasEnd:
        print(Fore.CYAN + "\n----- HLAVNE MENU -----" + Style.RESET_ALL)
        print("[1] Perspective transformation")
        print("[2] Circle detector w/distance calculator (OpenCV)")
        print("[3] Circle detector w/distance calculator")
        print("[0] Exit")
        print()
        decision = input(Fore.MAGENTA + "Zvolte moznost: " + Style.RESET_ALL)
        print()

        if (len(sys.argv) > 1):
            image = args["image"]
        else:
            if image is None:
                image = str(input(Fore.MAGENTA + "Zadajte cestu k fotke: " + Style.RESET_ALL))

        if decision == "1":
            if len(sys.argv) > 2:
                width = args["width"]
                height = args["height"]
                coords = args["coords"]
            else:
                width = str(input(Fore.MAGENTA + "Zadajte sirku fotky (v pixeloch) po transformovani: " + Style.RESET_ALL))
                height = str(input(Fore.MAGENTA + "Zadajte vysku fotky (v pixeloch) po transformovani: " + Style.RESET_ALL))
                coords = str(input(Fore.MAGENTA +
                    "Zadajte suradnice rohov objektu urceneho na transformovanie (napr. [(327,303), (1128, 307), ...]): " + Style.RESET_ALL))

            points = np.array(eval(coords), dtype="float32")

            perspective = perspective_transformation.PerspectiveTransformation(image, width, height, points)
            transformedImage = perspective.doTransformation()
            cv2.imshow("Image", transformedImage)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            status = cv2.imwrite(newImage, transformedImage)
            cv2.waitKey(1000000)

            print("Obrazok bol ulozeny na disk: ", status)

        elif decision == "2":
            minRadius = 0
            maxRadius = 0
            objSize = 0
            numOfExpectedCircles = 0

            print("\n[1] Original image")
            print("[2] Transformed Image")
            print()
            decision2 = input(Fore.MAGENTA + "Zvolte moznost: " + Style.RESET_ALL)
            print()

            if len(sys.argv) > 9:
                print("Nothing for now")
            else:
                minRadius = input(Fore.MAGENTA + "Zadajte najmensi polomer hladaneho kruhu: " + Style.RESET_ALL)
                maxRadius = input(Fore.MAGENTA + "Zadajte najvacsi polomer hladaneho kruhu: " + Style.RESET_ALL)
                objSize = input(Fore.MAGENTA + "Zadajte priemer kruhu v skutocnosti: " + Style.RESET_ALL)
                numOfExpectedCircles = input(Fore.MAGENTA + "Zadajte kolko kruhov by malo byt detegovanych: " + Style.RESET_ALL)

            circleDetector = None

            if decision2 == "1":
                circleDetector = circle_detector_with_cv.CircleDetectorWithCV(image, int(minRadius), int(maxRadius), int(objSize), int(numOfExpectedCircles))
            elif decision2 == "2":
                circleDetector = circle_detector_with_cv.CircleDetectorWithCV(newImage, int(minRadius), int(maxRadius), int(objSize), int(numOfExpectedCircles))

            print("\n[1] Hough Circle Transform")
            print("[2] Hough Circle Transform + Harris Corner Detector")
            print()
            decision3 = input(Fore.MAGENTA + "Zvolte moznost: " + Style.RESET_ALL)
            print()

            if decision3 == "1":
                cv2.imshow("Image", circleDetector.findAllCircles(1))
            elif decision3 == "2":
                cv2.imshow("Image", circleDetector.findAllCircles(2))

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif decision == "3":
            minRadius = 0
            maxRadius = 0
            objSize = 0

            print("\n[1] Original image")
            print("[2] Transformed Image")
            print()
            decision3 = input(Fore.MAGENTA + "Zvolte moznost: " + Style.RESET_ALL)
            print()

            if len(sys.argv) > 9:
                print("Nothing for now")
            else:
                minRadius = input(Fore.MAGENTA + "Zadajte najmensi polomer hladaneho kruhu: " + Style.RESET_ALL)
                maxRadius = input(Fore.MAGENTA + "Zadajte najvacsi polomer hladaneho kruhu: " + Style.RESET_ALL)
                objSize = input(Fore.MAGENTA + "Zadajte priemer kruhu v skutocnosti: " + Style.RESET_ALL)

            circleDetector = None

            if decision3 == "1":
                circleDetector = circle_detector_without_cv.CircleDetectorWithoutCV(image, int(minRadius), int(maxRadius), int(objSize))
            elif decision3 == "2":
                circleDetector = circle_detector_without_cv.CircleDetectorWithoutCV(newImage, int(minRadius), int(maxRadius), int(objSize))

            name = circleDetector.findAllCircles()
            img = Image.open(name)
            img.show()

        elif decision == "0":
            wasEnd = True
            #newImage = image

if __name__ == "__main__":
    main()


