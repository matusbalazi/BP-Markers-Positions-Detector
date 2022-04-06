import perspective_transformation
import circle_detector_with_cv
import circle_detector_without_cv
import subprocess
from colorama import Fore, Style
import argparse
import sys
import cv2
from PIL import Image
import numpy as np


# Represents application and user interface
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
    newImage = "output_images/afterPerspectiveTransformation.jpg"

    image = None

    while not wasEnd:
        print(Fore.CYAN + "\n----- MAIN MENU -----" + Style.RESET_ALL)
        print("[1] Perspective transformation")
        print("[2] Circle detector w/distance calculator (OpenCV)")
        print("[3] Circle detector w/distance calculator")
        print("[0] Exit detection and start analyzing data")
        print()
        decision = input(Fore.MAGENTA + "Choose an option: " + Style.RESET_ALL)
        print()

        if (len(sys.argv) > 1):
            image = args["image"]
        else:
            if image is None:
                image = str(input(Fore.MAGENTA + "Enter path to the image: " + Style.RESET_ALL))

        if decision == "1":
            if len(sys.argv) > 2:
                width = args["width"]
                height = args["height"]
                coords = args["coords"]
            else:
                width = str(input(
                    Fore.MAGENTA + "Enter width of the image (in pixels) after transformation: " + Style.RESET_ALL))
                height = str(input(
                    Fore.MAGENTA + "Enter height of the image (in pixels) after transformation: " + Style.RESET_ALL))
                coords = str(input(Fore.MAGENTA +
                                   "Enter coordinates of the corners of the object to be transformed (e.g. [(327,303), (1128, 307), ...]): " + Style.RESET_ALL))

            points = np.array(eval(coords), dtype="float32")

            perspective = perspective_transformation.PerspectiveTransformation(image, width, height, points)
            transformedImage = perspective.doTransformation()
            cv2.imshow("Transformed Image", transformedImage)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            status = cv2.imwrite(newImage, transformedImage)
            cv2.waitKey(1000000)

            print(Fore.YELLOW + "Image has been saved to disk:" + Style.RESET_ALL + " ", status)

        elif decision == "2":
            minRadius = 0
            maxRadius = 0
            objSize = 0
            numOfExpectedCircles = 0

            print("\n[1] Original image")
            print("[2] Transformed Image")
            print()
            decision2 = input(Fore.MAGENTA + "Choose an option: " + Style.RESET_ALL)
            print()

            if len(sys.argv) > 9:
                print("Nothing for now")
            else:
                minRadius = input(
                    Fore.MAGENTA + "Enter smallest radius of the circle you are looking for: " + Style.RESET_ALL)
                maxRadius = input(
                    Fore.MAGENTA + "Enter largest radius of the circle you are looking for: " + Style.RESET_ALL)
                objSize = input(Fore.MAGENTA + "Enter circle diameter in reality (in milimeters): " + Style.RESET_ALL)
                numOfExpectedCircles = input(
                    Fore.MAGENTA + "Enter how many circles should be detected in the image: " + Style.RESET_ALL)

            circleDetector = None

            if decision2 == "1":
                circleDetector = circle_detector_with_cv.CircleDetectorWithCV(image, int(minRadius), int(maxRadius),
                                                                              int(objSize), int(numOfExpectedCircles),
                                                                              1)
            elif decision2 == "2":
                circleDetector = circle_detector_with_cv.CircleDetectorWithCV(newImage, int(minRadius), int(maxRadius),
                                                                              int(objSize), int(numOfExpectedCircles),
                                                                              2)

            print("\n[1] Hough Circle Transform")
            print("[2] Hough Circle Transform + Harris Corner Detector")
            print()
            decision3 = input(Fore.MAGENTA + "Choose an option: " + Style.RESET_ALL)
            print()

            img = None
            if decision3 == "1":
                if decision2 == "1":
                    filename = "output_images/detectedCirclesOriginal_HCT.jpg"
                else:
                    filename = "output_images/detectedCirclesTransformed_HCT.jpg"
                value, img = circleDetector.findAllCircles(1, 1)
                if value == 1:
                    cv2.imshow("Image", img)
                else:
                    decision4 = input(
                        Fore.MAGENTA + "Do you want to run advanced circle detection? [Y/N]: " + Style.RESET_ALL)
                    if decision4 == "Y":
                        value, img = circleDetector.findAllCircles(1, 0)
                        cv2.imshow("Image", img)
                    elif decision4 == "N":
                        cv2.imshow("Image", img)
                cv2.imwrite(filename, img)
            elif decision3 == "2":
                if decision2 == "1":
                    filename = "output_images/detectedCirclesOriginal_HCT+HCD.jpg"
                else:
                    filename = "output_images/detectedCirclesTransformed_HCT+HCD.jpg"
                value, img = circleDetector.findAllCircles(2, 1)
                if value == 1:
                    cv2.imshow("Image", img)
                else:
                    decision4 = input(
                        Fore.MAGENTA + "Do you want to run advanced circle detection? [Y/N]: " + Style.RESET_ALL)
                    if decision4 == "Y":
                        value, img = circleDetector.findAllCircles(2, 0)
                        cv2.imshow("Image", img)
                    elif decision4 == "N":
                        cv2.imshow("Image", img)
                cv2.imwrite(filename, img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif decision == "3":
            minRadius = 0
            maxRadius = 0
            objSize = 0
            numOfExpectedCircles = 0

            print("\n[1] Original image")
            print("[2] Transformed Image")
            print()
            decision3 = input(Fore.MAGENTA + "Choose an option: " + Style.RESET_ALL)
            print()

            if len(sys.argv) > 9:
                print("Nothing for now")
            else:
                minRadius = input(
                    Fore.MAGENTA + "Enter smallest radius of the circle you are looking for: " + Style.RESET_ALL)
                maxRadius = input(
                    Fore.MAGENTA + "Enter largest radius of the circle you are looking for: " + Style.RESET_ALL)
                objSize = input(Fore.MAGENTA + "Enter circle diameter in reality (in milimeters): " + Style.RESET_ALL)
                numOfExpectedCircles = input(
                    Fore.MAGENTA + "Enter how many circles should be detected in the image: " + Style.RESET_ALL)

            circleDetector = None

            if decision3 == "1":
                circleDetector = circle_detector_without_cv.CircleDetectorWithoutCV(image, int(minRadius),
                                                                                    int(maxRadius), int(objSize),
                                                                                    int(numOfExpectedCircles), 1)
            elif decision3 == "2":
                circleDetector = circle_detector_without_cv.CircleDetectorWithoutCV(newImage, int(minRadius),
                                                                                    int(maxRadius), int(objSize),
                                                                                    int(numOfExpectedCircles), 2)

            value, name = circleDetector.findAllCircles(1)
            if value == 1:
                img = Image.open(name)
                img.show()
            else:
                decision4 = input(
                    Fore.MAGENTA + "Do you want to run advanced circle detection? [Y/N]: " + Style.RESET_ALL)
                if decision4 == "Y":
                    value, name = circleDetector.findAllCircles(0)
                    img = Image.open(name)
                    img.show()
                elif decision4 == "N":
                    img = Image.open(name)
                    img.show()


        elif decision == "0":
            wasEnd = True

    # Extracts the necessary measurements in the correct format to a file
    file = open("result.txt", "r")
    listOfLines = file.readlines()
    file.close()

    file = open("measures.txt", "w")

    i = 0
    listOfMeasures = []
    for line in listOfLines:
        if i < len(listOfLines) - 1:
            measure = float(line[9:])
            listOfMeasures.append(measure)
            file.write(str(measure) + "\n")
        i += 1

    file.close()

    # Executes an external program, which will find XYZ
    # positions of markers placed on the effector
    subprocess.call("./find_markers_positions.py", shell=True)


if __name__ == "__main__":
    main()
