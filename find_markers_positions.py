#!/usr/bin/env python3

# This script is used for finding XYZ positions of markers placed on the effector

# As an input are 21 distance measurements between pairs of markers in the usual
# counter clockwise winding order:
#       Nozzle->M0, Nozzle->M1, Nozzle->M2, Nozzle->M3, Nozzle->M4, Nozzle->M5,
#       M0->M1, M0->M2, M0->M3, M0->M4, M0->M5,
#       M1->M2, M1->M3, M1->M4, M1->M5,
#       M2->M3, M2->M4, M2->M5,
#       M3->M4, M3->M5,
#       M4->M5

from __future__ import division
from colorama import Fore, Style
import numpy as np
import scipy.optimize
import argparse
import sys
import xml.etree.cElementTree as ET


# Converts position vector to matrix with nozzle
def positionVectorToMatrixWithNozzle(pPositionVector, pIntermediateSolution):
    return np.append(
        np.array([[0.0, 0.0, 0.0]]), positionVectorToMatrixWithoutNozzle(pIntermediateSolution) - pPositionVector,
        axis=0  # Nozzle
    )


# Converts position vector to matrix without nozzle
def positionVectorToMatrixWithoutNozzle(pPositionVector):
    return np.array(
        [
            [0.0, 0.0, 0.0],  # M0
            [pPositionVector[0], 0.0, 0.0],
            [pPositionVector[1], pPositionVector[2], 0.0],
            [pPositionVector[3], pPositionVector[4], 0.0],
            [pPositionVector[5], pPositionVector[6], 0.0],
            [pPositionVector[7], pPositionVector[8], 0.0],
        ]
    )


# Calculates cost with nozzle measurements
def costWithNozzle(pPositions, pMeasurements):
    # Parameters
    # ----------
    # pPositions : A 7x2 matrix of marker positions
    #             Nozzle is first
    #             Markers are in the usual ccw order:
    #             M0, M1, M2, M3, M4, M5
    # pMeasurements : The 21 distance measurements between pairs of markers
    #                Pairs are in the usual ccw order:
    #                Nozzle-M0, Nozzle-M1, Nozzle-M2, Nozzle-M3, Nozzle-M4, Nozzle-M5
    #                M0-M1, M0-M2, M0-M3, M0-M4, M0-M5,
    #                M1-M2, M1-M3, M1-M4, M1-M5,
    #                M2-M3, M2-M4, M2-M5,
    #                M3-M4, M3-M5,
    #                M4-M5

    return (
            + pow(np.linalg.norm(pPositions[0] - pPositions[1], 2) - pMeasurements[0], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[2], 2) - pMeasurements[1], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[3], 2) - pMeasurements[2], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[4], 2) - pMeasurements[3], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[5], 2) - pMeasurements[4], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[6], 2) - pMeasurements[5], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[2], 2) - pMeasurements[6], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[3], 2) - pMeasurements[7], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[4], 2) - pMeasurements[8], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[5], 2) - pMeasurements[9], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[6], 2) - pMeasurements[10], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[3], 2) - pMeasurements[11], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[4], 2) - pMeasurements[12], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[5], 2) - pMeasurements[13], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[6], 2) - pMeasurements[14], 2)
            + pow(np.linalg.norm(pPositions[3] - pPositions[4], 2) - pMeasurements[15], 2)
            + pow(np.linalg.norm(pPositions[3] - pPositions[5], 2) - pMeasurements[16], 2)
            + pow(np.linalg.norm(pPositions[3] - pPositions[6], 2) - pMeasurements[17], 2)
            + pow(np.linalg.norm(pPositions[4] - pPositions[5], 2) - pMeasurements[18], 2)
            + pow(np.linalg.norm(pPositions[4] - pPositions[6], 2) - pMeasurements[19], 2)
            + pow(np.linalg.norm(pPositions[5] - pPositions[6], 2) - pMeasurements[20], 2)
    )


# Calculates cost without nozzle
def costWithoutNozzle(pPositions, pMeasurements):
    # Parameters
    # ----------
    # pPositions : A 6x2 matrix of marker positions
    #             Nozzle is first
    #             Markers are in the usual ccw order:
    # pMeasurements : The 15 distance measurements between pairs of markers
    #                Pairs are in the usual ccw order:
    #                M0-M1, M0-M2, M0-M3, M0-M4, M0-M5,
    #                M1-M2, M1-M3, M1-M4, M1-M5,
    #                M2-M3, M2-M4, M2-M5,
    #                M3-M4, M3-M5,
    #                M4-M5

    return (
            +pow(np.linalg.norm(pPositions[0] - pPositions[1], 2) - pMeasurements[0], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[2], 2) - pMeasurements[1], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[3], 2) - pMeasurements[2], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[4], 2) - pMeasurements[3], 2)
            + pow(np.linalg.norm(pPositions[0] - pPositions[5], 2) - pMeasurements[4], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[2], 2) - pMeasurements[5], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[3], 2) - pMeasurements[6], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[4], 2) - pMeasurements[7], 2)
            + pow(np.linalg.norm(pPositions[1] - pPositions[5], 2) - pMeasurements[8], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[3], 2) - pMeasurements[9], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[4], 2) - pMeasurements[10], 2)
            + pow(np.linalg.norm(pPositions[2] - pPositions[5], 2) - pMeasurements[11], 2)
            + pow(np.linalg.norm(pPositions[3] - pPositions[4], 2) - pMeasurements[12], 2)
            + pow(np.linalg.norm(pPositions[3] - pPositions[5], 2) - pMeasurements[13], 2)
            + pow(np.linalg.norm(pPositions[4] - pPositions[5], 2) - pMeasurements[14], 2)
    )


# Finds reasonable markers positions based on a set of measurements
def solve(measurements, method):
    print(method)

    marker_measurements = measurements
    if np.size(measurements) == 21:
        marker_measurements = measurements[(21 - 15):]

    # M0 has known positions (0, 0, 0)
    # M1 has unknown x-position
    # All others have unknown xy-positions
    num_params = 0 + 1 + 2 + 2 + 2 + 2

    bound = 1000.0
    lower_bound = [
        0.0,
        0.0,
        0.0,
        -bound,
        0.0,
        -bound,
        0.0,
        -bound,
        0.0,
    ]
    upper_bound = [
        bound,
        bound,
        bound,
        bound,
        bound,
        bound,
        bound,
        bound,
        bound,
    ]

    # This is identical function to costWithoutNozzle,
    # except the shape of inputs
    def costXWithoutNozzle(posvec):
        positions = positionVectorToMatrixWithoutNozzle(posvec)
        return costWithoutNozzle(positions, marker_measurements)

    guess_0 = [0.0] * num_params

    # Here begins optimization methods for finding best intermediate cost
    intermediate_cost = 0.0
    intermediate_solution = []
    if method == "SLSQP":
        sol = scipy.optimize.minimize(
            costXWithoutNozzle,
            guess_0,
            method="SLSQP",
            bounds=list(zip(lower_bound, upper_bound)),
            tol=1e-20,
            options={"disp": True, "ftol": 1e-40, "eps": 1e-10, "maxiter": 500},
        )
        intermediate_cost = sol.fun
        intermediate_solution = sol.x
    elif method == "L-BFGS-B":
        sol = scipy.optimize.minimize(
            costXWithoutNozzle,
            guess_0,
            method="L-BFGS-B",
            bounds=list(zip(lower_bound, upper_bound)),
            options={"disp": True, "ftol": 1e-12, "gtol": 1e-12, "maxiter": 50000, "maxfun": 1000000},
        )
        intermediate_cost = sol.fun
        intermediate_solution = sol.x
    elif method == "PowellDirectionalSolver":
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, And, Or

        solver = PowellDirectionalSolver(num_params)
        solver.SetRandomInitialPoints(lower_bound, upper_bound)
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetTermination(Or(VTR(1e-25), COG(1e-10, 20)))
        solver.SetStrictRanges(lower_bound, upper_bound)
        solver.SetGenerationMonitor(VerboseMonitor(5))
        solver.Solve(costXWithoutNozzle)
        intermediate_cost = solver.bestEnergy
        intermediate_solution = solver.bestSolution
    elif method == "differentialEvolutionSolver":
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, ChangeOverGeneration, And, Or
        from mystic.strategy import Best1Exp, Best1Bin

        stop = Or(VTR(1e-18), ChangeOverGeneration(1e-9, 500))
        npop = 3
        stepmon = VerboseMonitor(100)
        solver = DifferentialEvolutionSolver2(num_params, npop)
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetRandomInitialPoints(lower_bound, upper_bound)
        solver.SetStrictRanges(lower_bound, upper_bound)
        solver.SetGenerationMonitor(stepmon)
        solver.Solve(
            costXWithoutNozzle,
            termination=stop,
            strategy=Best1Bin,
        )
        intermediate_cost = solver.bestEnergy
        intermediate_solution = solver.bestSolution
    else:
        print("Method %s is not supported!" % method)
        sys.exit(1)
    print(Fore.GREEN + "Best intermediate cost:" + Style.RESET_ALL + " ", intermediate_cost)
    print(Fore.GREEN + "Best intermediate positions:" + Style.RESET_ALL + "\n%s" % positionVectorToMatrixWithoutNozzle(
        intermediate_solution))
    intermediatePositions = positionVectorToMatrixWithoutNozzle(intermediate_solution)
    if np.size(measurements) == 15:
        print("Got only 15 samples, so will not try to find nozzle position\n")
        return
    nozzle_measurements = measurements[: (21 - 15)]

    # Look for nozzle's xyz-offset relative to marker 0
    num_params = 3
    lower_bound = [
        0.0,
        0.0,
        -bound,
    ]
    upper_bound = [bound, bound, 0.0]

    # This is identical function to costWithNozzle,
    # except the shape of inputs
    def costXWithNozzle(posvec):
        positions = positionVectorToMatrixWithNozzle(posvec, intermediate_solution)
        return costWithNozzle(positions, measurements)

    # Here begins optimization methods for finding best final cost
    guess_0 = [0.0, 0.0, 0.0]
    final_cost = 0.0
    final_solution = []
    if method == "SLSQP":
        sol = scipy.optimize.minimize(
            costXWithNozzle,
            guess_0,
            method="SLSQP",
            bounds=list(zip(lower_bound, upper_bound)),
            tol=1e-20,
            options={"disp": True, "ftol": 1e-40, "eps": 1e-10, "maxiter": 500},
        )
        final_cost = sol.fun
        final_solution = sol.x
    elif method == "L-BFGS-B":
        sol = scipy.optimize.minimize(
            costXWithNozzle,
            guess_0,
            method="L-BFGS-B",
            bounds=list(zip(lower_bound, upper_bound)),
            options={"disp": True, "ftol": 1e-12, "gtol": 1e-12, "maxiter": 50000, "maxfun": 1000000},
        )
        final_cost = sol.fun
        final_solution = sol.x
    elif method == "PowellDirectionalSolver":
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, And, Or

        solver = PowellDirectionalSolver(num_params)
        solver.SetRandomInitialPoints(lower_bound, upper_bound)
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetTermination(Or(VTR(1e-25), COG(1e-10, 20)))
        solver.SetStrictRanges(lower_bound, upper_bound)
        solver.SetGenerationMonitor(VerboseMonitor(5))
        solver.Solve(costXWithNozzle)
        final_cost = solver.bestEnergy
        final_solution = solver.bestSolution
    elif method == "differentialEvolutionSolver":
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, ChangeOverGeneration, And, Or
        from mystic.strategy import Best1Exp, Best1Bin

        stop = Or(VTR(1e-18), ChangeOverGeneration(1e-9, 500))
        npop = 3
        stepmon = VerboseMonitor(100)
        solver = DifferentialEvolutionSolver2(num_params, npop)
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetRandomInitialPoints(lower_bound, upper_bound)
        solver.SetStrictRanges(lower_bound, upper_bound)
        solver.SetGenerationMonitor(stepmon)
        solver.Solve(
            costXWithNozzle,
            termination=stop,
            strategy=Best1Bin,
        )
        final_cost = solver.bestEnergy
        final_solution = solver.bestSolution

    print()
    z_value = input(
        Fore.MAGENTA + "Please insert Z height (distance between nozzle plane and markers plane): " + Style.RESET_ALL)
    print()

    print(Fore.GREEN + "Best final cost:" + Style.RESET_ALL + " ", final_cost)
    print(Fore.GREEN + "Best final positions:" + Style.RESET_ALL)
    final = positionVectorToMatrixWithNozzle(final_solution, intermediate_solution)[1:]

    # Generates myMarkerParams.xml where are placed XYZ positions of markers on effector
    root = ET.Element("opencv_storage")
    markerPositions = ET.SubElement(root, "marker_positions", type_id="opencv-matrix")
    ET.SubElement(markerPositions, "rows").text = "6"
    ET.SubElement(markerPositions, "cols").text = "3"
    ET.SubElement(markerPositions, "dt").text = "d"
    data = ET.SubElement(markerPositions, "data")
    comment = ET.Comment(" Below Are Effector Markers Positions (REQUIRED) ")
    markerPositions.insert(3, comment)
    matrix = '\n'

    for num in range(0, 6):
        print(
            # "{0: 8.3f} {1: 8.3f} {2: 8.3f} <!-- Marker {3} -->".format(final[num][0], final[num][1], final[num][2], num)
            "{0: 8.3f} {1: 8.3f} {2: 8.3f} <!-- Marker {3} -->".format(final[num][0], final[num][1], float(z_value),
                                                                       num)
        )
        # matrix = matrix + str(final[num][0]) + "\t" + str(final[num][1]) + "\t" + str(final[num][2]) + "\n"
        matrix = matrix + str(round(final[num][0], 3)) + "  " + str(round(final[num][1], 3)) + "  " + str(
            round(float(z_value), 3)) + "\n"

    data.text = matrix

    ET.SubElement(root, "marker_diameter").text = "90.0"
    ET.SubElement(root, "marker_type").text = "disk"

    tlMarkerCenter = ET.SubElement(root, "topleft_marker_center", type_id="opencv-matrix")
    ET.SubElement(tlMarkerCenter, "rows").text = "1"
    ET.SubElement(tlMarkerCenter, "cols").text = "2"
    ET.SubElement(tlMarkerCenter, "dt").text = "d"
    ET.SubElement(tlMarkerCenter, "data").text = "0 0"

    tree = ET.ElementTree(indent(root))
    tree.write("myMarkerParams.xml", xml_declaration=True, encoding="utf-8")

    # HERE BEGINS PART FOR OLDER VERSIONS OF HPM
    # bedMarkers = ET.SubElement(root, "bed_markers", type_id="opencv-matrix")
    # ET.SubElement(bedMarkers, "rows").text = "6"
    # ET.SubElement(bedMarkers, "cols").text = "3"
    # ET.SubElement(bedMarkers, "dt").text = "d"
    # data = ET.SubElement(bedMarkers, "data")
    # comment = ET.Comment(" Below Are Bed Markers Positions (NOT REQUIRED) ")
    # bedMarkers.insert(3, comment)
    # comment = ET.Comment(" these values are only samples ")
    # bedMarkers.insert(4, comment)
    # comment = ET.Comment(" we do not have bed markers in our configuration ")
    # bedMarkers.insert(5, comment)
    # matrix = "\n"
    # for i in intermediatePositions:
    #    matrix = matrix + str(i[0]) + "\t" + str(i[1]) + "\t" + str(i[2]) + "\n"
    # data.text = matrix
    # ET.SubElement(bedMarkers, "marker_diameter").text = "90.0"
    # ET.SubElement(bedMarkers, "marker_type").text = "disk"


# Indents elements in XML file
def indent(elem, level=0):
    i = "\n" + level * "  "
    j = "\n" + (level - 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem


class StoreAsArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(StoreAsArray, self).__call__(parser, namespace, values, option_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure out where hp-markers are compared to the nozzle by looking at the distances between marker to nozzle, and marker to marker."
    )
    parser.add_argument(
        "-m",
        "--method",
        help="Available methods are SLSQP (0, default), L-BFGS-B (1), PowellDirectionalSolver (2), and differentialEvolutionSolver (3). SLSQP and L-BFGS-B require scipy to be installed. The others require mystic to be installed.",
        default="SLSQP",
    )
    parser.add_argument(
        "-e",
        "--measurements",
        help="Specify the 6 measurements of distances between nozzle and marker centers, followed by the 15 measurements of distances between pairs of markers. The latter 15 measurements are the most important ones. Separate numbers by spaces.",
        action=StoreAsArray,
        type=float,
        nargs="+",
        default=np.array([]),
    )
    args = vars(parser.parse_args())
    if args["method"] == "0" or args["method"] == "default":
        args["method"] = "SLSQP"
    if args["method"] == "1":
        args["method"] = "L-BFGS-B"
    if args["method"] == "2":
        args["method"] = "PowellDirectionalSolver"
    if args["method"] == "3":
        args["method"] = "differentialEvolutionSolver"

    # Reads measurements from file and adds to a list
    measurements = args["measurements"]
    if np.size(measurements) == 0:
        file = open("measures.txt", "r")
        listOfLines = file.readlines()
        file.close()

        listOfMeasures = []
        i = 0
        for line in listOfLines:
            if i < len(listOfLines):
                measure = float(line.rstrip("\n"))
                listOfMeasures.append(measure)
            i += 1

        if len(listOfMeasures) == 21:
            measurements = np.array(
                [
                    listOfMeasures[0],
                    listOfMeasures[1],
                    listOfMeasures[2],
                    listOfMeasures[3],
                    listOfMeasures[4],
                    listOfMeasures[5],
                    listOfMeasures[6],
                    listOfMeasures[7],
                    listOfMeasures[8],
                    listOfMeasures[9],
                    listOfMeasures[10],
                    listOfMeasures[11],
                    listOfMeasures[12],
                    listOfMeasures[13],
                    listOfMeasures[14],
                    listOfMeasures[15],
                    listOfMeasures[16],
                    listOfMeasures[17],
                    listOfMeasures[18],
                    listOfMeasures[19],
                    listOfMeasures[20],
                ]
            )
        elif len(listOfMeasures) == 15:
            measurements = np.array(
                [
                    listOfMeasures[0],
                    listOfMeasures[1],
                    listOfMeasures[2],
                    listOfMeasures[3],
                    listOfMeasures[4],
                    listOfMeasures[5],
                    listOfMeasures[6],
                    listOfMeasures[7],
                    listOfMeasures[8],
                    listOfMeasures[9],
                    listOfMeasures[10],
                    listOfMeasures[11],
                    listOfMeasures[12],
                    listOfMeasures[13],
                    listOfMeasures[14],
                ]
            )
        else:
            # Here you can manually add values
            measurements = np.array(
                [

                ]
            )
    if np.size(measurements) != 15 and np.size(measurements) != 21:
        print(
            "Error: You specified %d numbers after your -e/--measurements option, which is not 15 or 21 numbers. It must be 15 or 21 numbers."
        )
        sys.exit(1)

    # Calculates XYZ markers positions based on a set of measurements
    solve(measurements, args["method"])
