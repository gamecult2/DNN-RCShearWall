import numpy as np
import pandas as pd
from utils import *
import random
import math
import csv
# import RCWall_Cyclic_Model as rcmodel
from GenerateCyclicLoading import *
import RCWall_Cyclic_Model_simple as rcmodel
from RCWall_Cyclic_Parameters import *

random.seed(22)

# ***************************************************************************************************
#           DEFINE NUMBER OF SAMPLE TO GENERATE
# ***************************************************************************************************
# Specify the file path
file_path = "ACI_445B_Shear_Wall_Database.csv"  # Replace with the actual path to your CSV file

# Specify the headers you want to extract
desired_headers = ['ID', 'Ref', 'tw', 'hw', 'lw', 'lbe', 'fc', 'fyb', 'fyw', 'rouYb', 'rouYw', 'loadF', 'Drift', 'Loading', 'Vmax']

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Initialize an empty list to store parameter values for each row
all_parameter_values = []
timeseries_length = 500
# Loop through each row and extract values of specified columns
with open("RCWall_Data/ResultsDatabase.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    for index, row in df.iterrows():
        ID, Ref, tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, Drift, Loading, Vmax = parameter_values = row[desired_headers].tolist()
        print("RUNNING SIMULATION N°", index, "--->", parameter_values)
        # Cyclic load parameters
        num_cycles = 6
        max_displacement = Drift
        repetition_cycles = 2
        num_points = math.ceil(timeseries_length / (num_cycles * repetition_cycles))  # Ensure at least 500 points in total.

        # DisplacementStep = list(generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles))
        DisplacementStep = list(generate_increasing_cyclic_loading_with_repetition(num_cycles,  max_displacement, num_points, repetition_cycles))
        DisplacementStep = DisplacementStep[: timeseries_length]  # Limit displacement of Cyclic analysis to 1000 points

        # Pushover parameters
        DispIncr = max_displacement / timeseries_length  # limit displacement for Pushover analysis to 1000 points
        # Save all samples in the same CSV file
        # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
        # writer.writerow(['InputParameters_values'] + parameter_values)  # The 10 Parameters used for the simulation
        # ***************************************************************************************************
        #           RUN ANALYSIS (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # CYCLIC ANALYSIS
        print("Running Cyclic Analysis")
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        [x1, y1] = rcmodel.run_cyclic(DisplacementStep, plotResults=False, printProgression=False, recordData=False)
        # Check if y1max is empty
        if not max(y1):
            y1max = 0
        else:
            y1max = max(y1)

        print('y1max = ', y1max )
        rcmodel.reset_analysis()
#
        # # RUN PUSHOVER ANALYSIS
        print("Running Pushover Analysis")
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        [x2, y2] = rcmodel.run_pushover(max_displacement, DispIncr, plotResults=False, printProgression=False, recordData=False)
        # Check if y2max is empty
        if not max(y2):
            y2max = 0
        else:
            y2max = max(y2)
        rcmodel.reset_analysis()

        # ***************************************************************************************************
        #           SAVE DATA (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # if 980 <= len(x1) <= 1020:
        # if 2980 <= len(x1) <= 3020 and 2980 <= len(x2) <= 3020:  # Check if the length of the response results is 1000 to write it to the file other results will be removed because of non-convergence

        # Save all samples in the same CSV file
        # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
        writer.writerow(parameter_values + [y1max] + [y2max])  # The 10 Parameters used for the simulation
        # ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
        # writer.writerow([y1max])  # Shear Response of the RC Shear Wall
        # # ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
        # writer.writerow([y2max])  # Pushover Response of the RC Shear Wall

        rcmodel.reset_analysis()



