import numpy as np
import random
import math
import csv
import time

import SCWall_Model as rcmodel

from SCWall_ParametersRange import *
from utils import *

random.seed(1)

# ***************************************************************************************************
#           DEFINE NUMBER OF SAMPLE TO GENERATE
# ***************************************************************************************************
# Define the number of samples to be generated
num_samples = 100
sequence_length = 501
batch_size = 10  # Define the batch size
# Open the CSV file for writing
with open("SCWall_Data/SCWall_Dataset_Full.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    converged = 0
    unconverged = 0
    batch_data = []
    starting_sample_index = 0
    for sample_index in range(starting_sample_index, num_samples):
        print(f"========================================= RUNNING SAMPLE \033[92mN° {sample_index}\033[0m =========================================")
        # ***************************************************************************************************
        #           GENERATE PARAMETERS FOR EACH SAMPLE
        # ***************************************************************************************************
        # Generate geometric parameters for each sample
        tw = round(random.uniform(minParameters[0], maxParameters[0]))
        tb = round(random.uniform(tw, maxParameters[1]))
        hw = round(random.uniform(minParameters[2], maxParameters[2]) / 10) * 10
        lw = round(random.uniform(tw * 6, maxParameters[3]) / 10) * 10
        lbe = round(random.uniform(lw * minParameters[4], lw * maxParameters[4]))
        fc = round(random.uniform(minParameters[5], maxParameters[5]))
        fyb = round(random.uniform(minParameters[6], maxParameters[6]))
        fyw = round(random.uniform(minParameters[7], maxParameters[7]))
        rouYb = round(random.uniform(minParameters[8], maxParameters[8]), 4)
        rouYw = round(random.uniform(minParameters[9], maxParameters[9]), 4)
        loadCoeff = round(random.uniform(minParameters[10], maxParameters[10]), 4)
        numED = int(2 * round(random.uniform(minParameters[11]/2, maxParameters[11]/2)))
        numPT = int(2 * round(random.uniform(minParameters[12]/2, maxParameters[12]/2)))
        areaED = round(random.uniform(minParameters[13], maxParameters[13]) / 2) * 2
        areaPT = round(random.uniform(minParameters[14], maxParameters[14]) / 2) * 2
        fyPT = round(random.uniform(minParameters[15], maxParameters[15]) / 5) * 5
        fyED = round(random.uniform(minParameters[16], maxParameters[16]))

        # Generate cyclic load parameters
        # initial_displacement = int(random.uniform(minLoading[1], maxLoading[1]))
        # max_displacement = int(random.uniform(minLoading[2], maxLoading[2]))
        num_cycles = int(random.uniform(minLoading[0], maxLoading[0]))
        max_displacement = int(random.uniform(hw * 0.005, hw * 0.040))
        repetition_cycles = int(random.uniform(minLoading[2], maxLoading[2]))
        num_points = math.ceil(sequence_length / (num_cycles * repetition_cycles))  # Ensure at least 500 points in total.

        # DisplacementStep = list(generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles))
        DisplacementStep = list(generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points, repetition_cycles))[:sequence_length]  # Limit to 500

        # Generate pushover load parameters
        DispIncr = max_displacement / sequence_length  # limit displacement for Pushover analysis to 500 points

        # Overall parameters
        parameter_values = [tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numED, numPT, areaED, areaPT, fyPT, fyED]
        cyclic_values = [round(value, 4) for value in [max_displacement, repetition_cycles, num_cycles]]
        pushover_values = [round(value, 4) for value in [max_displacement, DispIncr]]

        print(f"\033[92m -> (Characteristic): {parameter_values}")

        # ***************************************************************************************************
        #           RUN ANALYSIS (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # CYCLIC ANALYSIS
        print(f"\033[92m -> (Cyclic Analysis): {cyclic_values}\033[0m --> DisplacementStep: {len(DisplacementStep)}")
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numED, numPT, areaED, areaPT, fyPT, fyED, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        x, y = rcmodel.run_cyclic(DisplacementStep, printProgression=False)
        rcmodel.reset_analysis()
        # y1 = list(range(501))
        # PUSHOVER ANALYSIS
        # print(f"\033[92m -> (Pushover Analysis): {pushover_values}\033[0m")
        # rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF, printProgression=False)
        # rcmodel.run_gravity(printProgression=False)
        # x2, y2 = rcmodel.run_pushover(max_displacement, DispIncr, plotResults=False, printProgression=False, recordData=False)
        # rcmodel.reset_analysis()

        # ***************************************************************************************************
        #           SAVE DATA (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        if len(y) == sequence_length:  # and len(x2) == sequence_length and not y2_has_negative:
            converged += 1
            # writer.writerow(parameter_values)
            # writer.writerow(DisplacementStep[:-1])
            # writer.writerow(np.concatenate((y1[:1], y1[2:])))
            batch_data.append(parameter_values)
            batch_data.append(DisplacementStep[:-1])
            batch_data.append(np.concatenate((y[:1], y[2:])))

            # writer.writerow(np.concatenate((y1[:1], y1[2:])).astype(str).tolist())  # Skip the second element
            # writer.writerow(x2)  # Displacement Response of the RC Shear Wall
            # writer.writerow(y2)  # Pushover Response of the RC Shear Wall
        else:
            unconverged += 1

        print(f'Converged == {converged} / Unconverged == {unconverged}')
        rcmodel.reset_analysis()

        # Write data in batches
        if (sample_index + 1) % batch_size == 0 or sample_index == num_samples - 1:
          for row in batch_data:
              writer.writerow(row)
          batch_data = []  # Clear the batch data after writing
