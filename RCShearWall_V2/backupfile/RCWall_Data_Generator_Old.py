import numpy as np
from utils import *
import random
import math
import csv
import time

import RCWall_Model_SFI as rcmodel

from RCWall_Parameters_Range import *


random.seed(45)

# ***************************************************************************************************
#           DEFINE NUMBER OF SAMPLE TO GENERATE
# ***************************************************************************************************
# Define the number of samples to be generated
num_samples = 5
sequence_length = 501
batch_size = 1  # Define the batch size
# Open the CSV file for writing
with open("RCWall_Data/RCWall_Dataset_FullTest.csv", 'a', newline='') as file:
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
        tw = round(random.uniform(*parameter_ranges['tw']))
        tb = round(random.uniform(tw, parameter_ranges['tb'][1]))
        lw = round(random.uniform(tw * 6, parameter_ranges['lw'][1]) / 10) * 10
        ar = round(random.uniform(*parameter_ranges['ar']))
        hw = round((lw * ar))
        lbe = round(random.uniform(lw * parameter_ranges['lbe'][0], lw * parameter_ranges['lbe'][1]))
        fc = round(random.uniform(*parameter_ranges['fc']))
        fyb = round(random.uniform(*parameter_ranges['fyb']))
        fyw = round(random.uniform(*parameter_ranges['fyw']))
        rouYb = round(random.uniform(*parameter_ranges['rouYb']), 4)
        rouYw = round(random.uniform(*parameter_ranges['rouYw']), 4)
        rouXb = round(random.uniform(*parameter_ranges['rouXb']), 4)
        rouXw = round(random.uniform(*parameter_ranges['rouXw']), 4)
        loadF = round(random.uniform(*parameter_ranges['loadF']), 4)

        # Generate cyclic load parameters
        num_cycles = int(random.uniform(*loading_ranges['num_cycles']))
        max_displacement = int(random.uniform(*loading_ranges['max_displacement']))
        repetition_cycles = int(random.uniform(*loading_ranges['repetition_cycles']))
        num_points = math.ceil(sequence_length / (num_cycles * repetition_cycles))

        # DisplacementStep = list(generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles))[:sequence_length]  # Limit to 500
        DisplacementStep = list(generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points, repetition_cycles))[:sequence_length]  # Limit to 500
        # Generate pushover load parameters
        DispIncr = max_displacement / sequence_length  # limit displacement for Pushover analysis to 500 points

        # Overall parameters
        parameter_values = [tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF]
        cyclic_values = [round(value, 4) for value in [max_displacement, repetition_cycles, num_cycles]]
        pushover_values = [round(value, 4) for value in [max_displacement, DispIncr]]

        print(f"\033[92m -> (Characteristic): {parameter_values}")

        # ***************************************************************************************************
        #           RUN ANALYSIS (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # CYCLIC ANALYSIS
        print(f"\033[92m -> (Cyclic Analysis): {cyclic_values}\033[0m --> DisplacementStep: {len(DisplacementStep)}")
        rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        x1, y1 = rcmodel.run_analysis(DisplacementStep, printProgression=False)
        rcmodel.reset_analysis()
        print('len(y1)', len(y1))

        # PUSHOVER ANALYSIS
        # print(f"\033[92m -> (Pushover Analysis): {pushover_values}\033[0m")
        # rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF, printProgression=False)
        # rcmodel.run_gravity(printProgression=False)
        # x2, y2 = rcmodel.run_analysis(DisplacementStep, pushover=True, printProgression=False)
        # rcmodel.reset_analysis()
        # print('len(y2)', len(y2))
        # ***************************************************************************************************
        #           SAVE DATA (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        if len(y1) == sequence_length:  # and len(x2) == sequence_length and not y2_has_negative:
            converged += 1
            writer.writerow(parameter_values)
            writer.writerow(DisplacementStep[:-1])
            writer.writerow(np.concatenate((y1[:1], y1[2:])))

            # batch_data.append(parameter_values)
            # batch_data.append(DisplacementStep[:-1])
            # batch_data.append(np.concatenate((y1[:1], y1[2:])))

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
