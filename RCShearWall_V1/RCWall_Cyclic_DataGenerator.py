import numpy as np
from Units import *
import random
import math
import csv

import RCWall_Cyclic_Model as rcmodel
# import RCWall_Cyclic_Model_simple as rcmodel

from RCWall_Cyclic_Parameters import *
from GenerateCyclicLoading import *

random.seed(45)


# (FOR VALIDATION ONLY) Define the parameter ranges
# maxParameters = minParameters = [
#     102 * mm,      # (tw) Minimum thickness
#     3.81 * m,      # (hw) wall height
#     1.22 * m,      # (lw) wall length (min t*6)
#     190 * mm,      # (lbe) BE length (as a percentage of wall length)
#     41.7 * MPa,    # (fc) Concrete Compressive Strength
#     434 * MPa,     # (fyb) Steel Yield Strength BE
#     448 * MPa,     # (fyw) Steel Yield Strength Web
#     0.03,          # (rhoBE) BE long reinforcement ratio (Minimum = 0.01)
#     0.0030,        # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
#     0.0835]           # (loadF) axial load ratio
#
# minLoading = maxLoading = [
#     10,  # num_cycles
#     5 * mm,  # initial_displacement
#     86 * mm,  # max_displacement
#     2]   # repetition_cycles]       # (loadF) Maximum displacement

# ***************************************************************************************************
#           DEFINE NUMBER OF SAMPLE TO GENERATE
# ***************************************************************************************************
# Define the number of samples to be generated
num_samples = 100000
sequence_length = 2000

# Open the CSV file for writing
with open("RCWall_Data/RCWall_Dataset_Full(ShortWall).csv", 'a', newline='') as file:
    writer = csv.writer(file)
    converged = []
    nonconverged = []
    for sample_index in range(num_samples):
        print("RUNNING SIMULATION N°", sample_index)
        # ***************************************************************************************************
        #           GENERATE PARAMETERS FOR EACH SAMPLE
        # ***************************************************************************************************
        # Geometric parameters
        tw = round(random.uniform(minParameters[0], maxParameters[0]))
        hw = round(random.uniform(minParameters[1], maxParameters[1]) / 10) * 10
        lw = round(random.uniform(tw * 6, maxParameters[2]) / 10) * 10
        # lbe = round(random.uniform(minParameters[3], maxParameters[3]))
        lbe = round(random.uniform(lw*minParameters[3], lw*maxParameters[3]))
        fc = round(random.uniform(minParameters[4], maxParameters[4]))
        fyb = round(random.uniform(minParameters[5], maxParameters[5]))
        fyw = round(random.uniform(minParameters[6], maxParameters[6]))
        rouYb = round(random.uniform(minParameters[7], maxParameters[7]), 4)
        rouYw = round(random.uniform(minParameters[8], maxParameters[8]), 4)
        loadCoeff = round(random.uniform(minParameters[9], maxParameters[9]), 4)

        # Cyclic load parameters
        num_cycles = int(random.uniform(minLoading[0], maxLoading[0]))
        max_displacement = int(random.uniform(hw*0.005, hw*0.040))
        repetition_cycles = int(random.uniform(minLoading[2], maxLoading[2]))
        num_points = math.ceil(sequence_length / (num_cycles * repetition_cycles))  # Ensure at least 500 points in total.

        # DisplacementStep = list(generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles))
        DisplacementStep = list(generate_increasing_cyclic_loading_with_repetition(num_cycles,  max_displacement, num_points, repetition_cycles))
        DisplacementStep = DisplacementStep[: sequence_length]  # Limit displacement of Cyclic analysis to 1000 points

        # Pushover parameters
        DispIncr = max_displacement / sequence_length  # limit displacement for Pushover analysis to 1000 points

        # Overall parameters
        parameter_values = [tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff]
        cyclic_values = [round(value, 4) for value in [max_displacement, repetition_cycles, num_cycles]]
        pushover_values = [round(value, 4) for value in [max_displacement, DispIncr]]
        print("\033[92m USED PARAMETERS -> (Characteristic):", parameter_values, "-> (Cyclic Loading):", cyclic_values, "-> (Pushover Loading):", pushover_values, "\033[0m")

        # Save all samples in the same CSV file
        # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
        # writer.writerow(['InputParameters_values'] + parameter_values)  # The 10 Parameters used for the simulation
        # ***************************************************************************************************
        #           RUN ANALYSIS (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # CYCLIC ANALYSIS
        print("\033[92m Running Cyclic Analysis :", cyclic_values, "\033[0m", '--> DisplacementStep :', len(DisplacementStep),)
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        [x1, y1] = rcmodel.run_cyclic(DisplacementStep, plotResults=False, printProgression=False, recordData=False)
        rcmodel.reset_analysis()

        # RUN PUSHOVER ANALYSIS
        print("\033[92m Running Pushover Analysis :", pushover_values, "\033[0m")
        rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        rcmodel.run_gravity(printProgression=False)
        [x2, y2] = rcmodel.run_pushover(max_displacement, DispIncr, plotResults=False, printProgression=False, recordData=False)
        rcmodel.reset_analysis()

        # ***************************************************************************************************
        #           SAVE DATA (CYCLIC + PUSHOVER)
        # ***************************************************************************************************
        # if 980 <= len(x1) <= 1020:
        y2_has_negative = np.any(y2 < 0)  # Check if y2 has any negative number

        if len(x1) == sequence_length and len(x2) == sequence_length and not y2_has_negative:  # Check if the length of the response results is 1000 to write it to the file other results will be removed because of non-convergence
            # if len(x2) == timeseries_length:
            converged.append(sample_index)
            # Save all samples in the same CSV file
            # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
            writer.writerow(['InputParameters_values'] + parameter_values + cyclic_values)  # The 10 Parameters used for the simulation
            writer.writerow(['InputDisplacement_values'] + DisplacementStep)  # Cyclic Displacement imposed to the RC Shear Wall
            # ----------------------- Outputs (Hysteresis Curve - ShearBase Vs Lateral Displacement) -------------------------------------------------------
            x1 = np.delete(x1, 1)
            y1 = np.delete(y1, 1)
            writer.writerow(['OutputCyclicDisplacement_values'] + x1.astype(str).tolist())  # Displacement Response of the RC Shear Wall
            writer.writerow(['OutputCyclicShear_values'] + y1.astype(str).tolist())  # Shear Response of the RC Shear Wall
            # ----------------------- Outputs (Pushover Curve -  ShearBase Vs Lateral Displacement) --------------------------------------------------------
            writer.writerow(['OutputPushoverDisplacement_values'] + x2.astype(str).tolist())  # Displacement Response of the RC Shear Wall
            writer.writerow(['OutputPushoverShear_values'] + y2.astype(str).tolist())  # Pushover Response of the RC Shear Wall
        else:
            nonconverged.append(sample_index)
        rcmodel.reset_analysis()



