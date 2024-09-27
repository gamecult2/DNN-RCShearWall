import numpy as np
from Units import *
import random
import math
import csv

import Building_Model as buildmodel
# import RCWall_Cyclic_Model_simple as rcmodel

from Building_Parameters import *


random.seed(45)


# ***************************************************************************************************
#           DEFINE NUMBER OF SAMPLE TO GENERATE
# ***************************************************************************************************
# Define the number of samples to be generated
num_samples = 100000
sequence_length = 501

# Open the CSV file for writing
with open("Building_Dataset_Full.csv", 'a', newline='') as file:
    writer = csv.writer(file)
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

        # GM load parameters

        # Overall parameters
        parameter_values = [tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff]
        print("\033[92m USED PARAMETERS -> (Characteristic):", parameter_values, "-> (Cyclic Loading):", cyclic_values, "-> (Pushover Loading):", pushover_values, "\033[0m")

        # Save all samples in the same CSV file
        # ------------------------ Inputs (Structural Parameters + Cyclic Loading) ---------------------------------------------------------------------
        # writer.writerow(['InputParameters_values'] + parameter_values)  # The 10 Parameters used for the simulation
        # ***************************************************************************************************
        #           RUN ANALYSIS (THA)
        # ***************************************************************************************************
        # THA ANALYSIS
        print("\033[92m Running THA Analysis :", cyclic_values, "\033[0m", '--> DisplacementStep :', len(DisplacementStep),)
        buildmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, printProgression=False)
        [x1, y1] = buildmodel.run_THA(DisplacementStep,,
        buildmodel.reset_analysis()

        # ***************************************************************************************************
        #           SAVE DATA (THA)
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



