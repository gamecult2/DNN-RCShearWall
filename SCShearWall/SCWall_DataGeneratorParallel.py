import csv
import random
import math
import numpy as np
import os
import sys
from multiprocessing import current_process

# Import RCWall_Model and functions
import SCWall_Model as rcmodel
from SCWall_ParametersRange import minParameters, maxParameters, minLoading, maxLoading
from utils import generate_increasing_cyclic_loading_with_repetition

# Global constants
sequence_length = 501


def generate_sample(instance_id, sample_index, run_index=4):
    # Set the random seed using both instance_id and sample_index
    random_seed = hash((instance_id, sample_index, run_index)) % (2 ** 32 - 1)
    print(random_seed)
    random.seed(random_seed)
    worker_id = current_process().name

    print(f"===================================== RUNNING SAMPLE \033[92mN° {sample_index}\033[0m in Core \033[92mN°{instance_id}\033[0m =====================================\n")

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
    numED = int(2 * round(random.uniform(minParameters[11] / 2, maxParameters[11] / 2)))
    numPT = int(2 * round(random.uniform(minParameters[12] / 2, maxParameters[12] / 2)))
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

    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points, repetition_cycles)[:sequence_length]

    parameter_values = [tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numED, numPT, areaED, areaPT, fyPT, fyED]

    # Run analysis
    rcmodel.build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numED, numPT, areaED, areaPT, fyPT, fyED, printProgression=False)
    rcmodel.run_gravity(printProgression=False)
    x, y = rcmodel.run_cyclic(DisplacementStep, printProgression=False)
    rcmodel.reset_analysis()

    if len(y) == sequence_length:
        return worker_id, parameter_values, DisplacementStep[:-1], np.concatenate((y[:1], y[2:]))

    return None


def write_sample_data(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[1])
        writer.writerow(data[2])
        writer.writerow(data[3])


def process_samples(instance_id, start_index, end_index):
    valid_samples = 0
    file_path = f"SCWall_Data/Worker_{instance_id}_Data.csv"
    for sample_index in range(start_index, end_index):
        sample_result = generate_sample(instance_id, sample_index)
        if sample_result is not None:
            valid_samples += 1
            write_sample_data(file_path, sample_result)
    return valid_samples


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python SCWall_DataGenerator.py <instance_id> <start_index> <end_index>")
        sys.exit(1)

    instance_id = int(sys.argv[1])
    start_index = int(sys.argv[2])
    end_index = int(sys.argv[3])
    worker_id = f"Worker_{instance_id}"

    valid_samples_count = process_samples(instance_id, start_index, end_index)

    print(f"Data generation complete for {worker_id}. Total valid samples generated: {valid_samples_count}")
