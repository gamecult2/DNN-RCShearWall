import csv
import random
import math
import numpy as np
import os
import sys
from multiprocessing import current_process

import RCWall_Model_SFI as rcmodel

from RCWall_Parameters_Range import parameter_ranges, loading_ranges
from functions import generate_increasing_cyclic_loading_with_repetition


def write_sample_data(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data[1:])  # Write all rows at once


def generate_sample(instance_id, sample_index, run_index=4):
    # Set the random seed using both instance_id and sample_index
    random.seed(hash((instance_id, sample_index, run_index)) & 0xFFFFFFFF)
    worker_id = current_process().name

    print(f"===================================== RUNNING SAMPLE \033[92mN° {sample_index}\033[0m in Core \033[92mN°{instance_id}\033[0m =====================================\n")

    # Generate parameters
    tw = round(random.uniform(*parameter_ranges['tw']))
    tb = round(random.uniform(tw, parameter_ranges['tb'][1]))
    lw = round(random.uniform(tw * 6, parameter_ranges['lw'][1]) / 10) * 10
    ar = round(random.uniform(*parameter_ranges['ar']))
    hw = round(lw * ar)
    lbe = round(random.uniform(lw * parameter_ranges['lbe'][0], lw * parameter_ranges['lbe'][1]))
    fc = round(random.uniform(*parameter_ranges['fc']))
    fyb = round(random.uniform(*parameter_ranges['fyb']))
    fyw = round(random.uniform(*parameter_ranges['fyw']))
    rouYb = round(random.uniform(*parameter_ranges['rouYb']), 4)
    rouYw = round(random.uniform(*parameter_ranges['rouYw']), 4)
    rouXb = round(random.uniform(*parameter_ranges['rouXb']), 4)
    rouXw = round(random.uniform(*parameter_ranges['rouXw']), 4)
    loadF = round(random.uniform(*parameter_ranges['loadF']), 4)

    # Generate loading
    num_cycles = int(random.uniform(*loading_ranges['num_cycles']))
    max_displacement = int(random.uniform(*loading_ranges['max_displacement']))
    repetition_cycles = int(random.uniform(*loading_ranges['repetition_cycles']))
    num_points = math.ceil(SEQUENCE_LENGTH / (num_cycles * repetition_cycles))
    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]

    parameter_values = [tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF]

    # Run Cyclic Analysis
    rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF, printProgression=False)
    rcmodel.run_gravity()
    x1, y1 = rcmodel.run_analysis(DisplacementStep, printProgression=False)
    rcmodel.reset_analysis()

    if len(y1) == SEQUENCE_LENGTH:
        return worker_id, parameter_values, DisplacementStep[:-1], np.concatenate((y1[:1], y1[2:]))

    return None


def process_samples(instance_id, start_index, end_index):
    valid_samples = 0
    file_path = f"RCWall_Data/Worker_{instance_id}_Data.csv"

    for sample_index in range(start_index, end_index):
        sample_result = generate_sample(instance_id, sample_index)
        if sample_result is not None:
            valid_samples += 1
            write_sample_data(file_path, sample_result)

    return valid_samples


def run_sequential(num_samples):
    random.seed(45)
    with open("RCWall_Data/RCWall_Dataset_FullTest.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        converged = 0
        unconverged = 0
        batch_data = []
        starting_sample_index = 0
        for sample_index in range(starting_sample_index, num_samples):
            sample_result = generate_sample(0, sample_index)
            if sample_result is not None:
                converged += 1
                writer.writerow(sample_result[1])  # parameter_values
                writer.writerow(sample_result[2])  # DisplacementStep[:-1]
                writer.writerow(sample_result[3])  # np.concatenate((y1[:1], y1[2:]))
            else:
                unconverged += 1
            print(f'Converged == {converged} / Unconverged == {unconverged}')


# Global constants
SEQUENCE_LENGTH = 501  # One data point to remove in data preparation

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python RCWall_Data_Generator_Old.py <instance_id> <start_index> <end_index>")
        sys.exit(1)

    instance_id = int(sys.argv[1])
    start_index = int(sys.argv[2])
    end_index = int(sys.argv[3])
    worker_id = f"Worker_{instance_id}"

    valid_samples_count = process_samples(instance_id, start_index, end_index)
    print(f"Data generation complete for {worker_id}. Total valid samples generated: {valid_samples_count}")
