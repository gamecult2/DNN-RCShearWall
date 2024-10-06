import numpy as np
from utils import *
import random
import math
import csv
import time
import os
import sys
from multiprocessing import Pool, cpu_count, current_process

import RCWall_Model_SFI as rcmodel
from RCWall_Parameters_Range import parameter_ranges, loading_ranges
from functions import generate_increasing_cyclic_loading_with_repetition

SEQUENCE_LENGTH = 501  # One data point to remove in data preparation


def generate_sample(instance_id, sample_index, run_index=4):
    # Set the random seed using both instance_id and sample_index
    random.seed(hash((instance_id, sample_index, run_index)) & 0xFFFFFFFF)
    worker_id = current_process().name

    print(f"===================================== RUNNING SAMPLE \033[92mN° {sample_index}\033[0m in Core \033[92mN°{instance_id}\033[0m =====================================\n")

    # Generate parameters
    tw = round(np.random.uniform(*parameter_ranges['tw']))
    tb = round(np.random.uniform(tw, parameter_ranges['tb'][1]))
    lw = round(np.random.uniform(tw * 6, parameter_ranges['lw'][1]) / 10) * 10
    ar = round(np.random.uniform(*parameter_ranges['ar']))
    hw = round(lw * ar)
    lbe = round(np.random.uniform(lw * parameter_ranges['lbe'][0], lw * parameter_ranges['lbe'][1]))
    fc = round(np.random.uniform(*parameter_ranges['fc']))
    fyb = round(np.random.uniform(*parameter_ranges['fyb']))
    fyw = round(np.random.uniform(*parameter_ranges['fyw']))
    rouYb = round(np.random.uniform(*parameter_ranges['rouYb']), 4)
    rouYw = round(np.random.uniform(*parameter_ranges['rouYw']), 4)
    rouXb = round(np.random.uniform(*parameter_ranges['rouXb']), 4)
    rouXw = round(np.random.uniform(*parameter_ranges['rouXw']), 4)
    loadF = round(np.random.uniform(*parameter_ranges['loadF']), 4)

    # Generate loading
    num_cycles = int(np.random.uniform(*loading_ranges['num_cycles']))
    max_displacement = int(np.random.uniform(*loading_ranges['max_displacement']))
    repetition_cycles = int(np.random.uniform(*loading_ranges['repetition_cycles']))
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


def run_sequential(num_samples):
    """Run sample generation in a sequential mode."""
    random.seed(45)
    converged, unconverged = 0, 0
    start_time = time.time()

    with open("RCWall_Data/RCWall_Dataset_Sequential.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        for sample_index in range(num_samples):
            sample_result = generate_sample(0, sample_index)
            if sample_result is not None:
                converged += 1
                writer.writerow(sample_result[1])  # parameter_values
                writer.writerow(sample_result[2])  # DisplacementStep[:-1]
                writer.writerow(sample_result[3])  # np.concatenate((y1[:1], y1[2:]))
            else:
                unconverged += 1
            print(f'Converged: {converged}, Unconverged: {unconverged}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time (sequential): {total_time:.2f} seconds")
    print(f"Total samples generated: {converged}")


def process_samples_parallel(args):
    instance_id, start_index, end_index = args
    valid_samples = 0
    file_path = f"RCWall_Data/Worker_{instance_id}_Data.csv"

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for sample_index in range(start_index, end_index):
            sample_result = generate_sample(instance_id, sample_index)
            if sample_result is not None:
                valid_samples += 1
                writer.writerow(sample_result[1])  # parameter_values
                writer.writerow(sample_result[2])  # DisplacementStep[:-1]
                writer.writerow(sample_result[3])  # np.concatenate((y1[:1], y1[2:]))

    return valid_samples


def run_parallel(num_samples):
    num_processes = cpu_count()
    num_processes = 1
    samples_per_process = num_samples // num_processes
    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        args = [(i, i * samples_per_process, (i + 1) * samples_per_process) for i in range(num_processes)]
        results = pool.map(process_samples_parallel, args)

    total_valid_samples = sum(results)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time (parallel): {total_time:.2f} seconds")
    print(f"Total valid samples generated: {total_valid_samples}")


def run_generator(mode, num_samples):
    if "sequential" == mode:
        run_sequential(num_samples)
    elif mode == "parallel":
        run_parallel(num_samples)
    else:
        print("Invalid mode. Use 'sequential' or 'parallel'.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command-line execution
        mode = sys.argv[1]
        num_samples = int(sys.argv[2])
        run_generator(mode, num_samples)
    else:
        # Modify these values to run the generator directly from an IDE
        MODE = "sequential"  # Change to "parallel" for parallel execution
        NUM_SAMPLES = 1000  # Change this to the desired number of samples
        SEQUENCE_LENGTH = 501  # Change this to modify the sequence length

        print(f"Running in {MODE} mode with {NUM_SAMPLES} samples and sequence length {SEQUENCE_LENGTH}...")
        run_generator(MODE, NUM_SAMPLES)
