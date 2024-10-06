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

# Define SEQUENCE_LENGTH as a global constant
SEQUENCE_LENGTH = 501
OUTPUT_DIR = "RCWall_Data"


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
    Ag = tw * (lw - (2 * lbe)) + 2 * (tb * lbe)  # Calculate Ag based on provided formula
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

    parameter_values = [tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF, Ag]

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"Worker_{instance_id}_Data.csv")

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for sample_index in range(start_index, end_index):
            sample_result = generate_sample(instance_id, sample_index)
            if sample_result is not None:
                valid_samples += 1
                writer.writerows(sample_result[1:])  # Write parameter_values, DisplacementStep, and y1

    return valid_samples


def run_parallel(num_samples, num_processes=cpu_count()):
    """Run sample generation in parallel mode. Set num_processes to 1 for sequential execution."""
    samples_per_process = num_samples // num_processes
    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_samples,
            [(i, i * samples_per_process, (i + 1) * samples_per_process) for i in range(num_processes)]
        )

    total_time = time.time() - start_time
    print(f"Parallel execution time: {total_time:.2f} seconds, Valid samples: {sum(results)}")

'''
if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command-line execution
        NUM_SAMPLES = int(sys.argv[1])
        NUM_PROCESSES = int(sys.argv[2])
    else:
        # Modify these values to run the generator directly from an IDE
        NUM_SAMPLES = 10
        NUM_PROCESSES = 1 # cpu_count()

    print(f"Running in parallel mode with {NUM_SAMPLES} samples, {NUM_PROCESSES} processes...")
    run_parallel(NUM_SAMPLES, NUM_PROCESSES)
''''''