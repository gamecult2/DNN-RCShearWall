import numpy as np
from utils import *
import random
import math
from pathlib import Path
import csv
import time
import os
import sys
import logging
from multiprocessing import Pool, cpu_count, current_process, Lock, Manager
from contextlib import contextmanager

import RCWall_Model_SFI as rcmodel
from RCWall_Parameters_Range import parameter_ranges, loading_ranges
from functions import generate_cyclic_loading_linear, generate_cyclic_loading_exponential

# Define SEQUENCE_LENGTH as a global constant
SEQUENCE_LENGTH = 501
OUTPUT_DIR = Path("RCWall_Data/OriginalData/Run_3")
FORCE_THRESHOLD = 21000
DISP_THRESHOLD = 480


# Set up logging
def setup_logging(level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format='\033[94m%(asctime)s - %(processName)s - %(levelname)s - %(message)s\033[0m',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log')
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def generate_parameters():
    """Generate random parameters based on defined ranges."""
    tw = round(np.random.uniform(*parameter_ranges['tw']))
    tb = round(np.random.uniform(tw, tw * 3))
    lw = round(np.random.uniform(tw * 6, parameter_ranges['lw'][1]) / 10) * 10
    ar = round(np.random.uniform(*parameter_ranges['ar']), 1)
    hw = round(lw * ar, 2)
    lbe = round(np.random.uniform(lw * parameter_ranges['lbe'][0], lw * parameter_ranges['lbe'][1]))
    Ag = round(tw * (lw - (2 * lbe)) + 2 * (tb * lbe), 2)
    fc = round(np.random.uniform(*parameter_ranges['fc']) / 10) * 10
    fyb = round(np.random.uniform(*parameter_ranges['fyb']) / 10) * 10
    fyw = round(np.random.uniform(*parameter_ranges['fyw']) / 10) * 10
    fx = round(np.random.uniform(*parameter_ranges['fx']) / 10) * 10
    rouYb = round(np.random.uniform(*parameter_ranges['rouYb']), 4)
    rouYw = round(np.random.uniform(*parameter_ranges['rouYw']), 4)
    rouXb = round(np.random.uniform(*parameter_ranges['rouXb']), 4)
    rouXw = round(np.random.uniform(*parameter_ranges['rouXw']), 4)
    loadF = round(np.random.uniform(*parameter_ranges['loadF']), 4)

    analysis = np.random.choice(['cyclic', 'pushover'], p=[0.75, 0.25])
    protocol = 0 if analysis == 'cyclic' else 1

    parameter_values = [tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag, protocol]

    # return parameter_values, analysis

    # Generate loading
    num_cycles = np.random.randint(*loading_ranges['num_cycles'])
    max_displacement = max(np.random.randint(hw * 0.005, hw * 0.040), 2)
    repetition_cycles = np.random.randint(*loading_ranges['repetition_cycles'])
    num_points = math.ceil(SEQUENCE_LENGTH / (num_cycles * repetition_cycles))
    # displacement_step = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]

    loading_protocol = np.random.choice(['linear', 'exponential'])

    if loading_protocol == 'linear':
        displacement_step = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]
    else:
        initial_min = max(max_displacement / 32, 1)  # Ensure at least 1 mm
        initial_max = max(max_displacement / 10, initial_min + 1)  # Ensure initial_max > initial_min
        initial_displacement = np.random.randint(initial_min, initial_max)

        displacement_step = generate_cyclic_loading_exponential(num_cycles, initial_displacement, max_displacement, num_points=num_points, repetition_cycles=repetition_cycles)[:SEQUENCE_LENGTH]

    return parameter_values, analysis, displacement_step


def analyse_sample(instance_id, sample_index, run_index=22):
    """Generate a single simulation sample."""
    np.random.seed(hash((instance_id, sample_index, run_index)) & 0xFFFFFFFF)
    worker_id = current_process().name

    logger.info(f"Starting sample {sample_index} on worker {worker_id}")

    # Generate parameters and loading
    parameter_values, analysis, displacement_step = generate_parameters()

    # Extract parameters for model building
    tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag, protocol = parameter_values

    # Run Cyclic Analysis
    eleH, eleL = 14, 12
    rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, eleH, eleL, printProgression=False)
    rcmodel.run_gravity()
    x1, y1, c1, a1, c2, a2 = rcmodel.run_analysis(displacement_step, analysis=analysis, printProgression=False)
    rcmodel.reset_analysis()

    # Validate results
    if len(y1) != SEQUENCE_LENGTH:
        logger.warning(f"Invalid sequence length: {len(y1)}")
        return None

    # Validate results
    if np.any(np.abs(x1) > DISP_THRESHOLD):
        logger.warning("Displacement threshold exceeded")
        return None

    if np.any(np.abs(y1) > FORCE_THRESHOLD):
        logger.warning("Force threshold exceeded")
        return None

    logger.info(f"Sample {sample_index} completed successfully")

    # return parameter_values, displacement_step[:-1], np.concatenate((y1[:1], y1[2:])), c1, a1, c2, a2
    return parameter_values, np.concatenate((x1[:1], x1[2:])), np.concatenate((y1[:1], y1[2:])), c1, a1, c2, a2


def process_samples(instance_id, start_index, end_index):
    """Process samples and write to worker-specific files."""
    valid_samples = 0

    # Create single combined file for this worker
    combined_file_path = os.path.join(OUTPUT_DIR, f"FullData/Worker_{instance_id}_Data.csv")
    cyclic_file_path = os.path.join(OUTPUT_DIR, f"CyclicData/Cyclic_{instance_id}_Data.csv")
    monotonic_file_path = os.path.join(OUTPUT_DIR, f"MonotonicData/Monotonic_{instance_id}_Data.csv")

    # Ensure output directory exists
    os.makedirs(f"{OUTPUT_DIR}/FullData", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/CyclicData", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/MonotonicData", exist_ok=True)

    with open(combined_file_path, 'w', newline='') as combined_file, \
            open(cyclic_file_path, 'w', newline='') as cyclic_file, \
            open(monotonic_file_path, 'w', newline='') as monotonic_file:

        combined_writer = csv.writer(combined_file)
        cyclic_writer = csv.writer(cyclic_file)
        monotonic_writer = csv.writer(monotonic_file)
        try:
            for sample_index in range(start_index, end_index):
                sample_result = analyse_sample(instance_id, sample_index)

                if sample_result is not None:
                    parameter_values, displacement, shear, c1, a1, c2, a2 = sample_result
                    combined_writer.writerows(sample_result[:])

                    # Check the last value of parameter_values
                    if parameter_values[-1] == 0:
                        cyclic_writer.writerows(sample_result[:])  # Save to cyclic file
                    elif parameter_values[-1] == 1:
                        monotonic_writer.writerows(sample_result[:])  # Save to monotonic file

                    valid_samples += 1
                    if valid_samples % 100 == 0:
                        logger.info(f"Worker {instance_id}: Completed {valid_samples} valid samples")

        except Exception as e:
            logger.error(f"Error in worker {instance_id}: {str(e)}")
            raise

    return valid_samples, instance_id


def run_parallel(num_samples, num_processes=cpu_count()):
    """Run simulation in parallel with synchronized data writing."""
    samples_per_process = num_samples // num_processes
    start_time = time.time()
    logger.info(f"Starting parallel simulation with {num_processes} processes")

    # Create and run the process pool within the Manager context
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_samples,
            [(i, i * samples_per_process, (i + 1) * samples_per_process) for i in range(num_processes)]
        )

        valid_samples = sum(result[0] for result in results)
        total_time = time.time() - start_time
        logger.info(f"Simulation completed in {total_time:.2f} seconds with {valid_samples} valid samples")

        return valid_samples, total_time


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) == 3:
        num_samples = int(sys.argv[1])
        num_processes = int(sys.argv[2])
    else:
        num_samples = 100000
        num_processes = cpu_count()

    logger.info(f"Starting simulation with {num_samples} samples using {num_processes} processes")
    valid_samples, execution_time = run_parallel(num_samples, num_processes)

    logger.info(f"Simulation completed successfully:\n"
                f"Valid samples: {valid_samples}\n"
                f"Execution time: {execution_time:.2f} seconds")
