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
from multiprocessing import Pool, cpu_count, current_process

import RCWall_Model_SFI as rcmodel
from RCWall_Parameters_Range import parameter_ranges, loading_ranges
from functions import generate_cyclic_loading_linear


# Define SEQUENCE_LENGTH as a global constant
SEQUENCE_LENGTH = 501
OUTPUT_DIR = Path("RCWall_Data")
FORCE_THRESHOLD = 27000.0


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
    tb = round(np.random.uniform(tw, parameter_ranges['tb'][1]))
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

    parameter_values = [tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag]

    return parameter_values


def generate_loading():
    # Generate loading
    num_cycles = np.random.randint(*loading_ranges['num_cycles'])
    max_displacement = np.random.randint(*loading_ranges['max_displacement'])
    repetition_cycles = np.random.randint(*loading_ranges['repetition_cycles'])
    num_points = math.ceil(SEQUENCE_LENGTH / (num_cycles * repetition_cycles))
    displacement_step = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]

    return displacement_step


def generate_sample(instance_id, sample_index, run_index=4):
    """Generate a single simulation sample."""
    np.random.seed(hash((instance_id, sample_index, run_index)) & 0xFFFFFFFF)
    worker_id = current_process().name

    logger.info(f"Starting sample {sample_index} on worker {worker_id}")

    # Generate parameters and loading
    parameter_values = generate_parameters()
    displacement_step = generate_loading()

    # Extract parameters for model building
    tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag = parameter_values

    # Run Cyclic Analysis
    rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, printProgression=False)
    rcmodel.run_gravity()
    x1, y1 = rcmodel.run_analysis(displacement_step, analysis='cyclic', printProgression=False)
    rcmodel.reset_analysis()

    # Validate results
    if len(y1) != SEQUENCE_LENGTH:
        logger.warning(f"Invalid sequence length: {len(y1)}")
        return None

    if np.any(np.abs(y1) > FORCE_THRESHOLD):
        logger.warning("Force threshold exceeded")
        return None

    logger.info(f"Sample {sample_index} completed successfully")

    return parameter_values, displacement_step[:-1], np.concatenate((y1[:1], y1[2:]))


def process_samples(instance_id, start_index, end_index):
    """Process a batch of samples."""

    valid_samples = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up paths for the different files
    data_file_path = os.path.join(OUTPUT_DIR, "Full_Data.csv")
    param_file_path = os.path.join(OUTPUT_DIR, "Parameters.csv")

    with open(data_file_path, 'a', newline='') as data_file, \
         open(param_file_path, 'a', newline='') as param_file:
        data_writer = csv.writer(data_file)
        param_writer = csv.writer(param_file)
        for sample_index in range(start_index, end_index):
            sample_result = generate_sample(instance_id, sample_index)
            if sample_result is not None:
                parameter_values, displacement, shear = sample_result
                data_writer.writerows(sample_result[:])

                valid_samples += 1
                print(f"Total valid samples: {valid_samples}")

    return valid_samples


def run_parallel(num_samples, num_processes=cpu_count()):
    """Run simulation in parallel."""
    samples_per_process = num_samples // num_processes
    start_time = time.time()
    logger.info(f"Starting parallel simulation with {num_processes} processes")

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_samples,
            [(i, i * samples_per_process, (i + 1) * samples_per_process) for i in range(num_processes)]
        )

    total_time = time.time() - start_time
    logger.info(f"Simulation completed in {total_time:.2f} seconds with " f"{sum(results)} valid samples")

    return sum(results), total_time


if __name__ == "__main__":
    """Main entry point for the simulation."""
    if len(sys.argv) == 3:
        num_samples = int(sys.argv[1])
        num_processes = int(sys.argv[2])
    else:
        num_samples = 50
        num_processes = 10

    logger.info(f"Starting simulation with {num_samples} samples using {num_processes} processes")
    valid_samples, execution_time = run_parallel(num_samples, num_processes)

    logger.info(f"Simulation completed successfully:\n"
                f"Valid samples: {valid_samples}\n"
                f"Execution time: {execution_time:.2f} seconds")
