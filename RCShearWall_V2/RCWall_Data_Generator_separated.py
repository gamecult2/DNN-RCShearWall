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


class SynchronizedDataWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_locks = {}
        self.buffer_size = 10  # Number of samples to buffer before writing
        self.buffers = {
            'data': [],
            'parameters': [],
            'displacement': [],
            'shear': [],
            'c1': [],
            'a1': [],
            'c2': [],
            'a2': []
        }

    def _get_filepath(self, file_type):
        return self.output_dir / f"{file_type}_Data.csv"

    @contextmanager
    def _get_file_lock(self, file_type):
        if file_type not in self.file_locks:
            self.file_locks[file_type] = Lock()
        lock = self.file_locks[file_type]
        try:
            lock.acquire()
            yield
        finally:
            lock.release()

    def _write_buffer(self, file_type):
        if not self.buffers[file_type]:
            return

        filepath = self._get_filepath(file_type)
        with self._get_file_lock(file_type):
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffers[file_type])
            self.buffers[file_type] = []

    def add_sample(self, sample_data):
        parameter_values, displacement, shear, c1, a1, c2, a2 = sample_data

        # Add to respective buffers
        self.buffers['parameters'].append(parameter_values)
        self.buffers['displacement'].append(displacement)
        self.buffers['shear'].append(shear)
        # self.buffers['c1_a1'].extend([c1, a1])
        # self.buffers['c2_a2'].extend([c2, a2])
        self.buffers['c1'].append(c1)
        self.buffers['a1'].append(a1)
        self.buffers['c2'].append(c2)
        self.buffers['a2'].append(a2)

        # Write buffers if they reach the threshold
        if len(self.buffers['parameters']) >= self.buffer_size:
            for file_type in self.buffers:
                self._write_buffer(file_type)

    def flush(self):
        """Write all remaining data in buffers."""
        for file_type in self.buffers:
            self._write_buffer(file_type)


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

    analysis = np.random.choice(['cyclic', 'pushover'], p=[0.70, 0.30])

    parameter_values = [tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag]

    return parameter_values, analysis


def generate_loading():
    # Generate loading
    num_cycles = np.random.randint(*loading_ranges['num_cycles'])
    max_displacement = np.random.randint(*loading_ranges['max_displacement'])
    repetition_cycles = np.random.randint(*loading_ranges['repetition_cycles'])
    num_points = math.ceil(SEQUENCE_LENGTH / (num_cycles * repetition_cycles))
    # displacement_step = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]

    loading_protocol = np.random.choice(['linear', 'exponential'])

    if loading_protocol == 'linear':
        displacement_step = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]
    else:
        initial_min = max_displacement / 32  # Lower limit
        initial_max = max_displacement / 10  # Upper limit
        initial_displacement = np.random.uniform(initial_min, initial_max)
        displacement_step = generate_cyclic_loading_exponential(num_cycles, initial_displacement, max_displacement, num_points=num_points, repetition_cycles=repetition_cycles)[:SEQUENCE_LENGTH]

    return displacement_step


def analyse_sample(instance_id, sample_index, run_index=6):
    """Generate a single simulation sample."""
    np.random.seed(hash((instance_id, sample_index, run_index)) & 0xFFFFFFFF)
    worker_id = current_process().name

    logger.info(f"Starting sample {sample_index} on worker {worker_id}")

    # Generate parameters and loading
    parameter_values, analysis = generate_parameters()
    displacement_step = generate_loading()

    # Extract parameters for model building
    tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag = parameter_values

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

    if np.any(np.abs(y1) > FORCE_THRESHOLD):
        logger.warning("Force threshold exceeded")
        return None

    logger.info(f"Sample {sample_index} completed successfully")

    # return parameter_values, displacement_step[:-1], np.concatenate((y1[:1], y1[2:])), c1, a1, c2, a2
    return parameter_values, np.concatenate((x1[:1], x1[2:])), np.concatenate((y1[:1], y1[2:])), c1, a1, c2, a2

'''
def process_samples(instance_id, start_index, end_index):
    """Process a batch of samples."""

    valid_samples = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up paths for the different files
    data_path = os.path.join(OUTPUT_DIR, "Full_Data.csv")
    parameter_path = os.path.join(OUTPUT_DIR, "Parameters_Data.csv")
    displacement_path = os.path.join(OUTPUT_DIR, "Displacement_Data.csv")
    shear_path = os.path.join(OUTPUT_DIR, "Shear_Data.csv")
    c1_a1_path = os.path.join(OUTPUT_DIR, "c1_a1_Data.csv")
    c2_a2_path = os.path.join(OUTPUT_DIR, "c2_a2_Data.csv")

    with (open(data_path, 'a', newline='') as data_file, \
          open(parameter_path, 'a', newline='') as parameter_file, \
          open(displacement_path, 'a', newline='') as displacement_file, \
          open(shear_path, 'a', newline='') as shear_file, \
          open(c1_a1_path, 'a', newline='') as c1_a1_file, \
          open(c2_a2_path, 'a', newline='') as c2_a2_file):

        # Set up CSV writers
        data_writer = csv.writer(data_file)
        parameter_writer = csv.writer(parameter_file)
        displacement_writer = csv.writer(displacement_file)
        shear_writer = csv.writer(shear_file)
        c1_a1_writer = csv.writer(c1_a1_file)
        c2_a2_writer = csv.writer(c2_a2_file)

        # Process each sample
        for sample_index in range(start_index, end_index):
            sample_result = analyse_sample(instance_id, sample_index)
            if sample_result is not None:
                parameter_values, displacement, shear, c1, a1, c2, a2 = sample_result
                # Write parameter_values, displacement, shear each on new line
                data_writer.writerows(sample_result[:3])
                parameter_writer.writerow(parameter_values)
                displacement_writer.writerow(displacement)
                shear_writer.writerow(shear)

                # Write c1, a1 each on new line
                c1_a1_writer.writerow(c1)
                c1_a1_writer.writerow(a1)

                # Write c2, a2 each on new line
                c2_a2_writer.writerow(c2)
                c2_a2_writer.writerow(a2)

                valid_samples += 1
                print(f"Total valid samples: {valid_samples}")

    return valid_samples
'''


def process_samples(instance_id, start_index, end_index, data_writer):
    """Process a batch of samples with synchronized writing."""
    valid_samples = 0

    for sample_index in range(start_index, end_index):
        sample_result = analyse_sample(instance_id, sample_index)
        if sample_result is not None:
            data_writer.add_sample(sample_result)
            valid_samples += 1
            print(f"Total valid samples: {valid_samples}")

    data_writer.flush()
    return valid_samples


def run_parallel(num_samples, num_processes=cpu_count()):
    """Run simulation in parallel with synchronized data writing."""
    samples_per_process = num_samples // num_processes
    start_time = time.time()
    logger.info(f"Starting parallel simulation with {num_processes} processes")

    # Create a shared data writer
    data_writer = SynchronizedDataWriter(OUTPUT_DIR)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_samples,
            [(i, i * samples_per_process, (i + 1) * samples_per_process, data_writer)
             for i in range(num_processes)]
        )

    total_time = time.time() - start_time
    logger.info(f"Simulation completed in {total_time:.2f} seconds with "
                f"{sum(results)} valid samples")

    return sum(results), total_time


if __name__ == "__main__":
    if len(sys.argv) == 3:
        num_samples = int(sys.argv[1])
        num_processes = int(sys.argv[2])
    else:
        num_samples = 3000
        num_processes = cpu_count()

    logger.info(f"Starting simulation with {num_samples} samples using {num_processes} processes")
    valid_samples, execution_time = run_parallel(num_samples, num_processes)

    logger.info(f"Simulation completed successfully:\n"
                f"Valid samples: {valid_samples}\n"
                f"Execution time: {execution_time:.2f} seconds")
