import time
import subprocess
import multiprocessing


def run_generator(instance_id, start_index, end_index):
    subprocess.run(['python', 'RCWall_Data_Generator_Parallel.py', str(instance_id), str(start_index), str(end_index)])


if __name__ == "__main__":
    num_samples = 160000  # Total number of samples
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores

    samples_per_process = num_samples // num_processes
    processes = []

    # Start timer
    start_time = time.time()

    for i in range(num_processes):
        start_index = i * samples_per_process
        end_index = start_index + samples_per_process

        p = multiprocessing.Process(target=run_generator, args=(i, start_index, end_index))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Stop timer
    end_time = time.time()

    # Print total execution time
    total_time = end_time - start_time
    print(f"Total execution time (main process): {total_time:.2f} seconds")