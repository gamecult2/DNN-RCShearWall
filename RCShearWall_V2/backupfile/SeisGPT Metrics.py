import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def calculate_metrics(target, output):
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    r2 = r2_score(target, output)
    R, p = pearsonr(target, output)
    return mse, mae, R


def plotting(target, predicted, x_label, floor_numbers, name, wave_name, overall_mse, overall_mae, overall_r, save_fig=False):
    plt.rcParams.update({'font.size': 9, "font.family": ["Cambria", "Cambria"]})

    x_values = np.arange(0, len(target) * 0.02, 0.02)  # Adjust time values if needed
    plt.figure(figsize=(6 * 0.75, 3 * 0.75))
    plt.plot(x_values, target, color='blue', linewidth=0.3, label='FEM')
    plt.plot(x_values, predicted, color='red', linewidth=0.15, label='SeisGPT')

    ax = plt.gca()
    plt.text(0.99, 0.03, f"Overall MSE: {overall_mse:.3f}, Overall MAE: {overall_mae:.3f}, Overall R: {overall_r:.3f}",
             transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')

    y_label = 'Acceleration (m/s²)'
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Update the title to include building name, direction, and wave name
    plt.title(f"Building {name} - Floors {floor_numbers} - Wave {wave_name}",
              fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 9})

    plt.xlim(xmin=0, xmax=max(x_values))
    plt.legend()

    if save_fig:
        folder_path = 'output_figures'
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f"Building_{name}_Floors{floor_numbers}.svg"), format='svg', dpi=300, bbox_inches='tight')

    plt.close()


def read_and_process_files(directory, floor_numbers, name, direction):
    files = os.listdir(directory)

    # Determine the correct files based on the direction
    if direction == 'x':
        fem_files = [f for f in files if f.endswith('7_x_FEM.txt')]
        seisgpt_files = [f for f in files if f.endswith('7_x_SeisGPT.txt')]
    elif direction == 'y':
        fem_files = [f for f in files if f.endswith('7_y_FEM.txt')]
        seisgpt_files = [f for f in files if f.endswith('7_y_SeisGPT.txt')]
    else:
        raise ValueError("Direction must be either 'x' or 'y'.")

    # Prepare to store overall metrics
    overall_mse = []
    overall_mae = []
    overall_r = []

    for fem_file, seisgpt_file in zip(fem_files, seisgpt_files):
        # Extract wave_name from filenames
        wave_name = fem_file.replace(f'{direction}_FEM.txt', '')

        # Define file paths
        fem_path = os.path.join(directory, fem_file)
        seisgpt_path = os.path.join(directory, seisgpt_file)

        # Read data
        fem_data = np.loadtxt(fem_path)
        seisgpt_data = np.loadtxt(seisgpt_path)

        # Print total number of floors based on the number of rows
        total_floors = fem_data.shape[0]
        print(f"Total number of floors in the building for wave '{wave_name}': {total_floors}")

        for floor_number in range(total_floors):
            # Extract data for the specified floor
            fem_floor_data = fem_data[floor_number]  # Assuming floor_number is 1-indexed
            seisgpt_floor_data = seisgpt_data[floor_number]

            # Calculate metrics for each floor
            mse, mae, r = calculate_metrics(fem_floor_data, seisgpt_floor_data)
            overall_mse.append(mse)
            overall_mae.append(mae)
            overall_r.append(r)

    # Calculate overall metrics across all floors and waves
    overall_mse = np.mean(overall_mse)
    overall_mae = np.mean(overall_mae)
    overall_r = np.mean(overall_r)

    print(f"Overall MSE for the building: {overall_mse:.3f}")
    print(f"Overall MAE for the building: {overall_mae:.3f}")
    print(f"Overall R for the building: {overall_r:.3f}")

    # Plotting using the last floor's data for visualization (can be adjusted as needed)
    plotting(fem_data[0], seisgpt_data[0], 'Time (s)', floor_numbers, name, wave_name, overall_mse, overall_mae, overall_r)


# Directory to process
directory = 'seisGPT/Building11case2'
floor_numbers = [1]  # Specify the floor numbers you want to plot
name = '1'  # Specify the building name
direction = 'y'  # Specify the direction ('x' or 'y')

# Process the directory
print(f"Processing directory: {directory} for Floors {floor_numbers} in Direction {direction.upper()}")
read_and_process_files(directory, floor_numbers, name, direction)

print("Processing complete. Check the 'output_figures' folder for results.")
