import os
import glob
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


def plotting(target, predicted, x_label, floor_number, name, wave_name, mse, mae, r, save_fig=True):
    plt.rcParams.update({'font.size': 9, "font.family": ["Cambria", "Cambria"]})
    skip = 2
    x_values = np.arange(0, len(target) * 0.02, 0.02)[::skip]  # Adjust time values if needed
    plt.figure(figsize=(6 * 0.75, 3 * 0.75))
    # plt.plot(x_values, target, color='blue', linewidth=0.3, label='Numerical')
    # plt.plot(x_values, predicted, color='red', linewidth=0.15, label='SeisGPT')
    plt.plot(x_values, target[::skip], color='blue', linewidth=0.3, label='Numerical')
    plt.plot(x_values, predicted[::skip], color='red', linewidth=0.15, label='SeisGPT')

    ax = plt.gca()
    plt.text(0.99, 0.03, f"MSE: {mse:.3f}, MAE: {mae:.3f}, R: {r:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')

    y_label = 'Displacement (mm)'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    title = f"Building {name} - Floor {floor_number} - Direction {direction.upper()} - {wave_name[0].upper()}{wave_name[1:]}"
    # Update the title to include building name, floor number, direction, and wave name
    plt.title(f"{title}",
              fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 9})

    plt.xlim(xmin=0, xmax=max(x_values))
    plt.legend()

    if save_fig:
        folder_path = 'output_figures'
        os.makedirs(folder_path, exist_ok=True)
        print(folder_path)
        print(title)
        plt.savefig(os.path.join(folder_path, f"{title}.svg"), format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def read_and_process_files(directory, floor_number, name, direction):
    files = os.listdir(directory)
    # Determine the correct files based on the direction
    if direction == 'x':
        fem_file = next(f for f in files if f.endswith('x_FEM.txt'))
        print(fem_file)
        seisgpt_file = next(f for f in files if f.endswith('x_SeisGPT.txt'))
        print(seisgpt_file)
    elif direction == 'y':
        fem_file = next(f for f in files if f.endswith('y_FEM.txt'))
        print(fem_file)
        seisgpt_file = next(f for f in files if f.endswith('y_SeisGPT.txt'))
        print(seisgpt_file)
    else:
        raise ValueError("Direction must be either 'x' or 'y'.")

    # Determine the correct files based on the direction
    if direction == 'x':
        fem_files = [f for f in files if f.endswith('x_FEM.txt')]
        seisgpt_files = [f for f in files if f.endswith('x_SeisGPT.txt')]
    elif direction == 'y':
        fem_files = [f for f in files if f.endswith('y_FEM.txt')]
        seisgpt_files = [f for f in files if f.endswith('y_SeisGPT.txt')]
    else:
        raise ValueError("Direction must be either 'x' or 'y'.")

    # Check if we have at least two files for each category
    if len(fem_files) < 2 or len(seisgpt_files) < 2:
        raise ValueError("Not enough files found for the specified direction.")

    # Select the second file
    fem_file = fem_files[1]  # Select the second FEM file (index 1)
    seisgpt_file = seisgpt_files[1]  # Select the second SeisGPT file (index 1)

    # Extract wave_name from filenames
    wave_name = fem_file.replace(f'_{direction}_FEM.txt', '').replace(f'_{direction}_SeisGPT.txt', '')

    # Define file paths
    fem_path = os.path.join(directory, fem_file)
    seisgpt_path = os.path.join(directory, seisgpt_file)

    # Read data
    fem_data = np.loadtxt(fem_path)
    seisgpt_data = np.loadtxt(seisgpt_path)

    # Print total number of floors based on the number of rows
    total_floors = fem_data.shape[0]
    print(f"Total number of floors in the building: {total_floors}")

    # Extract data for the specified floor
    fem_floor_data = fem_data[floor_number - 1]  # Assuming floor_number is 1-indexed
    seisgpt_floor_data = seisgpt_data[floor_number - 1]

    # Calculate metrics
    mse, mae, r = calculate_metrics(fem_floor_data, seisgpt_floor_data)
    plotting(fem_floor_data, seisgpt_floor_data, 'Time (s)', floor_number, name, wave_name, mse, mae, r)


# Directory to process
directory = 'seisGPT/Building4case1'
floor_number = 4  # Specify the floor number you want to plot
name = '1'  # Specify the building name
direction = 'y'  # Specify the direction ('x' or 'y')

# Process the directory
print(f"Processing directory: {directory} for Floor {floor_number} in Direction {direction.upper()}")
read_and_process_files(directory, floor_number, name, direction)

print("Processing complete. Check the 'output_figures' folder for results.")