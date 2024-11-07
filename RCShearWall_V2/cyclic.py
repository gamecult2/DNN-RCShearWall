import math
import matplotlib.pyplot as plt
import numpy as np
from functions import *


def plot_strain_history(strain):
    """
    Plot the strain history.

    Parameters:
    strain (list): List of strain values
    """
    # Create steps array for x-axis
    steps = np.arange(len(strain))

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, strain, 'b-o', linewidth=2, markersize=4)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add labels and title
    plt.xlabel('Step Number')
    plt.ylabel('Strain')
    plt.title('Cyclic Loading Strain History with Cycle Repetitions')

    # Add zero line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Set parameters
    params = {
        'num_cycles': 8,  # Number of different strain amplitudes
        'num_points': 50,  # Number of divisions
        'max_displacement': 86,  # Target strain
        'scale_pos': 1.0,  # Positive scale factor
        'scale_neg': 1.0,  # Negative scale factor
        'repetition_cycles': 2,  # Number of repetitions for each cycle
        'increase_type': 'linear'
    }

    # Generate strain history
    strain_history = generate_cyclic_displacement(**params)
    print(len(strain_history))
    # Plot the results
    plot_strain_history(strain_history)
