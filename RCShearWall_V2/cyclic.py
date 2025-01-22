import math
import matplotlib.pyplot as plt
import numpy as np
from functions import *


def plot_strain_history(strain, strain2, num_cycles):
    plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=100)

    # Plot main data series
    plt.plot(strain, color='r', linewidth=1.2, label='Cyclic loading - Linear')
    plt.plot(strain2, color='#3152a1', linewidth=1.2, label='Cyclic loading - Exponential')

    # Find max and min drift ratios
    max_drift = max(max(strain), max(strain2))
    min_drift = min(min(strain), min(strain2))

    # Find initial drift (max value of first cycle)
    init_drift_pos = max(max(strain[:80]), max(strain2[:80]))  # Adjust slice range based on data
    init_drift_neg = min(min(strain[:80]), min(strain2[:80]))  # Negative initial drift

    # Add horizontal lines for drift values
    plt.axhline(y=max_drift, color='black', linestyle='--', linewidth=2, alpha=0.4)
    plt.axhline(y=min_drift, color='black', linestyle='--', linewidth=2, alpha=0.4)
    plt.axhline(y=init_drift_pos, color='black', linestyle='--', linewidth=2, alpha=0.4)
    plt.axhline(y=init_drift_neg, color='black', linestyle='--', linewidth=2, alpha=0.4)

    # Set custom y-axis ticks and labels
    yticks = [min_drift, init_drift_neg, init_drift_pos, max_drift]
    ylabels = ['-Δmin', '-Δinit', 'Δinit', 'Δmax']
    plt.yticks(yticks, ylabels)

    # Add grid and axes lines
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted', alpha=1.0)

    # Set labels and title
    font_settings = {'fontname': 'Times New Roman', 'size': 10}
    plt.xlabel('Cycle Number', **font_settings)
    plt.ylabel('Drift Ratio (%)', **font_settings)

    # Format ticks
    plt.xticks(fontsize=10)

    # Generate the cycle numbers (odd cycles only)
    cycle_numbers = list(range(0, num_cycles+1))
    print(cycle_numbers)
    # Filter out the even cycles (keep only odd cycles)
    odd_cycle_numbers = [cycle for cycle in cycle_numbers if cycle % 2 == 0]
    print(odd_cycle_numbers)
    # Set the x-axis ticks: evenly distribute the ticks over the strain history data
    tick_positions = np.linspace(0, len(strain_history), len(odd_cycle_numbers))  # Distribute evenly for odd cycles
    tick_positions = tick_positions.astype(int)  # Convert to integer positions

    # Adjust x-axis ticks
    plt.xticks(ticks=tick_positions, labels=odd_cycle_numbers, fontsize=10)

    plt.legend(fontsize='small', frameon=True, loc='upper left', bbox_to_anchor=(0.05, 0.95))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('protocol.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()


num_cycles = 7
initial_displacement = 5
max_displacement = 100
num_points = 70
repetition_cycles = 2

strain_history = generate_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles)
strain_history2 = generate_cyclic_loading_exponential(num_cycles, initial_displacement, max_displacement, num_points, repetition_cycles)

print(strain_history2)
plot_strain_history(strain_history, strain_history2, num_cycles*repetition_cycles)
