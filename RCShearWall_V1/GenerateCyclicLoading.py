import math
import numpy as np



def generate_cyclic_load(max_displacement=75):
    duration = 10
    sampling_rate = 50
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)
    # Generate the cyclic load with displacement varying over time
    displacement = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return displacement


def generate_increasing_cyclic_loading(num_cycles=10, initial_displacement=5, max_displacement=60, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # amplitude = initial_displacement + max_displacement_increase * i / num_cycles
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


def generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = max_displacement * (i + 1) / num_cycles
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


def generate_increasing_cyclic_loading_with_exponential_growth(num_cycles, initial_displacement, max_displacement, frequency=1, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # Use exponential growth function for amplitude
        growth_factor = (max_displacement / initial_displacement) ** (1 / (num_cycles - 1))
        amplitude = initial_displacement * growth_factor ** i
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2 * np.pi * frequency * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    # Ensure the last cycle has the final amplitude
    displacement[-num_points * repetition_cycles:] = max_displacement * np.sin(2 * np.pi * frequency * time[-num_points * repetition_cycles:])

    return displacement