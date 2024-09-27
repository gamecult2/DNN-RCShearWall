import numpy as np
import matplotlib.pyplot as plt


def generate_increasing_cyclic_loading_1000(num_cycles, initial_displacement, max_displacement, repetition_cycles):
    target_displacement_step_length = 1000  # Desired displacement step length

    # Calculate the required number of points
    num_points = target_displacement_step_length // (num_cycles * repetition_cycles)

    # Generate the time array
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)

    # Initialize the displacement array
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # Calculate the amplitude for the current cycle
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)

        # Generate the displacement for the current cycle
        displacement_segment = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

        # Add the displacement segment to the overall displacement array
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = displacement_segment

    return time, displacement


def generate_increasing_cyclic_loading(num_cycles=10, initial_displacement=0, max_displacement=60, num_points=50, repetition_cycles=1):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # amplitude = initial_displacement + max_displacement_increase * i / num_cycles
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return time, displacement


def generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = max_displacement * (i + 1) / num_cycles
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return time, displacement


def generate_increasing_cyclic_loading_with_final_amplitude(num_cycles, initial_displacement, max_displacement, frequency=1, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * frequency * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return time, displacement


def generate_increasing_cyclic_loading_with_exponential_growth(num_cycles, initial_displacement, max_displacement, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # Use exponential growth function for amplitude
        growth_factor = (max_displacement / initial_displacement) ** (1 / (num_cycles - 1))
        amplitude = initial_displacement * growth_factor ** i
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    # Ensure the last cycle has the final amplitude
    displacement[-num_points * repetition_cycles:] = max_displacement * np.sin(2 * np.pi * time[-num_points * repetition_cycles:])

    return time, displacement

# Example usage
num_cycles = 1
initial_displacement = 4
max_displacement = 80
repetition_cycles = 1

# generate_increasing_cyclic_loading_with_repetition
time1, displacement1 = generate_increasing_cyclic_loading_with_repetition(num_cycles=1, max_displacement=80, num_points=2000, repetition_cycles=1)[: 500]
time1, displacement1 = generate_increasing_cyclic_loading(num_cycles=6, initial_displacement=2, max_displacement=8, num_points=130, repetition_cycles=2)
time1 = time1
displacement1 = displacement1
print(len(time1))
print(len(displacement1))
# time1, displacement1 = generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, num_points=25, repetition_cycles=repetition_cycles)
# time1, displacement1 = generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points=25, repetition_cycles=repetition_cycles)
# time1, displacement1 = generate_increasing_cyclic_loading_with_repetition(num_cycles, max_displacement, frequency=1, num_points=50, repetition_cycles=repetition_cycles)
# time2, displacement2 = generate_increasing_cyclic_loading_with_final_amplitude(num_cycles, initial_displacement, max_displacement+1, frequency=1, num_points=50, repetition_cycles=repetition_cycles)
# time3, displacement3 = generate_increasing_cyclic_loading_with_exponential_growth(num_cycles, initial_displacement, max_displacement, num_points=50, repetition_cycles=repetition_cycles)
# time4, displacement4 = generate_increasing_cyclic_loading(num_cycles, initial_displacement, max_displacement, num_points=50, repetition_cycles=repetition_cycles)


# Set consistent font settings
plt.rc('font', family='Times New Roman', size=16)  # Set default font for all elements

# Create the plot
plt.figure(figsize=(7/3, 6/3), dpi=100)

# Read test output data (assuming you have it in variables time1, displacement1, time3, displacement3)
plt.plot(time1, displacement1, color='red', linewidth=1.1, label='Cyclic Loading (Linear growth)')
# plt.plot(time3, displacement3, color='blue', linewidth=1.1, label='Cyclic Loading (Exponential growth)')

# Add gridlines and axes
plt.axhline(0, color='black', linewidth=0.4)
plt.axvline(0, color='black', linewidth=0.4)
plt.grid(linestyle='dotted')

# Customize labels and title
# plt.xlabel('Cycle Number')
# plt.ylabel('Displacement (mm)')
# plt.title("Cyclic Loading Protocol", fontweight='normal', size=18)  # Slightly larger title

# plt.xticks(np.arange(min(min(time1), min(time3)), max(max(time1), max(time3)) + 1, 2), rotation=0, ha='right')
# Ensure tight layout and display the plot
plt.tight_layout()
# plt.legend(loc='upper left', fontsize='small')

plt.savefig('CyclicLoadingProtoco.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()
