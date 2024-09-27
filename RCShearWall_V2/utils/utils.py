import math
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------
# Define units - All results will be in { mm, N, MPa and Sec }
# ------------------------------------------------------------------------
mm = 1.0  # 1 millimeter
N = 1.0  # 1 Newton
sec = 1.0  # 1 second

m = 1000.0 * mm  # 1 meter is 1000 millimeters
cm = 10.0 * mm  # 1 centimeter is 10 millimeters
kN = 1000.0 * N  # 1 kilo-Newton is 1000 Newtons
m2 = m * m  # Square meter
cm2 = cm * cm  # Square centimeter
mm2 = mm * mm  # Square millimeter
MPa = N / mm2  # MegaPascal (Pressure)
kPa = 0.001 * MPa  # KiloPascal (Pressure)
GPa = 1000 * MPa  # GigaPascal (Pressure)


def rebarArea(rebarDiameter):
    a = 3.1416 * (rebarDiameter / 2) ** 2  # compute area
    return a


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


def plotting(x_data, y_data, x_label, y_label, title, save_fig=True, plotValidation=True):
    plt.rcParams.update({'font.size': 10, "font.family": ["Times New Roman", "Cambria"]})

    # Plot Force vs. Displacement
    # plt.figure(figsize=(4.0, 4.2), dpi=100)
    plt.figure(figsize=(4 * 1.1, 3 * 1.25))
    # plt.figure(figsize=(7 / 3, 6 / 3), dpi=100)
    # Read test output data to plot
    if plotValidation:
        Test = np.loadtxt(f"../RCShearWall_V1/DataValidation/{title}.txt", delimiter="\t", unpack="False")
        plt.plot(Test[0, :], Test[1, :], color="black", linewidth=1.0, linestyle="--", label='Experimental Test')

    plt.plot(x_data, y_data, color='red', linewidth=1.2, label='Numerical Test')
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'size': 10}
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.yticks(fontname='Cambria', fontsize=10)
    plt.xticks(fontname='Cambria', fontsize=10)
    plt.title(f"Specimen : {title}", fontdict={'fontname': 'Times New Roman', 'fontstyle': 'normal', 'size': 10})
    plt.tight_layout()
    plt.legend(fontsize='small')

    if save_fig:
        plt.savefig('DataValidation/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')

    plt.show()