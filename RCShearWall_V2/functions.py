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


def tan_to_angle(tan_value):
    if np.isnan(tan_value):
        return np.nan
    return np.degrees((tan_value))


def discretize_loading(n, x):
    xmin = 1.0e20
    xmax = -xmin
    for i in x:
        xmin = min(xmin, i)
        xmax = max(xmax, i)
    Dx = xmax - xmin
    dx = Dx / max(n, 1)
    discretized = [x[0]]
    if abs(dx) > 1.0e-16:
        for i in range(1, len(x)):
            segment_delta = x[i] - x[i - 1]
            n_segment = max(1, int(math.ceil(abs(segment_delta / dx))))
            segment_spacing = segment_delta / n_segment
            iy = x[i - 1]
            for j in range(n_segment):
                iy += segment_spacing
                discretized.append(iy)
    return discretized


def generate_cyclic_displacement(num_cycles, num_points, max_displacement, scale_pos=1.0, scale_neg=1.0, repetition_cycles=1, increase_type='linear', exp_base=2.0):
    # Adjust scales if max displacement is negative
    if max_displacement < 0.0:
        scale_pos, scale_neg = scale_neg, scale_pos

    # Initialize displacement history, starting at zero
    displacement = [0.0]

    # Generate cycles with specified amplitude increase
    for c in range(num_cycles):
        if increase_type.lower() == 'linear':
            # Linear increase
            displacement_c = max_displacement * (c + 1) / num_cycles
        elif increase_type.lower() == 'exponential':
            # Exponential increase
            exp_factor = max_displacement / (exp_base ** (num_cycles - 1))
            displacement_c = exp_factor * (exp_base ** c)

        # Repeat each cycle according to repetition_cycles
        for _ in range(repetition_cycles):
            displacement.extend([
                displacement_c * scale_pos,    # Positive peak
                -displacement_c * scale_neg,   # Negative peak
                0.0                            # Return to zero
            ])

    # Discretize the displacement history
    return discretize_loading(num_points, displacement)


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


def generate_cyclic_loading(num_cycles=10, initial_displacement=5, max_displacement=60, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # amplitude = initial_displacement + max_displacement_increase * i / num_cycles
        amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


def generate_cyclic_loading_linear(num_cycles, max_displacement, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        amplitude = max_displacement * (i + 1) / num_cycles
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    return displacement


def generate_cyclic_loading_exponential(num_cycles, initial_displacement, max_displacement, frequency=1, num_points=50, repetition_cycles=2):
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
    plt.figure(figsize=(4.0, 4.2), dpi=100)
    # plt.figure(figsize=(4 * 1.1, 3 * 1.25))
    # plt.figure(figsize=(7 / 3, 6 / 3), dpi=100)
    # Read test output data to plot
    if plotValidation:
        Test = np.loadtxt(f"DataValidation/{title}.txt", delimiter="\t", unpack="False")
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


def plot_panel_response_animation(eH, eL, Nsteps, resp_per_panel_1, resp_per_panel_2, crack_angles_1, crack_angles_2):
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    # Initialize the figure and the grid data for the animation
    fig, ax = plt.subplots(figsize=(12, 10))
    grid_data = np.zeros((eH, eL))
    im = ax.imshow(grid_data, cmap="coolwarm", interpolation="nearest", aspect="auto", animated=True, vmin=-0.02, vmax=0.02)
    plt.colorbar(im, ax=ax, label="Response")

    # Set up plot labels and configuration
    # title = ax.set_title(f"Response per Panel - Timestep")
    ax.set_xlabel("Panel Index (k) - eL (x-axis)")
    ax.set_ylabel("Panel Index (i) - eH (y-axis)")
    ax.set_xticks(range(eL))
    ax.set_yticks(range(eH))
    ax.invert_yaxis()

    # Create a text array for annotations
    # texts = [[ax.text(k, i, "", ha="center", va="center", color="black", fontsize=8) for k in range(eL)] for i in range(eH)]

    # Create crack lines for each panel (two sets: red and blue)
    lines_1 = [[None for _ in range(eL)] for _ in range(eH)]  # First crack set
    lines_2 = [[None for _ in range(eL)] for _ in range(eH)]  # Second crack set

    for i in range(eH):
        for k in range(eL):
            # Create two lines per panel with different colors
            lines_1[i][k] = Line2D([], [], color='red', animated=True)
            lines_2[i][k] = Line2D([], [], color='blue', animated=True)
            ax.add_line(lines_1[i][k])
            ax.add_line(lines_2[i][k])

    # Add legend
    legend_elements = [Line2D([0], [0], color='red', label='Crack 1'),
                       Line2D([0], [0], color='blue', label='Crack 2')]
    ax.legend(handles=legend_elements)

    def update_crack_line(line, center_x, center_y, angle, length, thickness):
        """Update the crack line position and properties"""
        if np.isnan(angle):
            line.set_data([], [])
            return

        # Calculate line endpoints
        dx = length * np.cos(angle) / 2
        dy = length * np.sin(angle) / 2

        x_data = [center_x - dx, center_x + dx]
        y_data = [center_y - dy, center_y + dy]

        line.set_data(x_data, y_data)
        line.set_linewidth(abs(thickness) * 300)  # Scale thickness for visibility

    # Function to update the frame data
    def update(j):
        # Update grid data for timestep k
        for i in range(eH):
            for k in range(eL):
                panel_key = (i, k)

                # Get responses for both layers
                resp1 = resp_per_panel_1[j].get(panel_key, 0)
                resp2 = resp_per_panel_2[j].get(panel_key, 0)

                # Use average response for the heatmap
                grid_data[i, k] = (resp1 + resp2) / 2

                # Update crack lines for both sets
                crack_angle1 = crack_angles_1[j].get(panel_key, np.nan)
                crack_angle2 = crack_angles_2[j].get(panel_key, np.nan)

                # Calculate center of the panel
                center_x = k
                center_y = i

                # Update both crack lines
                update_crack_line(
                    lines_1[i][k],
                    center_x,
                    center_y,
                    crack_angle1,
                    length=0.9,
                    thickness=resp1
                )

                update_crack_line(
                    lines_2[i][k],
                    center_x,
                    center_y,
                    crack_angle2,
                    length=0.9,
                    thickness=resp2
                )

        # Update the image data
        im.set_array(grid_data)

        # Return all artists that need to be redrawn
        all_artists = [im]
        all_artists.extend([line for row in lines_1 for line in row])
        all_artists.extend([line for row in lines_2 for line in row])
        return all_artists

    # Create the animation with blit=True
    ani = FuncAnimation(
        fig,
        update,
        frames=Nsteps,
        blit=True,
        repeat=True,
        interval=5
    )

    return ani


def plot_deformation_animation(Eds, timeV):
    """
    Creates an animation of the deformed shape.
    """
    import matplotlib.pyplot as plt
    import opsvis as opsv

    fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
                'marker': '', 'markersize': 6}
    anim = opsv.anim_defo(Eds, timeV, 5, fmt_defo=fmt_defo, xlim=[-1000, 1000], ylim=[-1000, 3000])
    return anim


def plot_max_strain_and_angles(max_strains_matrix_1, max_strains_matrix_2, eH, eL):
    """
    Plot maximum strains and corresponding angles for both layers

    Parameters:
    - max_strains_matrix_1: numpy array (eH, eL, 2) for layer 1
    - max_strains_matrix_2: numpy array (eH, eL, 2) for layer 2
    - eH, eL: dimensions of the panel grid
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot max strains for layer 1
    im1 = ax1.imshow(max_strains_matrix_1[:, :, 0], cmap='coolwarm', aspect='auto')
    ax1.set_title('Maximum Strains - Layer 1')
    ax1.set_xlabel('Panel Index (k)')
    ax1.set_ylabel('Panel Index (i)')
    plt.colorbar(im1, ax=ax1, label='Strain')
    ax1.invert_yaxis()

    # Plot corresponding angles for layer 1
    im2 = ax2.imshow(max_strains_matrix_1[:, :, 1], cmap='hsv', aspect='auto')
    ax2.set_title('Corresponding Angles - Layer 1')
    ax2.set_xlabel('Panel Index (k)')
    ax2.set_ylabel('Panel Index (i)')
    plt.colorbar(im2, ax=ax2, label='Angle (rad)')
    ax2.invert_yaxis()

    # Plot max strains for layer 2
    im3 = ax3.imshow(max_strains_matrix_2[:, :, 0], cmap='coolwarm', aspect='auto')
    ax3.set_title('Maximum Strains - Layer 2')
    ax3.set_xlabel('Panel Index (k)')
    ax3.set_ylabel('Panel Index (i)')
    plt.colorbar(im3, ax=ax3, label='Strain')
    ax3.invert_yaxis()

    # Plot corresponding angles for layer 2
    im4 = ax4.imshow(max_strains_matrix_2[:, :, 1], cmap='hsv', aspect='auto')
    ax4.set_title('Corresponding Angles - Layer 2')
    ax4.set_xlabel('Panel Index (k)')
    ax4.set_ylabel('Panel Index (i)')
    plt.colorbar(im4, ax=ax4, label='Angle (rad)')
    ax4.invert_yaxis()

    # Add ticks
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(range(eL))
        ax.set_yticks(range(eH))

    plt.tight_layout()
    plt.show()

    # Return the figure for saving if needed
    return fig


def plot_max_panel_response(eH, eL, max_strains_matrix_1, max_strains_matrix_2):
    from matplotlib.lines import Line2D

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create grid data using average of maximum strains from both layers
    grid_data = (max_strains_matrix_1[:, :, 0] + max_strains_matrix_2[:, :, 0]) / 2

    # Plot heatmap
    im = ax.imshow(grid_data, cmap="coolwarm", interpolation="nearest", aspect="auto", vmin=-0.02, vmax=0.02)
    plt.colorbar(im, ax=ax, label="Maximum Response")

    # Set up plot labels and configuration
    ax.set_xlabel("Panel Index (k) - eL (x-axis)")
    ax.set_ylabel("Panel Index (i) - eH (y-axis)")
    ax.set_xticks(range(eL))
    ax.set_yticks(range(eH))
    ax.invert_yaxis()

    # Create crack lines for each panel
    lines_1 = [[None for _ in range(eL)] for _ in range(eH)]
    lines_2 = [[None for _ in range(eL)] for _ in range(eH)]

    def create_crack_line(center_x, center_y, angle, length, thickness, color):
        """Create a crack line with the given properties"""
        if np.isnan(angle):
            return Line2D([], [], color=color)

        # Calculate line endpoints
        dx = length * np.cos(angle) / 2
        dy = length * np.sin(angle) / 2

        x_data = [center_x - dx, center_x + dx]
        y_data = [center_y - dy, center_y + dy]

        line = Line2D(x_data, y_data, color=color, linewidth=abs(thickness) * 300)
        return line

    # Plot crack lines for maximum strains
    for i in range(eH):
        for k in range(eL):
            # Get maximum strains and angles
            max_strain1 = max_strains_matrix_1[i, k, 0]
            max_angle1 = max_strains_matrix_1[i, k, 1]
            max_strain2 = max_strains_matrix_2[i, k, 0]
            max_angle2 = max_strains_matrix_2[i, k, 1]

            # Create and add lines
            line1 = create_crack_line(k, i, max_angle1, 0.9, max_strain1, 'red')
            line2 = create_crack_line(k, i, max_angle2, 0.9, max_strain2, 'blue')

            ax.add_line(line1)
            ax.add_line(line2)

    # Add legend
    legend_elements = [Line2D([0], [0], color='red', label='Max Crack 1'),
                       Line2D([0], [0], color='blue', label='Max Crack 2')]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()
    return fig
