import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os

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


def calculate_metrics(target, output):
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    r2 = r2_score(target, output)
    R, p = pearsonr(target, output)
    return mse, mae, R


def rebarArea(rebarDiameter):
    a = 3.1416 * (rebarDiameter / 2) ** 2  # compute area
    return a


def tan_to_angle(tan_value):
    if np.isnan(tan_value):
        return np.nan
    return np.degrees((tan_value))


def discretize_loading(n, x):
    # Use math.inf for infinity
    xmin, xmax = math.inf, -math.inf

    # Find the min and max values
    for i in x:
        xmin = min(xmin, i)
        xmax = max(xmax, i)

    # Calculate the spacing between discretized points
    Dx = xmax - xmin
    dx = Dx / max(n, 1)

    # Initialize the discretized list with the first value of x
    discretized = [x[0]]

    # Discretize if the spacing dx is significant
    if abs(dx) > 1.0e-16:
        for i in range(1, len(x)):
            segment_delta = x[i] - x[i - 1]
            n_segment = max(1, int(math.ceil(abs(segment_delta / dx))))  # Avoid division by zero
            segment_spacing = segment_delta / n_segment
            iy = x[i - 1]

            # Append the intermediate values
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
                displacement_c * scale_pos,  # Positive peak
                -displacement_c * scale_neg,  # Negative peak
                0.0  # Return to zero
            ])

    # Discretize the displacement history
    return discretize_loading(num_points, displacement)


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


def generate_cyclic_loading_exponential(num_cycles, initial_displacement, max_displacement, num_points=50, repetition_cycles=2):
    time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)
    displacement = np.zeros_like(time)

    for i in range(num_cycles):
        # Use exponential growth function for amplitude
        growth_factor = (max_displacement / initial_displacement) ** (1 / (num_cycles - 1))
        amplitude = initial_displacement * growth_factor ** i
        displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

    # Ensure the last cycle has the final amplitude
    displacement[-num_points * repetition_cycles:] = max_displacement * np.sin(2 * np.pi * time[-num_points * repetition_cycles:])

    return displacement


def calculate_envelope_curve(x_data, y_data):
    peaks, _ = find_peaks(y_data)
    valleys, _ = find_peaks(-y_data)

    # Collect positive and negative peak points
    pos_x_peaks = x_data[peaks]
    pos_y_peaks = y_data[peaks]
    neg_x_peaks = x_data[valleys]
    neg_y_peaks = y_data[valleys]

    # Create envelope curve points
    envelope_x = np.concatenate((neg_x_peaks[::-1], [0], pos_x_peaks))
    envelope_y = np.concatenate((neg_y_peaks[::-1], [0], pos_y_peaks))
    smooth_window = 2
    envelope_x = np.convolve(
        envelope_x, np.ones(smooth_window) / smooth_window, mode='valid'
    )
    envelope_y = np.convolve(
        envelope_y, np.ones(smooth_window) / smooth_window, mode='valid'
    )
    start_idx = (smooth_window - 1) // 2
    envelope_x = envelope_x[start_idx:start_idx + len(envelope_x)]
    envelope_y = envelope_y[start_idx:start_idx + len(envelope_y)]

    return envelope_x, envelope_y


def plotting(x_data, y_data, x_label, y_label, title, save_fig=True, plotValidation=True, plot_envelope=False):
    plt.rcParams.update({'font.size': 11, "font.family": ["Times New Roman", "Cambria"]})

    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=100)

    # Variables to store max and min values
    exp_max, exp_min = float('-inf'), float('inf')
    num_max, num_min = np.max(y_data), np.min(y_data)

    if plotValidation:
        file_path = f"DataValidation/{title}.txt"
        if os.path.exists(file_path):
            Test = np.loadtxt(file_path, delimiter="\t", unpack="False")
            plt.plot(Test[0, :], Test[1, :], color="#433f40", linewidth=1.1, linestyle="--", label='Experimental Test')

            # Store experimental max and min values
            exp_max = np.max(Test[1, :])
            exp_min = np.min(Test[1, :])

            # Add horizontal lines for experimental max and min
            if plot_envelope:
                plt.axhline(y=exp_max, color='red', linestyle='--', linewidth=1.0)
                plt.axhline(y=exp_min, color='red', linestyle='--', linewidth=1.0)

    plt.plot(x_data, y_data, color='#3152a1', linewidth=1.5, label='Numerical Test')
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    if plot_envelope:
        # Add horizontal lines for numerical max and min
        plt.axhline(y=num_max, color='#3152a1', linestyle='-', linewidth=1.0)
        plt.axhline(y=num_min, color='#3152a1', linestyle='-', linewidth=1.0)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'size': 11}
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.yticks(fontname='Cambria', fontsize=11)
    plt.xticks(fontname='Cambria', fontsize=11)
    plt.title(f"Specimen : {title}", fontdict={'fontname': 'Times New Roman', 'fontstyle': 'normal', 'size': 11})

    plt.legend(fontsize='small')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.90), fontsize='small')

    if save_fig:
        plt.savefig('DataValidation/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# ==============================================================================================================
def plot_panel_response_animation(eH, eL, Nsteps, resp_per_panel_1, resp_per_panel_2, crack_angles_1, crack_angles_2):
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    # grid_data = np.zeros((eH, eL))
    # im = ax.imshow(grid_data, cmap="coolwarm", interpolation="nearest", aspect="auto", animated=True, vmin=-0.02, vmax=0.02)
    # plt.colorbar(im, ax=ax, label="Response")

    # Set up plot labels and configuration
    # title = ax.set_title(f"Response per Panel - Timestep")
    ax.set_xlim(-0.5, eL-0.5)
    ax.set_ylim(-0.5, eH-0.5)
    ax.set_xlabel("Panel Index (eL)")
    ax.set_ylabel("Panel Index (eH)")

    ax.set_xticks(range(eL))
    ax.set_yticks(range(eH))

    # Set grid lines between cells
    ax.set_xticks(np.arange(-0.5, eL, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, eH, 1), minor=True)
    ax.grid(which='major', visible=False)
    ax.grid(which='minor', color='gray', linestyle='-', alpha=0.3)
    # ax.invert_yaxis()

    # Create crack lines for each panel (two sets: red and blue)
    lines_1 = [[None for _ in range(eL)] for _ in range(eH)]  # First crack set
    lines_2 = [[None for _ in range(eL)] for _ in range(eH)]  # Second crack set

    for i in range(eH):
        for k in range(eL):
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
        if np.isnan(angle) or angle == 10:
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

                # Get responses for both layers using indexing (instead of get)
                resp1 = resp_per_panel_1[j, i, k]  # Indexing the array
                resp2 = resp_per_panel_2[j, i, k]  # Indexing the array

                # Use average response for the heatmap
                # grid_data[i, k] = (resp1 + resp2) / 2

                # Update crack lines for both sets
                crack_angle1 = crack_angles_1[j, i, k]  # Indexing the array
                crack_angle2 = crack_angles_2[j, i, k]  # Indexing the array

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
        # im.set_array(grid_data)

        # Return all artists that need to be redrawn
        all_artists = []
        all_artists.extend([line for row in lines_1 for line in row])
        all_artists.extend([line for row in lines_2 for line in row])
        return all_artists

    # Create the animation with blit=True
    ani = FuncAnimation(fig, update, frames=Nsteps, blit=True, repeat=False, interval=5)

    return ani


def plot_max_panel_response(eH, eL, max_cracks_1, max_angles_1, max_cracks_2, max_angles_2, title, ar, save_fig=True):
    from matplotlib.lines import Line2D

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # Reshape the flattened arrays back to their original shape
    max_strains_matrix_1 = np.reshape(np.column_stack((max_cracks_1, max_angles_1)), (eH, eL, 2))
    max_strains_matrix_2 = np.reshape(np.column_stack((max_cracks_2, max_angles_2)), (eH, eL, 2))

    # Initialize the figure ar = hw/lw  === 3
    # fig, ax = plt.subplots(figsize=(3, 5))
    fig, ax = plt.subplots(figsize=(3, 3*ar))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # grid_data = np.maximum(max_strains_matrix_1[:, :, 0], max_strains_matrix_2[:, :, 0])
    # im = ax.imshow(grid_data, cmap="coolwarm", interpolation="nearest", aspect="auto", vmin=-0.02, vmax=0.02)  #
    # plt.colorbar(im, ax=ax, label="Maximum Response")

    ax.set_xlim(-0.5, eL - 0.5)
    ax.set_ylim(-0.5, eH - 0.5)
    ax.set_xlabel("Panel in Lw", fontfamily='Times New Roman', fontsize=10)
    ax.set_ylabel("Panel in Hw", fontfamily='Times New Roman', fontsize=10)

    # Set grid lines between cells
    ax.set_xticks(np.arange(-0.5, eL, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, eH, 1), minor=True)
    ax.grid(which='major', visible=False)
    ax.grid(which='minor', color='gray', linestyle='--', alpha=0.3)
    # ax.invert_yaxis()

    def create_crack_line(center_x, center_y, angle, length, thickness, color):
        if np.isnan(angle) or angle == 10:
            return None

        dx = length * np.cos(angle)/2
        dy = length * np.sin(angle)/2

        x_data = [center_x - dx, center_x + dx]
        y_data = [center_y - dy, center_y + dy]

        line = Line2D(x_data, y_data, color=color, linewidth=abs(thickness) * 100)
        return line

    # Plot crack lines for maximum strains
    for i in range(eH):
        for k in range(eL):
            max_strain1 = max_strains_matrix_1[i, k, 0]
            max_angle1 = max_strains_matrix_1[i, k, 1]
            max_strain2 = max_strains_matrix_2[i, k, 0]
            max_angle2 = max_strains_matrix_2[i, k, 1]

            line1 = create_crack_line(k, i, max_angle1, 0.9, max_strain1, 'red')
            line2 = create_crack_line(k, i, max_angle2, 0.9, max_strain2, 'blue')

            if line1:
                ax.add_line(line1)
            if line2:
                ax.add_line(line2)

    # Add legend
    legend_elements = [Line2D([0], [0], color='red', label='Crack 1'),
                       Line2D([0], [0], color='blue', label='Crack 2')]
    ax.legend(handles=legend_elements, prop={'family': 'Times New Roman', 'size': 10})

    if save_fig:
        plt.savefig('DataValidation/Crack/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    return fig


# ==============================================================================================================
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
# ==============================================================================================================
