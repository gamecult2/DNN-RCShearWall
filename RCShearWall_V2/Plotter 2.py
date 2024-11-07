import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io


def load_data(file_path):
    """Load data from text file"""
    try:
        return np.loadtxt(file_path)
    except:
        print(f"Warning: Could not load {file_path}")
        return None


def process_data(step, eleH, eleL):
    """Process data for a given time step"""
    print(f"Processing step {step}")
    C1 = [[None for _ in range(eleL + 1)] for _ in range(eleH + 1)]
    s1 = [[None for _ in range(eleL + 1)] for _ in range(eleH + 1)]
    s2 = [[None for _ in range(eleL + 1)] for _ in range(eleH + 1)]

    for i in range(1, eleH + 1):
        for j in range(1, eleL + 1):
            cracking_angles = load_data(f'plot/MVLEM_cracking_angle_ele_{i}_panel_{j}.txt')
            interlock1 = load_data(f'plot/MVLEM_strain_stress_concr1_ele_{i}_panel_{j}.txt')
            interlock2 = load_data(f'plot/MVLEM_strain_stress_concr2_ele_{i}_panel_{j}.txt')

            if cracking_angles is not None:
                C1[i][j] = cracking_angles[step, 1:]  # Assuming first column is time
            if interlock1 is not None:
                s1[i][j] = interlock1[step, 1:]  # Assuming first column is time
            if interlock2 is not None:
                s2[i][j] = interlock2[step, 1:]  # Assuming first column is time

    return C1, s1, s2


def plot_crack_pattern(step, eleH, eleL, CRACK_FACTOR=10):
    """Plot crack pattern for a given step"""
    plt.figure(figsize=(5, 8), dpi=100)

    # Define mesh coordinates
    x1 = list(range(0, eleL + 1, 1))
    y1 = list(range(0, eleH + 1, 1))

    # Draw mesh grid
    for y in y1:
        plt.plot(x1, [y] * len(x1), 'k-', linewidth=0.2)
    for x in x1:
        plt.plot([x] * len(y1), y1, 'k-', linewidth=0.2)

    # Calculate center points
    x0 = np.zeros((eleH, eleL))
    y0 = np.zeros((eleH, eleL))
    for i in range(1, eleH + 1):
        for j in range(1, eleL + 1):
            x0[i - 1, j - 1] = j - 0.5
            y0[i - 1, j - 1] = i - 0.5

    # Get data for current step
    C1, s1, s2 = process_data(step, eleH, eleL)

    # Plot cracks
    def plot_cracks(s_matrix, C1, c):
        for i in range(1, eleH + 1):
            for j in range(1, eleL + 1):
                if (C1[i][j] is None or s_matrix[i][j] is None or
                        len(C1[i][j]) == 0 or len(s_matrix[i][j]) == 0):
                    continue

                theta = C1[i][j][0] + np.pi / 2  # Add 90 degrees (π/2 radians) to rotate the crack

                if theta == 10:
                    continue

                # Create points for crack line
                x = np.linspace(j - 1, j, 100)
                k = np.tan(theta)
                b = y0[i - 1, j - 1] - k * x0[i - 1, j - 1]
                y = k * x + b

                # Filter points outside panel boundaries
                mask = (y >= i - 1) & (y <= i)
                if not any(mask):
                    continue

                x = x[mask]
                y = y[mask]

                # Set line width based on strain value
                strain = s_matrix[i][j][0]  # First value after time column
                if strain < 5.0e-5 * CRACK_FACTOR:
                    continue
                elif strain < 1.0e-4 * CRACK_FACTOR:
                    lw = 0.15
                elif strain < 3.0e-4 * CRACK_FACTOR:
                    lw = 0.45
                elif strain < 1e-3 * CRACK_FACTOR:
                    lw = 1.5
                elif strain < 2.0e-3 * CRACK_FACTOR:
                    lw = 3
                else:
                    lw = 4

                plt.plot(x, y, c, linewidth=lw)

    # Plot both sets of cracks
    plot_cracks(s1, C1, 'r')
    plot_cracks(s2, C1, 'b')  # Uncomment to plot second set of cracks

    # Set plot properties
    plt.axis([-1, eleL + 1, -1, eleH + 1])
    plt.axis('off')

    return plt.gcf()


def create_crack_visualization(eleH, eleL, time_steps, output_folder='output'):
    """Main function to create crack visualization"""
    # Create output directories
    os.makedirs(f'{output_folder}/crack_jpeg', exist_ok=True)

    # Store frames for GIF
    frames = []

    # Process each time step
    for step in time_steps:
        # Create plot
        fig = plot_crack_pattern(step, eleH, eleL)

        # Save as JPEG
        plt.savefig(f'{output_folder}/crack_jpeg/crack_{step * 0.001:.4f}ms.jpg',
                    bbox_inches='tight', pad_inches=0, dpi=600)

        # Save frame for GIF
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        frames.append(Image.open(buf))

        plt.close()

    # Create GIF
    frames[0].save(f'{output_folder}/crack_jpeg/crack_animation.gif',
                   save_all=True,
                   append_images=frames[1:],
                   duration=200,
                   loop=0)


if __name__ == "__main__":
    # Define element dimensions
    eleH = 10  # Number of elements in height
    eleL = 10  # Number of elements in length

    # Define time steps
    time_steps = range(0, 760, 4)

    # Create visualization
    create_crack_visualization(eleH, eleL, time_steps)