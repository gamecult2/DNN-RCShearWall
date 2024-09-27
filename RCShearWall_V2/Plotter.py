import os
import numpy as np
import matplotlib.pyplot as plt

# Constants
eleH = 10  # Number of elements in height
eleL = 8  # Number of elements in length
crack_factor = 4
eleL = 8  # Number of elements in length
steps = 1000
dt = 20

def load_data(file_path):
    return np.loadtxt(file_path, delimiter=None)


def process_data(step):
    print(f"Processing step {step}")

    C1 = [[None for _ in range(eleL)] for _ in range(eleH)]
    s1 = [[None for _ in range(eleL)] for _ in range(eleH)]
    s2 = [[None for _ in range(eleL)] for _ in range(eleH)]

    for i in range(1, eleH):
        for j in range(1, eleL):
            cracking_angles = load_data(f'plot/MVLEM_cracking_angle_ele_{i}_panel_{j}.txt')
            interlock1 = load_data(f'plot/MVLEM_strain_stress_concr1_ele_{i}_panel_{j}.txt')
            interlock2 = load_data(f'plot/MVLEM_strain_stress_concr2_ele_{i}_panel_{j}.txt')

            C1[i][j] = cracking_angles[step, 1:]  # Assuming first column is time
            s1[i][j] = interlock1[step, 1:]  # Assuming first column is time
            s2[i][j] = interlock2[step, 1:]  # Assuming first column is time

    return C1, s1, s2


def plot_cracks(step):
    C1, s1, s2 = process_data(step)

    plt.figure(figsize=(5, 8))

    # x1 = [0, 70, 120, 170, 220, 270, 320, 370, 420, 470, 540]
    # y1 = list(range(0, 2251, 50))
    x1 = list(range(0, eleL+1, 1))
    y1 = list(range(0, eleH+1, 1))

    # Plot grid lines
    for y in y1:
        plt.axhline(y, color='k', linewidth=0.5)
    for x in x1:
        plt.axvline(x, color='k', linewidth=0.5)

    # Plot cracks
    for j in range(1, eleL):
        for i in range(1, eleH):
            print('C1', C1)
            theta1 = C1[i][j][0]  # Assuming first value is the angle
            if theta1 == 10:
                continue

            x_center = (x1[j] + x1[j + 1]) / 2
            print('x_center', x_center)
            y_center = (y1[i] + y1[i + 1]) / 2
            print('y_center', y_center)
            x = np.linspace(x1[j], x1[j + 1], 100)
            k = np.tan(theta1)
            b = y_center - k * x_center
            y = k * x + b

            mask = (y > y1[i]) & (y < y1[i + 1])
            x, y = x[mask], y[mask]

            if len(x) == 0:
                continue

            strain = max(s1[i][j][0], s2[i][j][0])  # Assuming first value is the strain
            if strain < 5.0e-5 * crack_factor:
                continue
            elif strain < 1.0e-4 * crack_factor:
                linewidth = 0.15
            elif strain < 3.0e-4 * crack_factor:
                linewidth = 0.45
            elif strain < 1e-3 * crack_factor:
                linewidth = 1.5
            elif strain < 2.0e-3 * crack_factor:
                linewidth = 3
            else:
                linewidth = 4

            plt.plot(x, y, 'r', linewidth=linewidth)

    plt.axis([0-2, eleL+2, 0-1, eleH+2])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    return plt.gcf()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img


def main():
    # os.makedirs('./crack_svg', exist_ok=True)
    os.makedirs('./crack_jpeg', exist_ok=True)

    gif_frames = []

    for step in range(1, steps, dt):
        fig = plot_cracks(step)

        # fig.savefig(f'./crack_svg/crack_{0.001 * step:.4f}ms.svg', format='svg', dpi=1200)
        fig.savefig(f'./crack_jpeg/crack_{0.001 * step:.4f}ms.jpg', format='jpeg', dpi=600)

        img = fig2img(fig)  # Convert figure to image
        gif_frames.append(img)

        plt.close(fig)

    # Save GIF
    gif_frames[0].save('./crack_jpeg/5_25ms.gif', save_all=True, append_images=gif_frames[1:], duration=200, loop=0)


if __name__ == "__main__":
    main()
