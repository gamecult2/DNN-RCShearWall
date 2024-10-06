import math
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ShearWallAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RC-SC Shear Wall Analysis with RCNN")
        self.geometry("1200x800")

        # Variables
        self.material_type = tk.StringVar(value="RC")
        self.parameters = {
            "Tw (mm)": tk.DoubleVar(value=200),
            "Lw (mm)": tk.DoubleVar(value=2830),
            "Hw (mm)": tk.DoubleVar(value=4500),
            "Lbe (mm)": tk.DoubleVar(value=250),
            "fc (MPa)": tk.DoubleVar(value=40),
            "fy (MPa)": tk.DoubleVar(value=510),
            "rouYb": tk.DoubleVar(value=0.029),
            "rouYw": tk.DoubleVar(value=0.003),
            "loadcoef": tk.DoubleVar(value=0.1)
        }
        self.cyclic_params = {
            "n": tk.IntVar(value=6),
            "r": tk.IntVar(value=2),
            "D0": tk.DoubleVar(value=5),
            "Dm": tk.DoubleVar(value=80)
        }

        # GUI Setup
        self.setup_gui()

    def setup_gui(self):
        frame_left = tk.Frame(self)
        frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        frame_right = tk.Frame(self)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_input_section(frame_left)
        self.create_generated_section(frame_right)

    def create_input_section(self, parent):
        frame = tk.LabelFrame(parent, text=f"Structural Design Parameters")
        frame.pack(fill=tk.X, pady=10)

        # RC/SC Switch
        switch_frame = tk.Frame(frame)
        switch_frame.pack(pady=5)
        tk.Label(switch_frame, text="").pack(side=tk.LEFT)
        tk.Radiobutton(switch_frame, text="RC Wall", variable=self.material_type, value="RC").pack(side=tk.LEFT)
        tk.Radiobutton(switch_frame, text="SC Wall", variable=self.material_type, value="SC").pack(side=tk.LEFT)
        tk.Label(switch_frame, text="").pack(side=tk.LEFT)

        # Structural Design Parameters
        for param, var in self.parameters.items():
            param_frame = tk.Frame(frame)
            param_frame.pack(fill=tk.X, pady=2)
            tk.Label(param_frame, text=param, width=15, anchor=tk.W).pack(side=tk.LEFT)
            tk.Entry(param_frame, textvariable=var, width=10).pack(side=tk.LEFT)

        # Cyclic Loading Parameters
        cyclic_frame = tk.LabelFrame(parent, text="Cyclic Loading Parameters")
        cyclic_frame.pack(fill=tk.X, pady=10)

        # Switch variable
        self.protocol_type = tk.StringVar(value="normal")

        # Switch button frame
        switch_frame = tk.Frame(cyclic_frame)
        switch_frame.pack(pady=5)
        tk.Label(switch_frame, text="Protocol:").pack(side=tk.LEFT)
        normal_btn = tk.Radiobutton(switch_frame, text="Normal", variable=self.protocol_type, value="normal")
        normal_btn.pack(side=tk.LEFT)
        exponential_btn = tk.Radiobutton(switch_frame, text="Exponential", variable=self.protocol_type, value="exponential")
        exponential_btn.pack(side=tk.LEFT)

        # Loop through parameters
        for param, var in self.cyclic_params.items():
            param_frame = tk.Frame(cyclic_frame)
            param_frame.pack(fill=tk.X, pady=2)
            tk.Label(param_frame, text=param, width=15, anchor=tk.W).pack(side=tk.LEFT)

            # Entry with conditional protocol based on switch variable
            if self.protocol_type.get() == "normal":
                entry = tk.Entry(param_frame, textvariable=var, width=10)
            else:
                entry = tk.Entry(param_frame, textvariable=tk.DoubleVar(value=np.exp(var.get())), width=10)  # Assuming values are stored in log scale for exponential display
            entry.pack(side=tk.LEFT)
        # Generate Button
        generate_btn = tk.Button(parent, text="Generate", command=self.generate_model)
        generate_btn.pack(pady=10)

    def create_generated_section(self, parent):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        self.fig.tight_layout(pad=5.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_model(self):
        self.update_plots()

    def update_plots(self):
        self.axes[0, 0].clear()
        self.axes[0, 1].clear()
        self.axes[1, 0].clear()
        self.axes[1, 1].clear()

        # Placeholder for actual model generation and calculations
        self.axes[0, 0].set_title("Generated Model")
        # self.axes[0, 0].imshow(plt.imread("https://www.researchgate.net/profile/P-Zakian-2/publication/272182647/figure/fig1/AS:295077579640841@1447363385729/A-special-shear-wall-cross-section_W640.jpg"))  # Replace with actual generated model image

        self.axes[0, 1].set_title("Results Output")
        self.plot_hysteresis_loop(self.axes[0, 1])

        self.axes[1, 0].set_title("Generated Section")
        self.plot_generated_section(self.axes[1, 0])

        self.axes[1, 1].set_title("Generated Cyclic Loading")
        self.plot_cyclic_loading(self.axes[1, 1])

        self.canvas.draw()

    def plot_hysteresis_loop(self, ax):
        # Generate Hysteresis Loop
        disp = np.linspace(-80, 80, 500)
        base_shear = 150 * np.sin(disp / 15)  # Placeholder for actual calculation
        ax.plot(disp, base_shear, label="RCNN Results")
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Base Shear (kN)")
        ax.legend()

    def plot_generated_section(self, ax):
        print('h')
        # ax.imshow(plt.imread("https://www.researchgate.net/profile/P-Zakian-2/publication/272182647/figure/fig1/AS:295077579640841@1447363385729/A-special-shear-wall-cross-section_W640.jpg"))  # Replace with actual generated section image

    def plot_cyclic_loading(self, ax):
        num_cycles = self.cyclic_params["n"].get()
        repetition_cycles = self.cyclic_params["r"].get()
        initial_displacement = self.cyclic_params["D0"].get()
        max_displacement = self.cyclic_params["Dm"].get()
        num_points = math.ceil(500 / (num_cycles * repetition_cycles))

        time = np.linspace(0, num_cycles * repetition_cycles, num_points * num_cycles * repetition_cycles)[: 500]
        displacement = np.zeros_like(time)

        protocol = self.protocol_type.get()
        if protocol == 'normal':
            for i in range(num_cycles):
                amplitude = initial_displacement + (max_displacement - initial_displacement) * i / (num_cycles - 1)
                displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])
        else:
            for i in range(num_cycles):
                # Use exponential growth function for amplitude
                growth_factor = (max_displacement / initial_displacement) ** (1 / (num_cycles - 1))
                amplitude = initial_displacement * growth_factor ** i
                displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])
        ax.plot(time, displacement)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (mm)")


if __name__ == "__main__":
    app = ShearWallAnalysisApp()
    app.mainloop()
