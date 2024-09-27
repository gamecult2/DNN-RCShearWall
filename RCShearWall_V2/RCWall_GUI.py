import math
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras import backend as K
# from keras.saving.save import load_model
from utils import *
from RCWall_Data_Processing import *


# # Allocate space for Bidirectional(LSTM)
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#
# # Activate the GPU
# tf.config.list_physical_devices(device_type=None)
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs:", len(physical_devices))


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class ShearWallAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RC Shear Wall Analysis with DNN")
        self.geometry("1200x800")
        # Data Folder Path
        self.data_folder = Path("../RCWall_Data/Dataset_full")

        # Loaded Model Information
        # self.loaded_model = load_model("../DNN_Models/DNN_LSTM-AE(CYCLIC)300k", custom_objects={'r_square': r_square})
        self.model_info = 'LSTM-AE'  # Use 'self.model_info' for clarity

        # Scaler Paths
        self.param_scaler = self.data_folder / 'Scaler/param_scaler.joblib'
        self.disp_cyclic_scaler = self.data_folder / 'Scaler/disp_cyclic_scaler.joblib'
        self.shear_cyclic_scaler = self.data_folder / 'Scaler/shear_cyclic_scaler.joblib'

        # Variables
        self.load_type = tk.StringVar(value="RC")
        self.parameters = {
            "Tw (mm)": tk.DoubleVar(value=102),
            "Tb (mm)": tk.DoubleVar(value=102),
            "Hw (mm)": tk.DoubleVar(value=3810),
            "Lw (mm)": tk.DoubleVar(value=1220),
            "Lbe (mm)": tk.DoubleVar(value=190),
            "fc (MPa)": tk.DoubleVar(value=41.75),
            "fyb (MPa)": tk.DoubleVar(value=434),
            "fyw (MPa)": tk.DoubleVar(value=448),
            "rouYb (%)": tk.DoubleVar(value=0.0294 * 100),
            "rouYw (%)": tk.DoubleVar(value=0.003 * 100),
            "rouXb (%)": tk.DoubleVar(value=0.0294 * 100),
            "rouXw (%)": tk.DoubleVar(value=0.003 * 100),
            "loadcoef": tk.DoubleVar(value=0.092)
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
        self.create_plot_section(frame_right)

    def create_input_section(self, parent):
        frame = tk.LabelFrame(parent, text=f"RC Shear Wall Design Parameters")
        frame.pack(fill=tk.X, pady=10)

        # RC/SC Switch
        switch_frame = tk.Frame(frame)
        switch_frame.pack(pady=5)
        tk.Label(switch_frame, text="Analysis: ").pack(side=tk.LEFT)
        tk.Radiobutton(switch_frame, text="Monotonic", variable=self.load_type, value="RC").pack(side=tk.LEFT)
        tk.Radiobutton(switch_frame, text="Cyclic", variable=self.load_type, value="SC").pack(side=tk.LEFT)
        tk.Label(switch_frame, text="").pack(side=tk.LEFT)

        # Structural Design Parameters with Sliders
        param_ranges = {
            "Tw (mm)": (90, 400),
            "Tb (mm)": (90, 400),
            "Hw (mm)": (1000, 6000),
            "Lw (mm)": (540, 4000),
            "Lbe (mm)": (54, 500),
            "fc (MPa)": (20, 70),
            "fyb (MPa)": (275, 650),
            "fyw (MPa)": (275, 650),
            "rouYb (%)": (0.5, 5.5),
            "rouYw (%)": (0.2, 3.0),
            "rouXb (%)": (0.5, 5.5),
            "rouXw (%)": (0.2, 3.0),
            "loadcoef": (0.01, 0.1)}

        # Structural Design Parameters
        for param, var in self.parameters.items():
            param_frame = tk.Frame(frame)
            param_frame.pack(fill=tk.X, pady=2)
            tk.Label(param_frame, text=param, width=15, anchor=tk.W).pack(side=tk.LEFT)

            slider = tk.Scale(param_frame, showvalue=0, variable=var, from_=param_ranges[param][0], to=param_ranges[param][1], orient=tk.HORIZONTAL, length=200, resolution=0.1 if 'rou' in param else (0.001 if 'load' in param else 1))
            slider.pack(side=tk.LEFT)

            entry = tk.Entry(param_frame, textvariable=var, width=8)
            entry.pack(side=tk.LEFT, padx=5)

        # Cyclic Loading Parameters
        cyclic_frame = tk.LabelFrame(parent, text="Cyclic Loading Parameters")
        cyclic_frame.pack(fill=tk.X, pady=10)

        # Switch variable
        self.protocol_type = tk.StringVar(value="normal")

        # Switch button frame
        switch_frame = tk.Frame(cyclic_frame)
        switch_frame.pack(pady=5)
        tk.Label(switch_frame, text="Protocol: ").pack(side=tk.LEFT)
        normal_btn = tk.Radiobutton(switch_frame, text="Normal", variable=self.protocol_type, value="normal")
        normal_btn.pack(side=tk.LEFT)
        exponential_btn = tk.Radiobutton(switch_frame, text="Exponential", variable=self.protocol_type, value="exponential")
        exponential_btn.pack(side=tk.LEFT)

        # Cyclic Loading Parameters with Sliders and Entries
        cyclic_ranges = {
            "n": (1, 20),
            "r": (1, 10),
            "D0": (0, 10),
            "Dm": (10, 160)}

        for param, var in self.cyclic_params.items():
            param_frame = tk.Frame(cyclic_frame)
            param_frame.pack(fill=tk.X, pady=2)
            tk.Label(param_frame, text=param, width=15, anchor=tk.W).pack(side=tk.LEFT)

            slider = tk.Scale(param_frame, showvalue=0, variable=var, from_=cyclic_ranges[param][0], to=cyclic_ranges[param][1], orient=tk.HORIZONTAL, length=200, resolution=0.01 if 'D' in param else 1)
            slider.pack(side=tk.LEFT)

            entry = tk.Entry(param_frame, textvariable=var, width=8)
            entry.pack(side=tk.LEFT, padx=5)

        # Generate Button
        generate_btn = tk.Button(parent, text="Generate", command=self.update_plots)
        generate_btn.pack(pady=10)

    def create_plot_section(self, parent):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        self.fig.tight_layout(pad=5.0)
        self.axes[0, 0].set_title("RC Shear Wall Elevation")
        self.axes[1, 0].set_title("RC Shear Wall Section")
        self.axes[0, 1].set_title("Results Output")
        self.axes[1, 1].set_title("Cyclic Loading Protocol")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plots(self):
        self.axes[0, 0].clear()
        self.axes[1, 0].clear()
        self.axes[0, 1].clear()
        self.axes[1, 1].clear()

        # Placeholder for actual model generation and calculations
        self.axes[0, 0].set_title("RC Shear Wall Elevation")
        self.plot_shear_wall_elevation(self.axes[0, 0])
        self.axes[1, 0].set_title("RC Shear Wall Section")
        self.plot_shear_wall_section(self.axes[1, 0])
        self.axes[0, 1].set_title("Results Output")
        self.plot_hysteresis_loop(self.axes[0, 1])
        self.axes[1, 1].set_title("Cyclic Loading Protocol")
        self.plot_cyclic_loading(self.axes[1, 1])

        self.canvas.draw()

    def generate_cyclic_loading(self):
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
        if protocol == 'exponential':
            for i in range(num_cycles):
                growth_factor = (max_displacement / initial_displacement) ** (1 / (num_cycles - 1))
                amplitude = initial_displacement * growth_factor ** i
                displacement[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles] = amplitude * np.sin(2.0 * np.pi * time[i * num_points * repetition_cycles: (i + 1) * num_points * repetition_cycles])

        return displacement

    def plot_hysteresis_loop(self, ax):
        # Extract parameters
        tw = self.parameters["Tw (mm)"].get()
        tb = self.parameters["Tb (mm)"].get()
        hw = self.parameters["Hw (mm)"].get()
        lw = self.parameters["Lw (mm)"].get()
        lbe = self.parameters["Lbe (mm)"].get()
        fc = self.parameters["fc (MPa)"].get()
        fyb = self.parameters["fyb (MPa)"].get()
        fyw = self.parameters["fyw (MPa)"].get()
        rouYb = self.parameters["rouYb (%)"].get()
        rouYw = self.parameters["rouYw (%)"].get()
        rouXb = self.parameters["rouXb (%)"].get()
        rouXw = self.parameters["rouXw (%)"].get()
        loadCoeff = self.parameters["loadcoef"].get()

        displacement = self.generate_cyclic_loading()

        # Overall parameters
        parameters_input = np.array((tw, hw, lw, lbe, fc, fyb, fyw, rouYb / 100, rouYw / 100, loadCoeff)).reshape(1, -1)
        print("\033[92m USED PARAMETERS -> (Characteristic):", parameters_input)

        displacement_input = displacement.reshape(1, -1)[:, 1:500 + 1]
        '''
        # ------- Normalize New data ------------------------------------------
        parameters_input = normalize(parameters_input, scaler_filename=self.param_scaler, sequence=False, fit=False)
        displacement_input = normalize(displacement_input, scaler_filename=self.disp_cyclic_scaler, sequence=True, fit=False)

        # ------- Predict New data --------------------------------------------
        predicted_shear = self.loaded_model.predict([parameters_input, displacement_input])

        # ------- Denormalize New data ------------------------------------------
        parameter_values = denormalize(parameters_input, scaler_filename=self.param_scaler, sequence=False)
        DisplacementStep = denormalize(displacement_input, scaler_filename=self.disp_cyclic_scaler, sequence=True)
        predicted_shear = denormalize(predicted_shear, scaler_filename=self.shear_cyclic_scaler, sequence=True)
        predicted_shear -= 45
        
        # Generate Hysteresis Loop
        Test = np.loadtxt(f"../DataValidation/Thomsen_and_Wallace_RW2.txt", delimiter="\t", unpack="False")
        ax.plot(DisplacementStep[-1, 5:499], predicted_shear[-1, 5:499], label="DNN Results")
        ax.plot(Test[0, :], Test[1, :], color="black", linewidth=1.0, linestyle="--", label=f'Reference')
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Base Shear (kN)")
        ax.legend()
        '''

    def plot_cyclic_loading(self, ax):
        displacement = self.generate_cyclic_loading()
        ax.plot(displacement, color='red')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (mm)")

    def plot_shear_wall_elevation(self, ax):
        # Extract parameters
        Tw = self.parameters["Tw (mm)"].get()
        Lw = self.parameters["Lw (mm)"].get()
        Hw = self.parameters["Hw (mm)"].get()
        Lbe = self.parameters["Lbe (mm)"].get()
        fc = self.parameters["fc (MPa)"].get()
        loadcoef = self.parameters["loadcoef"].get()

        Lweb = Lw - (2 * Lbe)
        Aload = (0.85 * abs(fc) * Tw * Lw * loadcoef) / 1000

        # Draw the shear wall section
        BEx1 = [0, Lbe, Lbe, 0, 0]
        BEy1 = [0, 0, Hw, Hw, 0]
        WEBx = [Lbe, Lbe + Lweb, Lbe + Lweb, Lbe, Lbe]
        WEBy = [0, 0, Hw, Hw, 0]
        BEx2 = [Lbe + Lweb, Lbe + Lbe + Lweb, Lbe + Lbe + Lweb, Lbe + Lweb, Lbe + Lweb]
        BEy2 = [0, 0, Hw, Hw, 0]

        # Draw top and bottom
        Tx = [(-Lw * 0.05), (Lw + Lw * 0.05), (Lw + Lw * 0.05), (-Lw * 0.05), (-Lw * 0.05)]
        Ty = [0, 0, (-Hw * 0.15), (-Hw * 0.15), 0]

        Bx = [(-Lw * 0.05), (Lw + Lw * 0.05), (Lw + Lw * 0.05), (-Lw * 0.05), (-Lw * 0.05)]
        By = [Hw, Hw, (Hw + Hw * 0.15), (Hw + Hw * 0.15), Hw]

        ax.plot(Tx, Ty, 'black')
        ax.plot(Bx, By, 'black')
        ax.plot(BEx1, BEy1, 'black')
        ax.plot(WEBx, WEBy, 'black')
        ax.plot(BEx2, BEy2, 'black')
        ax.fill(BEx1, BEy1, 'grey')
        ax.fill(WEBx, WEBy, 'lightgrey')
        ax.fill(BEx2, BEy2, 'grey')

        # Add concentrated load arrow at the top of the shear wall
        load_x = Lw / 2
        load_y_start = Hw + Hw * 0.22
        load_y_end = Hw + Hw * 0.15
        ax.arrow(load_x, load_y_start, 0, -0.15 * Hw, head_width=(Lw * 0.07), head_length=(Hw * 0.07), fc='blue', ec='blue')
        ax.text(load_x, load_y_end + 0.14 * Hw, f'N = {Aload} kN', ha='center', va='center', color='blue')

        disp_x = - Lw * 0.25
        disp_y_start = Hw + Hw * 0.075
        disp_y_end = Hw + Hw * 0.15
        ax.arrow(disp_x, disp_y_start, Lw * 0.12, 0, head_width=(Hw * 0.07), head_length=(Lw * 0.07), fc='red', ec='red')
        ax.text(disp_x, disp_y_end + 0.14 * Hw, f'Dm', ha='center', va='center', color='red')

        # ax.set_aspect('equal', adjustable='box')
        # ax.set_xlim([-Lw * 0.40, Lw + Lw * 0.20])
        # ax.set_ylim([-Hw * 0.25, Hw * 1.5])
        ax.set_xlim([-4000 * 0.25, Lw + (4000 * 0.25)])
        ax.set_ylim([-6000 * 0.20, 6000 + (6000 * 0.40)])
        # ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Height (mm)")

    def plot_shear_wall_section(self, ax):
        # Extract parameters
        Tw = self.parameters["Tw (mm)"].get()
        Tb = self.parameters["Tb (mm)"].get()
        Lw = self.parameters["Lw (mm)"].get()
        Lbe = self.parameters["Lbe (mm)"].get()
        Lweb = Lw - (2 * Lbe)
        half = True

        if half:
            # Draw the shear wall section
            BEx1 = [0, Lbe, Lbe, 0, 0]
            BEy1 = [y - (Tb - Tw) / 2 for y in [0, 0, Tb, Tb, 0]]
            WEBx = [Lbe, Lbe + Lweb / 2, Lbe + Lweb / 2, Lbe, Lbe]
            WEBy = [0, 0, Tw, Tw, 0]

            ax.plot(BEx1, BEy1, 'black')
            ax.plot(WEBx, WEBy, 'black')
            ax.fill(BEx1, BEy1, 'grey')
            ax.fill(WEBx, WEBy, 'lightgrey')

        else:
            # Draw the shear wall section
            BEx1 = [0, Lbe, Lbe, 0, 0]
            BEy1 = [0, 0, Tb, Tb, 0]
            WEBx = [Lbe, Lbe + Lweb, Lbe + Lweb, Lbe, Lbe]
            WEBy = [0, 0, Tw, Tw, 0]
            BEx2 = [Lbe + Lweb, Lbe + Lbe + Lweb, Lbe + Lbe + Lweb, Lbe + Lweb, Lbe + Lweb]
            BEy2 = [0, 0, Tb, Tb, 0]

            ax.plot(BEx1, BEy1, 'black')
            ax.plot(WEBx, WEBy, 'black')
            ax.plot(BEx2, BEy2, 'black')
            ax.fill(BEx1, BEy1, 'grey')
            ax.fill(WEBx, WEBy, 'lightgrey')
            ax.fill(BEx2, BEy2, 'grey')

        # ax.set_xlim([-Lbe, Lbe + Lw / 2 + Lbe])
        # ax.set_ylim([-Tw * 0.5, Tw * 1.5])
        ax.set_xlim([-Lw * 0.25, Lw / 2 + (Lw * 0.25)])
        ax.set_ylim([-Tw, Tw + Tw])
        ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Width (mm)")


if __name__ == "__main__":
    app = ShearWallAnalysisApp()
    app.mainloop()
