import math
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Arrow
from matplotlib import gridspec



@dataclass
class CyclicParams:
    n: int = 6  # number of cycles
    r: int = 2  # repetition cycles
    D0: float = 5.0  # initial displacement
    Dm: float = 80.0  # max displacement


@dataclass
class WallParams:
    Tw: float = 102.0  # Wall thickness
    Tb: float = 102.0  # Boundary element thickness
    Hw: float = 3810.0  # Wall height
    Lw: float = 1220.0  # Wall length
    Lbe: float = 190.0  # Boundary element length
    fc: float = 41.75  # Concrete strength
    fyb: float = 434.0  # Boundary reinforcement yield strength
    fyw: float = 448.0  # Web reinforcement yield strength
    rouYb: float = 2.94  # Boundary longitudinal reinforcement ratio
    rouYw: float = 0.3  # Web longitudinal reinforcement ratio
    rouXb: float = 2.94  # Boundary transverse reinforcement ratio
    rouXw: float = 0.3  # Web transverse reinforcement ratio
    loadcoef: float = 0.092  # Load coefficient


class ShearWallAnalysisApp(tk.Tk):
    PARAM_RANGES = {
        "Tw": (90, 400),
        "Tb": (90, 400),
        "Hw": (1000, 6000),
        "Lw": (540, 4000),
        "Lbe": (54, 500),
        "fc": (20, 70),
        "fyb": (275, 650),
        "fyw": (275, 650),
        "rouYb": (0.5, 5.5),
        "rouYw": (0.2, 3.0),
        "rouXb": (0.5, 5.5),
        "rouXw": (0.2, 3.0),
        "loadcoef": (0.01, 0.1)
    }

    CYCLIC_RANGES = {
        "n": (1, 20),
        "r": (1, 10),
        "D0": (0, 10),
        "Dm": (10, 160)
    }

    def __init__(self):
        super().__init__()
        self.setup_window()
        self.setup_data()
        self.setup_variables()
        self.create_gui()

    def setup_window(self):
        """Initialize window properties"""
        self.title("RC Shear Wall Analysis with DNN")
        self.geometry("1200x800")

    def setup_data(self):
        """Initialize data-related attributes"""
        self.data_folder = Path("Processed_Data/Data_19K")
        self.model_info = 'LSTM-AE'
        self.setup_scalers()

    def setup_scalers(self):
        """Initialize scaler paths"""
        self.scalers = {
            'param': self.data_folder / 'Scaler/param_scaler.joblib',
            'disp_cyclic': self.data_folder / 'Scaler/disp_cyclic_scaler.joblib',
            'shear_cyclic': self.data_folder / 'Scaler/shear_cyclic_scaler.joblib'
        }

    def setup_variables(self):
        """Initialize tkinter variables"""
        self.load_type = tk.StringVar(value="RC")
        self.protocol_type = tk.StringVar(value="normal")
        self.wall_params = WallParams()
        self.cyclic_params = CyclicParams()
        self.tk_vars = self._create_tk_variables()

    def _create_tk_variables(self) -> Dict[str, tk.Variable]:
        """Create tkinter variables for all parameters"""
        vars_dict = {}

        # Wall parameters
        for field, value in vars(self.wall_params).items():
            vars_dict[field] = tk.DoubleVar(value=value)

        # Cyclic parameters
        for field, value in vars(self.cyclic_params).items():
            var_type = tk.IntVar if isinstance(value, int) else tk.DoubleVar
            vars_dict[field] = var_type(value=value)

        return vars_dict

    def create_gui(self):
        """Create the main GUI layout"""
        left_frame = ttk.Frame(self)
        right_frame = ttk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_input_section(left_frame)
        self.create_plot_section(right_frame)

    def create_input_section(self, parent):
        """Create the input parameters section"""
        self.create_wall_params_frame(parent)
        self.create_cyclic_params_frame(parent)
        ttk.Button(parent, text="Generate", command=self.update_plots).pack(pady=10)

    def create_wall_params_frame(self, parent):
        """Create frame for wall parameters"""
        frame = ttk.LabelFrame(parent, text="RC Shear Wall Design Parameters")
        frame.pack(fill=tk.X, pady=10)

        self.create_analysis_switch(frame)
        self.create_parameter_inputs(frame, self.wall_params, self.PARAM_RANGES)

    def create_cyclic_params_frame(self, parent):
        """Create frame for cyclic parameters"""
        frame = ttk.LabelFrame(parent, text="Cyclic Loading Parameters")
        frame.pack(fill=tk.X, pady=10)

        self.create_protocol_switch(frame)
        self.create_parameter_inputs(frame, self.cyclic_params, self.CYCLIC_RANGES)

    def create_analysis_switch(self, parent):
        """Create analysis type radio buttons"""
        frame = ttk.Frame(parent)
        frame.pack(pady=5)
        ttk.Label(frame, text="Analysis: ").pack(side=tk.LEFT)
        ttk.Radiobutton(frame, text="Monotonic", variable=self.load_type, value="RC").pack(side=tk.LEFT)
        ttk.Radiobutton(frame, text="Cyclic", variable=self.load_type, value="SC").pack(side=tk.LEFT)

    def create_protocol_switch(self, parent):
        """Create protocol type radio buttons"""
        frame = ttk.Frame(parent)
        frame.pack(pady=5)
        ttk.Label(frame, text="Protocol: ").pack(side=tk.LEFT)
        ttk.Radiobutton(frame, text="Normal", variable=self.protocol_type, value="normal").pack(side=tk.LEFT)
        ttk.Radiobutton(frame, text="Exponential", variable=self.protocol_type, value="exponential").pack(side=tk.LEFT)

    def create_parameter_inputs(self, parent, params_obj, ranges_dict):
        """Create parameter input rows with sliders and entry fields"""
        for field, value in vars(params_obj).items():
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=f"{field}", width=15, anchor=tk.W).pack(side=tk.LEFT)

            range_min, range_max = ranges_dict[field]
            resolution = 0.1 if 'rou' in field else (0.001 if 'load' in field else 1)

            scale = ttk.Scale(
                frame,
                from_=range_min,
                to=range_max,
                variable=self.tk_vars[field],
                orient=tk.HORIZONTAL,
                length=200
            )
            scale.pack(side=tk.LEFT)

            ttk.Entry(frame, textvariable=self.tk_vars[field], width=8).pack(side=tk.LEFT, padx=5)

    def create_plot_section(self, parent):
        """Create the plotting section"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        self.fig.tight_layout(pad=5.0)

        titles = [
            "RC Shear Wall Elevation",
            "Results Output",
            "RC Shear Wall Section",
            "Cyclic Loading Protocol"
        ]

        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_cyclic_loading(self):
        """Generate cyclic loading displacement data"""
        n = self.tk_vars['n'].get()
        r = self.tk_vars['r'].get()
        D0 = self.tk_vars['D0'].get()
        Dm = self.tk_vars['Dm'].get()

        num_points = math.ceil(500 / (n * r))
        time = np.linspace(0, n * r, num_points * n * r)[:500]
        displacement = np.zeros_like(time)

        if self.protocol_type.get() == 'normal':
            self._generate_normal_protocol(displacement, n, r, D0, Dm, num_points)
        else:
            self._generate_exponential_protocol(displacement, n, r, D0, Dm, num_points)

        return displacement

    def generate_cyclic_loading(self):
        num_cycles = self.tk_vars["n"].get()
        repetition_cycles = self.tk_vars["r"].get()
        initial_displacement = self.tk_vars["D0"].get()
        max_displacement = self.tk_vars["Dm"].get()
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



    def create_plot_section(self, parent):
        """Create the plotting section with enhanced layout"""
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])
        gs.update(wspace=0.3, hspace=0.3)

        self.axes = {
            'elevation': self.fig.add_subplot(gs[0, 0]),
            'hysteresis': self.fig.add_subplot(gs[0, 1]),
            'section': self.fig.add_subplot(gs[1, 0]),
            'cyclic': self.fig.add_subplot(gs[1, 1])
        }

        titles = {
            'elevation': "RC Shear Wall Elevation",
            'hysteresis': "Hysteresis Loop",
            'section': "RC Shear Wall Section",
            'cyclic': "Cyclic Loading Protocol"
        }

        for key, ax in self.axes.items():
            ax.set_title(titles[key], pad=10, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_hysteresis_loop(self, ax: Axes) -> None:
        """Plot hysteresis loop with enhanced visualization"""
        try:
            # Generate displacement data
            displacement = self.generate_cyclic_loading()
            displacement_input = displacement.reshape(1, -1)[:, 1:501]

            # Get normalized parameters
            parameters = self._get_normalized_parameters()

            # Predict using model (assuming self.loaded_model exists)
            predicted_shear = self._predict_shear(parameters, displacement_input)

            # Plot the results
            ax.plot(displacement_input[-1, 5:499], predicted_shear[-1, 5:499],
                    label="DNN Results", linewidth=1.5, color='blue')

            # Load and plot reference data
            self._plot_reference_data(ax)

            # Enhance plot appearance
            ax.set_xlabel("Displacement (mm)", fontsize=9)
            ax.set_ylabel("Base Shear (kN)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)

        except Exception as e:
            print(f"Error in hysteresis plotting: {e}")
            ax.text(0.5, 0.5, "Error in plotting hysteresis loop",
                    ha='center', va='center', color='red')

    def _plot_reference_data(self, ax: Axes) -> None:
        """Plot reference data for comparison"""
        try:
            reference_data = np.loadtxt("../DataValidation/Thomsen_and_Wallace_RW2.txt",
                                        delimiter="\t", unpack=False)
            ax.plot(reference_data[0, :], reference_data[1, :],
                    color="black", linewidth=1.0, linestyle="--",
                    label='Reference Data')
        except Exception as e:
            print(f"Error loading reference data: {e}")

    def plot_cyclic_loading(self, ax: Axes) -> None:
        """Plot cyclic loading protocol with enhanced visualization"""
        try:
            displacement = self.generate_cyclic_loading()
            time = np.arange(len(displacement))

            # Plot displacement time history
            ax.plot(time, displacement, color='red', linewidth=1.5, label='Loading Protocol')

            # Add envelope curves
            envelope = np.abs(displacement)
            # ax.plot(time, envelope, 'k--', alpha=0.3, linewidth=0.5)
            # ax.plot(time, -envelope, 'k--', alpha=0.3, linewidth=0.5)

            # Enhance plot appearance
            ax.set_xlabel("Time Step", fontsize=9)
            ax.set_ylabel("Displacement (mm)", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=8)

            # Add protocol type annotation
            protocol_type = self.protocol_type.get().capitalize()
            ax.text(0.02, 0.98, f"Protocol: {protocol_type}",
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top')

        except Exception as e:
            print(f"Error in cyclic loading plotting: {e}")
            ax.text(0.5, 0.5, "Error in plotting cyclic loading",
                    ha='center', va='center', color='red')

    def plot_shear_wall_elevation(self, ax: Axes) -> None:
        """Plot shear wall elevation with enhanced visualization"""
        try:
            # Get current parameters
            Lw = self.tk_vars['Lw'].get()
            Hw = self.tk_vars['Hw'].get()
            Lbe = self.tk_vars['Lbe'].get()
            Tw = self.tk_vars['Tw'].get()
            fc = self.tk_vars['fc'].get()
            loadcoef = self.tk_vars['loadcoef'].get()

            # Calculate derived dimensions
            Lweb = Lw - (2 * Lbe)
            Aload = (0.85 * abs(fc) * Tw * Lw * loadcoef) / 1000

            # Create wall components
            wall_components = self._create_wall_elevation_components(Lw, Hw, Lbe, Lweb)

            # Plot wall components
            self._plot_wall_components(ax, wall_components)

            # Add loads and annotations
            self._add_elevation_loads_annotations(ax, Lw, Hw, Aload)

            # Set plot limits and labels
            self._setup_elevation_plot(ax, Lw, Hw)

        except Exception as e:
            print(f"Error in elevation plotting: {e}")
            ax.text(0.5, 0.5, "Error in plotting elevation",
                    ha='center', va='center', color='red')

    def _create_wall_elevation_components(self, Lw: float, Hw: float,
                                          Lbe: float, Lweb: float) -> Dict[str, dict]:
        """Create wall elevation components coordinates"""
        return {
            'left_boundary': {
                'x': [0, Lbe, Lbe, 0, 0],
                'y': [0, 0, Hw, Hw, 0],
                'fill': 'grey'
            },
            'web': {
                'x': [Lbe, Lbe + Lweb, Lbe + Lweb, Lbe, Lbe],
                'y': [0, 0, Hw, Hw, 0],
                'fill': 'lightgrey'
            },
            'right_boundary': {
                'x': [Lbe + Lweb, Lw, Lw, Lbe + Lweb, Lbe + Lweb],
                'y': [0, 0, Hw, Hw, 0],
                'fill': 'grey'
            }
        }

    def plot_shear_wall_section(self, ax: Axes) -> None:
        """Plot shear wall section with enhanced visualization"""
        try:
            # Get current parameters
            Tw = self.tk_vars['Tw'].get()
            Tb = self.tk_vars['Tb'].get()
            Lw = self.tk_vars['Lw'].get()
            Lbe = self.tk_vars['Lbe'].get()

            # Create section components
            section_components = self._create_wall_section_components(
                Tw, Tb, Lw, Lbe, half_section=True)

            # Plot section components
            self._plot_section_components(ax, section_components)

            # Add reinforcement visualization
            self._add_section_reinforcement(ax, section_components)

            # Set plot limits and labels
            self._setup_section_plot(ax, Lw, Tw)

        except Exception as e:
            print(f"Error in section plotting: {e}")
            ax.text(0.5, 0.5, "Error in plotting section",
                    ha='center', va='center', color='red')

    def _create_wall_section_components(self, Tw: float, Tb: float,
                                        Lw: float, Lbe: float,
                                        half_section: bool = True) -> Dict[str, dict]:
        """Create wall section components coordinates"""
        components = {}

        if half_section:
            components['boundary'] = {
                'x': [0, Lbe, Lbe, 0, 0],
                'y': [0, 0, Tb, Tb, 0],
                'fill': 'grey'
            }
            components['web'] = {
                'x': [Lbe, Lw / 2, Lw / 2, Lbe, Lbe],
                'y': [0, 0, Tw, Tw, 0],
                'fill': 'lightgrey'
            }
        else:
            # Add full section components if needed
            pass

        return components

    def _plot_wall_components(self, ax: Axes, components: Dict[str, dict]) -> None:
        """Plot wall components with enhanced styling"""
        for component, coords in components.items():
            ax.fill(coords['x'], coords['y'], coords['fill'])
            ax.plot(coords['x'], coords['y'], 'black', linewidth=0.5)

    def _add_elevation_loads_annotations(self, ax: Axes, Lw: float,
                                         Hw: float, Aload: float) -> None:
        """Add loads and annotations to elevation plot"""
        # Add vertical load arrow
        arrow_start = (Lw / 2, Hw + Hw * 0.22)
        arrow_length = -0.15 * Hw
        arrow = Arrow(arrow_start[0], arrow_start[1], 0, arrow_length,
                      width=Lw * 0.07, color='blue')
        ax.add_patch(arrow)

        # Add load annotation
        ax.text(Lw / 2, Hw + Hw * 0.25, f'N = {Aload:.1f} kN',
                ha='center', va='bottom', color='blue', fontsize=8)

        # Add lateral displacement arrow
        disp_arrow = Arrow(-Lw * 0.15, Hw + Hw * 0.075, Lw * 0.12, 0,
                           width=Hw * 0.07, color='red')
        ax.add_patch(disp_arrow)
        ax.text(-Lw * 0.15, Hw + Hw * 0.15, 'Dm',
                ha='center', va='bottom', color='red', fontsize=8)

    def _setup_elevation_plot(self, ax: Axes, Lw: float, Hw: float) -> None:
        """Setup elevation plot limits and labels"""
        ax.set_xlim([-Lw * 0.25, Lw + Lw * 0.25])
        ax.set_ylim([-Hw * 0.05, Hw + Hw * 0.3])
        ax.set_xlabel("Length (mm)", fontsize=9)
        ax.set_ylabel("Height (mm)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _setup_section_plot(self, ax: Axes, Lw: float, Tw: float) -> None:
        """Setup section plot limits and labels"""
        ax.set_xlim([-Lw * 0.1, Lw / 2 + Lw * 0.1])
        ax.set_ylim([-Tw * 0.2, Tw * 1.2])
        ax.set_xlabel("Length (mm)", fontsize=9)
        ax.set_ylabel("Thickness (mm)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    def update_plots(self) -> None:
        """Update all plots with error handling"""
        try:
            # Clear all axes
            for ax in self.axes.values():
                ax.clear()

            # Update individual plots
            self.plot_shear_wall_elevation(self.axes['elevation'])
            self.plot_hysteresis_loop(self.axes['hysteresis'])
            self.plot_shear_wall_section(self.axes['section'])
            self.plot_cyclic_loading(self.axes['cyclic'])

            # Adjust layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error updating plots: {e}")
            # Show error message to user if needed

if __name__ == "__main__":
    app = ShearWallAnalysisApp()
    app.mainloop()