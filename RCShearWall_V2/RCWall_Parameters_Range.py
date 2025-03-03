from utils.functions import *

# ***************************************************************************************************
#           DEFINE PARAMETER RANGES
# ***************************************************************************************************
# Define parameter ranges in a dictionary
parameter_ranges = {
    'tw': (80.0 * mm, 350.0 * mm),  # Thickness
    'tb': (None, 600.0 * mm),  # Boundary thickness
    'lw': (None, 3.5 * m),  # Wall length (min t*6)
    'ar': (1.0, 3.5),  # Aspect Ratio
    'hw': (2.0 * m, 6.0 * m),  # Wall height
    'lbe': (0.10, 0.25),  # BE length (percentage of wall length)
    'fc': (20.0 * MPa, 70.0 * MPa),  # Concrete Compressive Strength
    'fyb': (250.0 * MPa, 650.0 * MPa),  # Steel Yield Strength BE
    'fyw': (250.0 * MPa, 650.0 * MPa),  # Steel Yield Strength Web
    'fx': (250.0 * MPa, 650.0 * MPa),  # Steel Yield Strength Web
    'rouYb': (0.005, 0.055),  # BE long reinforcement ratio
    'rouYw': (0.0025, 0.030),  # WEB long reinforcement ratio
    'rouXb': (0.0025, 0.050),  # BE Transversal reinforcement ratio
    'rouXw': (0.0025, 0.030),  # WEB Transversal reinforcement ratio
    'loadF': (0.000, 0.300)  # Axial load ratio
}

loading_ranges = {
    'num_cycles': (6, 12),  # Number of cycles
    'initial_displacement': (1 * mm, 10 * mm),  # initial_displacement
    'max_displacement': (10 * mm, 150 * mm),  # Maximum displacement
    'repetition_cycles': (1, 4)  # Repetition cycles
}
