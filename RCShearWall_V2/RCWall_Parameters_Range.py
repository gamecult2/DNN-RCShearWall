import numpy as np
from functions import *
import random
import math
import csv

# ***************************************************************************************************
#           DEFINE PARAMETER RANGES
# ***************************************************************************************************
# Define parameter ranges in a dictionary
parameter_ranges = {
    'tw': (80.0 * mm, 400.0 * mm),  # Thickness
    'tb': (None, 600.0 * mm),  # Boundary thickness
    'lw': (None, 3.0 * m),  # Wall length (min t*6)
    'ar': (0.5, 4.0),  # Aspect Ratio
    'hw': (2.0 * m, 6.0 * m),  # Wall height
    'lbe': (0.08, 0.20),  # BE length (percentage of wall length)
    'fc': (20.0 * MPa, 70.0 * MPa),  # Concrete Compressive Strength
    'fyb': (250.0 * MPa, 700.0 * MPa),  # Steel Yield Strength BE
    'fyw': (250.0 * MPa, 700.0 * MPa),  # Steel Yield Strength Web
    'fx': (250.0 * MPa, 700.0 * MPa),  # Steel Yield Strength Web
    'rouYb': (0.005, 0.055),  # BE long reinforcement ratio
    'rouYw': (0.0025, 0.030),  # WEB long reinforcement ratio
    'rouXb': (0.0025, 0.050),  # BE Transversal reinforcement ratio
    'rouXw': (0.0025, 0.030),  # WEB Transversal reinforcement ratio
    'loadF': (0.000, 0.300),  # Axial load ratio
}

loading_ranges = {
    'num_cycles': (6, 12),  # Number of cycles
    'initial_displacement': (1 * mm, 10 * mm),  # initial_displacement
    'max_displacement': (10 * mm, 150 * mm),  # Maximum displacement
    'repetition_cycles': (1, 3),  # Repetition cycles
}