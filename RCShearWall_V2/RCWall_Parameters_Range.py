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
    'tw': (100.0 * mm, 400.0 * mm),  # Thickness
    'tb': (None, 600.0 * mm),  # Boundary thickness
    'lw': (None, 3.0 * m),  # Wall length (min t*6)
    'ar': (1.0, 3.0),  # Aspect Ratio
    'hw': (2.0 * m, 6.0 * m),  # Wall height
    'lbe': (0.08, 0.20),  # BE length (percentage of wall length)
    'fc': (25.0 * MPa, 60.0 * MPa),  # Concrete Compressive Strength
    'fyb': (275.0 * MPa, 600.0 * MPa),  # Steel Yield Strength BE
    'fyw': (275.0 * MPa, 600.0 * MPa),  # Steel Yield Strength Web
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

'''
# Define the parameter ranges
minParameters = [
    100.0 * mm,  # (tw) Minimum thickness
    None,  # (tb) Boundary thickness
    None,  # (lw) wall length (min t*6)
    2.0 * m,  # (hw) wall height
    0.08,  # (lbe) BE length (as a percentage of wall length)
    25.0 * MPa,  # (fc) Concrete Compressive Strength
    275.0 * MPa,  # (fyb) Steel Yield Strength BE
    275.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.005,  # (rhoYBE) BE long reinforcement ratio Eurocode8 (EN 1998-1) (Minimum = 0.01 for walls with axial load and 0.005 without axial load). ACI 318 (American Concrete Institute)
    0.0025,  # (rhoYWEB) WEB long reinforcement ratio (Minimum = 0.0025)
    0.0025,  # (rhoXBE) BE Transversal reinforcement ratio
    0.0025,  # (rhoXWEB) WEB Transversal reinforcement ratio
    0.000  # (loadF) axial load ratio
]

maxParameters = [
    400.0 * mm,  # (tw) Maximum thickness
    600.0 * mm,  # (tb) Boundary thickness
    3.0 * m,  # (lw) wall length (min t*6)
    6.0 * m,  # (hw) wall height
    0.20,  # (lbe) BE length (as a percentage of wall length)
    60.0 * MPa,  # (fc) Concrete Compressive Strength
    600.0 * MPa,  # (fyb) Steel Yield Strength BE
    600.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.055,  # (rhoYBE) BE long reinforcement ratio
    0.030,  # (rhoYWEB) WEB long reinforcement ratio
    0.050,  # (rhoXBE) BE Transversal reinforcement ratio
    0.030,  # (rhoXWEB) WEB Transversal reinforcement ratio
    0.300  # (loadF) axial load ratio
]

# Define the parameter ranges
minLoading = [
    6,  # num_cycles
    # 1 * mm,  # initial_displacement
    10 * mm,  # max_displacement
    1  # repetition_cycles
]

maxLoading = [
    12,  # num_cycles
    # 10 * mm,  # initial_displacement
    150 * mm,  # max_displacement
    3  # repetition_cycles
]
'''