import numpy as np
from utils import *
import random
import math
import csv

# ***************************************************************************************************
#           DEFINE PARAMETER RANGES
# ***************************************************************************************************
# Define the parameter ranges
minParameters = [
    100.0 * mm,     # (tw) Minimum thickness
    None,           # (tb) Boundary thickness
    2.0 * m,        # (hw) wall height
    None,           # (lw) wall length (min t*6)
    0.08,           # (lbe) BE length (as a percentage of wall length)
    25.0 * MPa,     # (fc) Concrete Compressive Strength
    275.0 * MPa,    # (fyb) Steel Yield Strength BE
    275.0 * MPa,    # (fyw) Steel Yield Strength Web
    0.005,          # (rhoYBE) BE long reinforcement ratio Eurocode8 (EN 1998-1) (Minimum = 0.01 for walls with axial load and 0.005 without axial load). ACI 318 (American Concrete Institute)
    0.0025,         # (rhoYWEB) WEB long reinforcement ratio (Minimum = 0.0025)
    0.000,          # (loadF) axial load ratio
    2,              # numED = 4
    2,              # numPT = 2
    50 * mm2,       # areaED = 200
    50 * mm2,       # areaPT = 200
    1700 * MPa,     # fyPT = 1860
    275 * MPa       # fyED = 291
]

maxParameters = [
    400.0 * mm,     # (tw) Maximum thickness
    600.0 * mm,     # (tb) Boundary thickness
    6.0 * m,        # (hw) wall height
    3.0 * m,        # (lw) wall length (min t*6)
    0.20,           # (lbe) BE length (as a percentage of wall length)
    60.0 * MPa,     # (fc) Concrete Compressive Strength
    600.0 * MPa,    # (fyb) Steel Yield Strength BE
    600.0 * MPa,    # (fyw) Steel Yield Strength Web
    0.055,          # (rhoYBE) BE long reinforcement ratio
    0.030,          # (rhoYWEB) WEB long reinforcement ratio
    0.300,          # (loadF) axial load ratio
    6,              # numED = 4
    6,              # numPT = 2
    200 * mm2,      # areaED = 200
    200 * mm2,      # areaPT = 200
    2500 * MPa,     # fyPT = 1860
    600.0 * MPa     # fyED = 291
]

# Define the parameter ranges
minLoading = [
    6,  # num_cycles
    # 1 * mm,  # initial_displacement
    10 * mm,  # max_displacement
    1   # repetition_cycles
]

maxLoading = [
    12,  # num_cycles
    # 10 * mm,  # initial_displacement
    150 * mm,  # max_displacement
    3  # repetition_cycles
]