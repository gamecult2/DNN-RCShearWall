import numpy as np
from Units import *
import random
import math
import csv

# ***************************************************************************************************
#           DEFINE PARAMETER RANGES
# ***************************************************************************************************
# Define the parameter ranges
minParameters = [
    100.0 * mm,  # (tw) Minimum thickness
    1.5 * m,  # (hw) wall height
    None,  # (lw) wall length (min t*6)
    0.08,  # (lbe) BE length (as a percentage of wall length)
    25.0 * MPa,  # (fc) Concrete Compressive Strength
    275.0 * MPa,  # (fyb) Steel Yield Strength BE
    275.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.005,  # (rhoBE) BE long reinforcement ratio Eurocode 8 (EN 1998-1) (Minimum = 0.01 for walls with axial load and 0.005 without axial load). ACI 318 (American Concrete Institute)
    0.003,  # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
    0.005  # (loadF) axial load ratio
]

maxParameters = [
    200.0 * mm,  # (tw) Maximum thickness
    3.5 * m,  # (hw) wall height
    2.0 * m,  # (lw) wall length (min t*6)
    0.20,  # (lbe) BE length (as a percentage of wall length)
    50.0 * MPa,  # (fc) Concrete Compressive Strength
    500.0 * MPa,  # (fyb) Steel Yield Strength BE
    500.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.050,  # (rhoBE) BE long reinforcement ratio
    0.030,  # (rhoWEB) WEB long reinforcement ratio
    0.200  # (loadF) axial load ratio
]
'''
minParameters = [
    90.0 * mm,  # (tw) Minimum thickness
    1.5 * m,  # (hw) wall height
    None,  # (lw) wall length (min t*6)
    0.10,  # (lbe) BE length (as a percentage of wall length)
    25.0 * MPa,  # (fc) Concrete Compressive Strength
    275.0 * MPa,  # (fyb) Steel Yield Strength BE
    275.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.005,  # (rhoBE) BE long reinforcement ratio Eurocode 8 (EN 1998-1) (Minimum = 0.01 for walls with axial load and 0.005 without axial load). ACI 318 (American Concrete Institute)
    0.0025,  # (rhoWEB) WEB long reinforcement ratio (Minimum = 0.0025)
    0.005  # (loadF) axial load ratio
]

maxParameters = [
    400.0 * mm,  # (tw) Maximum thickness
    6.0 * m,  # (hw) wall height
    4.0 * m,  # (lw) wall length (min t*6)
    0.25,  # (lbe) BE length (as a percentage of wall length)
    70.0 * MPa,  # (fc) Concrete Compressive Strength
    520.0 * MPa,  # (fyb) Steel Yield Strength BE
    520.0 * MPa,  # (fyw) Steel Yield Strength Web
    0.050,  # (rhoBE) BE long reinforcement ratio
    0.030,  # (rhoWEB) WEB long reinforcement ratio
    0.200  # (loadF) axial load ratio
]
'''
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
