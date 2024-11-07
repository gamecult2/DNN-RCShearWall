import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import SCWall_Model as rcmodel
from utils import *


# ------------------------------ Selected Experiment for Validation Model ------------------------------
def Nouraldaim_Case2():
    # A novel resilient system of self-centering precast reinforced concrete walls with external dampers
    global name
    name = 'Nouraldaim_Case2'
    # Wall Geometry ------------------------------------------------------------------
    tw = 125.0 * mm  # Wall thickness
    tb = tw
    hw = 4.00 * m  # Wall height
    lw = 1.35 * m  # Wall length
    lbe = 125 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 40 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 430 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 430 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fx = fyw
    
    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 4  # BE long reinforcement diameter (mm)
    YbeDiam = 10 * mm  # BE long reinforcement diameter (mm)
    YwebNum = 6  # Web long reinforcement diameter (mm)
    YwebDiam = 10 * mm  # Web long reinforcement diameter (mm)
    rouYb = (rebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (rebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293

    loadF = 0.035
    numED = 3
    numPT = 2
    areaED = rebarArea(16 * mm) * mm2
    areaPT = rebarArea(12.7 * mm) * mm2
    fyPT = 1746 * MPa
    fyED = 460 * MPa
    tenPT = 0.50 * 1746 * MPa

    # DisplacementStep = generate_cyclic_load(duration=7, sampling_rate=100, max_displacement=86)
    # DisplacementStep = generate_increasing_cyclic_loading(num_cycles=7, initial_displacement=6, max_displacement=86, num_points=50, repetition_cycles=2)
    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles=8, max_displacement=hw * 0.04, num_points=100, repetition_cycles=1)

    return tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep


def PerezTW2():
    # https://sci-hub.st/10.1061/(asce)0733-9445(2004)130:4(618)
    global name
    name = 'Perez'
    # Wall Geometry ------------------------------------------------------------------
    tw = 152 * mm  # Wall thickness
    tb = tw
    hw = 9.23 * m  # Wall height
    lw = 2.54 * m  # Wall length
    lbe = 650 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 100 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 434 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 448 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fx = fyw
    
    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 8  # BE long reinforcement diameter (mm)
    YbeDiam = 9.53  # BE long reinforcement diameter (mm)
    YwebNum = 8  # Web long reinforcement diameter (mm)
    YwebDiam = 6.35  # Web long reinforcement diameter (mm)
    rouYb = (rebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (rebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293
    rouYb = 2.47 / 100  # Y boundary        0.003
    rouYw = 1 / 100  # Y web          0.0293
    loadF = 0.05

    numED = 4
    numPT = 6
    areaED = 800
    areaPT = (48.4 * cm2) / 6

    fyPT = 952 * MPa
    fyED = 450
    tenPT = 0.60 * 1103.16 * MPa

    # DisplacementStep = generate_cyclic_load(duration=7, sampling_rate=100, max_displacement=86)
    DisplacementStep = generate_increasing_cyclic_loading(num_cycles=4, initial_displacement=216, max_displacement=180, num_points=100, repetition_cycles=1)
    # DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles=6, max_displacement=180, num_points=100, repetition_cycles=1)

    return tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep


def PerezTW3():
    # https://sci-hub.st/10.1061/(asce)0733-9445(2004)130:4(618)
    global name
    name = 'PerezTW3'
    # Wall Geometry ------------------------------------------------------------------
    tw = 152 * mm  # Wall thickness
    tb = tw
    hw = 7.23 * m  # Wall height
    lw = 2.54 * m  # Wall length
    lbe = 650 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 50 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 434 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 448 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fx = fyw
    
    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 8  # BE long reinforcement diameter (mm)
    YbeDiam = 9.53  # BE long reinforcement diameter (mm)
    YwebNum = 8  # Web long reinforcement diameter (mm)
    YwebDiam = 6.35  # Web long reinforcement diameter (mm)
    rouYb = (rebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (rebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293
    rouYb = 0.003  # Y boundary        0.003
    rouYw = 0.0293  # Y web          0.0293
    loadF = 0.05

    numED = 0
    numPT = 6
    areaED = 800
    areaPT = rebarArea(32)

    fyPT = 952 * MPa
    fyED = 450
    tenPT = 0.553 * 1103.16 * MPa

    # DisplacementStep = generate_cyclic_load(duration=7, sampling_rate=100, max_displacement=86)
    # DisplacementStep = generate_increasing_cyclic_loading(num_cycles=4, initial_displacement=216, max_displacement=hw * 0.04,, num_points=100, repetition_cycles=1)
    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles=4, max_displacement=hw * 0.02, num_points=100, repetition_cycles=1)

    return tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep


def PerezTW5():
    # https://sci-hub.st/10.1061/(asce)0733-9445(2004)130:4(618)
    global name
    name = 'PerezTW5'
    # Wall Geometry ------------------------------------------------------------------
    tw = 152 * mm  # Wall thickness
    tb = tw
    hw = 7.23 * m  # Wall height
    lw = 2.54 * m  # Wall length
    lbe = 682.625 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 70.1581 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 413.685 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 413.685 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fx = fyw
    fx = fyw
    
    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 12  # BE long reinforcement diameter (mm)
    YbeDiam = 9.53  # BE long reinforcement diameter (mm)
    YwebNum = 8  # Web long reinforcement diameter (mm)
    YwebDiam = 6.35  # Web long reinforcement diameter (mm)
    rouYb = (rebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (rebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293
    rouYb = 2.47 / 100  # Y boundary        0.003
    rouYw = 1.75 / 100  # Y web          0.0293
    loadF = 0.041

    numED = 0
    numPT = 4
    areaED = 800
    areaPT = 806.45 * mm2

    fyPT = 827.371 * MPa
    fyED = 450
    tenPT = 0.553 * 1103.16 * MPa

    # DisplacementStep = generate_cyclic_load(duration=7, sampling_rate=100, max_displacement=86)
    # DisplacementStep = generate_increasing_cyclic_loading(num_cycles=4, initial_displacement=216, max_displacement=hw * 0.04,, num_points=100, repetition_cycles=1)
    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles=8, max_displacement=hw * 0.06, num_points=50, repetition_cycles=1)

    return tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep


def Smith():
    # https://www.sciencedirect.com/science/article/pii/S2352710224006752#sec3
    global name
    name = 'Smith'
    # Wall Geometry ------------------------------------------------------------------
    tw = 159 * mm  # Wall thickness
    tb = tw
    hw = 5.48 * m  # Wall height
    lw = 2.43 * m  # Wall length
    lbe = 200 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 50 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 430 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 430 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fx = fyw
    
    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 4  # BE long reinforcement diameter (mm)
    YbeDiam = 10  # BE long reinforcement diameter (mm)
    YwebNum = 6  # Web long reinforcement diameter (mm)
    YwebDiam = 10  # Web long reinforcement diameter (mm)
    rouYb = (rebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (rebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293
    loadF = 0.032

    numED = 4
    numPT = 2
    areaED = rebarArea(19 * mm)
    areaPT = rebarArea(13 * mm) * 3  # 03 wires for each PT

    fyPT = 1862 * MPa
    fyED = 448 * MPa
    tenPT = 0.55 * fyPT

    # DisplacementStep = generate_cyclic_load(duration=7, sampling_rate=100, max_displacement=86)
    # DisplacementStep = generate_increasing_cyclic_loading(num_cycles=10, initial_displacement=40, max_displacement=160, num_points=100, repetition_cycles=1)
    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles=6, max_displacement=hw * 0.0175, num_points=100, repetition_cycles=1)

    return tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep


def Nouraldaim_Case3():
    # https://www.sciencedirect.com/science/article/pii/S2352710224006752#sec3
    global name
    name = 'Nouraldaim_Case3'
    # Wall Geometry ------------------------------------------------------------------
    tw = 200 * mm  # Wall thickness
    tb = tw
    hw = 3.70 * m  # Wall height
    lw = 1.90 * m  # Wall length
    lbe = 190 * mm  # Boundary element length
    lweb = lw - (2 * lbe)

    # Material proprieties -----------------------------------------------------------
    fc = 30 * MPa  # Concrete peak compressive stress (+Tension, -Compression)
    fyb = 461 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fyw = 461 * MPa  # Steel tension yield strength (+Tension, -Compression)
    fx = fyw
    
    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    YbeNum = 4  # BE long reinforcement diameter (mm)
    YbeDiam = 10  # BE long reinforcement diameter (mm)
    YwebNum = 6  # Web long reinforcement diameter (mm)
    YwebDiam = 10  # Web long reinforcement diameter (mm)
    rouYb = (rebarArea(YbeDiam) * YbeNum) / (lbe * tw)  # Y boundary        0.003
    rouYw = (rebarArea(YwebDiam) * YwebNum) / (lweb * tw)  # Y web          0.0293
    loadF = 0.1

    numED = 4
    numPT = 3
    areaED = rebarArea(18 * mm) * 2
    areaPT = rebarArea(15.2 * mm)  # 03 wires for each PT

    fyPT = 1738 * MPa
    fyED = 448 * MPa
    tenPT = 0.40 * fyPT

    # DisplacementStep = generate_cyclic_load(duration=7, sampling_rate=100, max_displacement=86)
    # DisplacementStep = generate_increasing_cyclic_loading(num_cycles=10, initial_displacement=40, max_displacement=160, num_points=100, repetition_cycles=1)
    DisplacementStep = generate_increasing_cyclic_loading_with_repetition(num_cycles=10, max_displacement=hw * 0.04, num_points=50, repetition_cycles=1)

    return tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep


# ------- Select Model for Validation -----------------------------------------------------------------------------------------------
validation_model = Nouraldaim_Case2()
tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, DisplacementStep = validation_model

#  ---------------- RUN CYCLIC ANALYSIS ---------------------------------------------------------------
rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT)
rcmodel.run_gravity()
[x, y] = rcmodel.run_analysis(DisplacementStep, analysis='cyclic', printProgression=False)
rcmodel.reset_analysis()
plotting(x, y, 'Displacement (mm)', 'Base Shear (kN)', f'Cyclic {name}', save_fig=False, plotValidation=False)

# ---------------- RUN PUSHOVER ANALYSIS ---------------------------------------------------------------
# rcmodel.build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT)
# rcmodel.run_gravity()
# [x, y] = rcmodel.run_analysis(DisplacementStep, analysis='pushover', printProgression=False)
# rcmodel.reset_analysis()
# plotting(x, y, 'Displacement (mm)', 'Base Shear (kN)', f'Monotonic {name}', save_fig=False, plotValidation=False)
