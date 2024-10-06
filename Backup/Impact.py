
import math
import os
import numpy as np
import openseespy.opensees as ops
# import opsvis as opsv
# import vfo.vfo as vfo
import matplotlib.pyplot as plt
import hertz_impact_material

# ----------------------------------------------------------------
# Example : Simulation of column impact behavior using SFI_MVLEM
# Specimen: C_868kg_7.14m/s (Zhao Debo)
# Created by: Mushi Chang
# Date: 10/2022
# ----------------------------------------------------------------

ops.model('Basic', '-ndm', 2, '-ndf', 3)

# ------------------------------------------------------------------------
# node nodeId xCrd yCrd..
# ------------------------------------------------------------------------
#                              1    2   3     4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30
element_length = np.array([0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
#                              1  2    3    4    5    6    7    8    9    10   11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31
Nodes_coordinate_Y = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000])
# Impact height
Impact_coordinate_Y = 1500
HBeam = 500
BBeam = 200
ImpMass = 0.868
# Create Beam Nodes
for i in range(0, len(element_length)):
    ops.node(i + 1, 0, int(Nodes_coordinate_Y[i]))
    # print('node', i + 1, 0, int(Nodes_coordinate_Y[i]))
# Create impact node
ops.node(len(element_length) + 1, 0, Impact_coordinate_Y)
# print('node', len(element_length) + 1, 0, Impact_coordinate_Y)

# ------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------
ops.fix(1, 1, 1, 0)
ops.fix(len(element_length), 1, 0, 0)
ops.fix(len(element_length) + 1, 0, 1, 1)
# print('fix', len(element_length), 1, 0, 0)
# print('fix', len(element_length) + 1, 0, 1, 1)

# ------------------------------------------------------------------------
# Define Node Mass
# ------------------------------------------------------------------------

LBeam = Nodes_coordinate_Y[len(Nodes_coordinate_Y) - 1]
BeamMass = HBeam / 1000 * BBeam / 1000 * LBeam / 1000 * 2.4

# print(BeamMass)
for i in range(0, len(element_length)):
    ops.mass(i + 1, BeamMass / len(element_length), 0, 0)
    # print('mass', i + 1, BeamMass / len(element_length), 0, 0)
ops.mass(len(element_length) + 1, ImpMass, 0, 0)
# print('mass', len(element_length) + 1, ImpMass, 0, 0)

# ------------------------------------------------------------------------
# Define uniaxial materials for 2D RC Panel Constitutive Model (FSAM)
# ------------------------------------------------------------------------
# Steel X
fyx = 344.66    # yield strength of transverse reinforcement in Mpa
bx = 0.02   # strain hardening coefficient of transverse reinforcement
# Steel Y
fyY = 495     # yield strength of longitudinal reinforcement in MPa
by = 0.02   # strain hardening coefficient of longitudinal reinforcement
# Steel misc
Esy = 200000  # Young's modulus (199947.9615MPa)
Esx = 200000  # Young's modulus (199947.9615MPa)
R0 = 10      # Initial value of curvature parameter
A1 = 0.925   # Curvature degradation parameter
A2 = 0.15    # Curvature degradation parameter
# Build SteelMPF material
ops.uniaxialMaterial('SteelMPF', 1, fyx, fyx, Esx, bx, bx, R0, A1, A2)  #Steel X
ops.uniaxialMaterial('SteelMPF', 2, fyY, fyY, Esy, by, by, R0, A1, A2)  #Steel Y
# Concrete
# confined concrete
fpc = 26.26        # peak compressive stress
ec0 = -0.001900  # strain at peak compressive stress
ft = 3.114421     # peak tensile stress
et = 0.0002   # strain at peak tensile stress
Ec = 27576.46253    # Young's modulus
xcrnu = 1.035    # cracking strain (compression)
xcrp = 10000     # cracking strain (tension)
ru = 2.936         # shape parameter (compression)
rt = 10        # shape parameter (tension)
# Build ConcreteCM material
ops.uniaxialMaterial('ConcreteCM', 3, -fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp, '-GapClose', 0)  # unconfined concrete

# ---------------------------------------
#  Define 2D RC Panel Material (FSAM)
# ---------------------------------------

# Reinforcing ratios
rouX = 0.00094    # Reinforcing ratio of transverse rebar
rouY = 0.01659  # Reinforcing ratio of longitudinal rebar
nu = 0.2         # Friction coefficient
alfadow = 0.012  # Dowel action stiffness parameter

# Build ndMaterial FSAM
ops.nDMaterial('FSAM', 6, 0.0, 1, 2, 3, rouX, rouY, nu, alfadow)


v = 7.14
[kt1, kt2, deltay, gap] = hertz_impact_material.hertz_impact_material(0.7, 0.298 / 2.54, 2000000, v, 0.005)
print("kt1 =", kt1, "(N/mm)", "kt2 =", kt2, "(N/mm)", "deltay =", deltay, "(mm)", "gap =", gap, "(mm)")
IDimpactM = 101
ops.uniaxialMaterial('ImpactMaterial', IDimpactM, kt1, kt2, deltay, gap)

# ------------------------------
#  Define SFI_MVLEM and Zero-Length elements
# ------------------------------
for i in range(1, len(element_length)):
    ops.element('SFI_MVLEM', i, i, i+1, 5, 0.4, '-thick', BBeam, BBeam, BBeam, BBeam, BBeam, '-width', 100, 100, 100, 100, 100, '-mat', 6, 6, 6, 6, 6)
    print('SFI_MVLEM', i, i, i+1, 5, 0.4, '-thick', BBeam, BBeam, BBeam, BBeam, BBeam, '-width', 100, 100, 100, 100, 100, '-mat', 6, 6, 6, 6, 6)
    # BBeam, BBeam, BBeam, '-width', 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, '-mat', 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)

BeamImpArg = np.argwhere(Nodes_coordinate_Y == Impact_coordinate_Y)
# print(BeamImpArg)

ops.element('zeroLength', 101, int(BeamImpArg) + 1, len(element_length) + 1, '-mat', 101, '-dir', 1)
# print('element', 101, int(BeamImpArg) + 1, len(element_length) + 1, '-mat', 101, '-dir', 1)

dataDir = 'SFI_MVLEM_Fujikake_Zhao_C_868' + '_' + str('%.2f' %v) + '1111'
if not os.path.exists(dataDir):
    os.mkdir(dataDir)
os.chdir(dataDir)
# b= os.getcwd()
# print(b)
ops.recorder('Node', '-file', f'MVLEM_V_impact_point.txt', '-time', '-node', int(BeamImpArg), '-dof', 1, 'vel')
# ops.recorder('Node', '-file', f'MVLEM_A_impact_point.txt', '-time', '-node', int(BeamImpArg), '-dof', 1, 'accel')
# # print(dataDir)
# # print('recorder', 'Node', '-file', f'MVLEM_Dimpact.txt', '-time', '-node', int(BeamImpArg), '-dof', 1, 'disp')
# ops.recorder('Node', '-file', f'MVLEM_D_beam.txt', '-time', '-nodeRange', 1, len(element_length), '-dof', 1, 'disp')
# ops.recorder('Node', '-file', f'MVLEM_A_beam.txt', '-time', '-nodeRange', 1, len(element_length), '-dof', 1, 'accel')
# # Element recorders
ops.recorder('Element', '-file', f'MVLEM_F_impact.txt', '-time', '-ele', 101, 'force')
# ops.recorder('Element', '-file', f'MVLEM_Fglobal.txt', '-time', '-eleRange', 1, 10, 'globalForce')
# ops.recorder('Element', '-file', f'MVLEM_Dshear.txt', '-time', '-eleRange', 1, 10, 'ShearDef')
# ops.recorder('Element', '-file', f'MVLEM_Curvature.txt', '-time', '-eleRange', 1, 10, 'Curvature')
# Single RC panel (macro-fiber) responses
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel1_strain.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 1, 'panel_strain')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel10_strain.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 10, 'panel_strain')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel1_stress.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 1, 'panel_stress')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel10_stress.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 10, 'panel_stress')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel1_stress_concrete.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 1, 'panel_stress_concrete')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel10_stress_concrete.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 10, 'panel_stress_concrete')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel1_stress_steel.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 1, 'panel_stress_steel')
# ops.recorder('Element', '-file', f'MVLEM_impact_point_panel10_stress_steel.txt', '-time', '-ele', int(BeamImpArg), 'RCPanel', 10, 'panel_stress_steel')
# Unaxial Steel Recorders for all panels
for i in range(0, len(element_length) - 1):
    for j in range(0, 5):
        # Unaxial Steel Recorders for all panels
        # ops.recorder('Element', '-file', f'MVLEM_strain_stress_steelX_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelX')
        # ops.recorder('Element', '-file', f'MVLEM_strain_stress_steelY_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelY')
        print('recorder', 'Element', '-file', f'MVLEM_strain_stress_steelX_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelX')
        print('recorder', 'Element', '-file', f'MVLEM_strain_stress_steelY_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelY')
        # Unaxial Concrete Recorders for all panels
        # ops.recorder('Element', '-file', f'MVLEM_strain_stress_concr1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete1')
        # ops.recorder('Element', '-file', f'MVLEM_strain_stress_concr2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete2')
        print('recorder', 'Element', '-file', f'MVLEM_strain_stress_concr1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete1')
        print('recorder', 'Element', '-file', f'MVLEM_strain_stress_concr2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete2')
        # Shear Concrete Recorders for all panels
        # ops.recorder('Element', '-file', f'MVLEM_strain_stress_inter1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock1')
        # ops.recorder('Element', '-file', f'MVLEM_strain_stress_inter2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock2')
        # print('recorder', 'Element', '-file', f'MVLEM_strain_stress_inter1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock1')
        # print('recorder', 'Element', '-file', f'MVLEM_strain_stress_inter2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock2')
        ops.recorder('Element', '-file', f'MVLEM_cracking_angle_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'cracking_angles')
        print('recorder', 'Element', '-file', f'MVLEM_cracking_angle_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'cracking_angles')




# ------------------------------
# Create a Plain load pattern with a linear TimeSeries
# ------------------------------

ops.timeSeries('Rectangular', 1, 0, 0.005)
ops.pattern('Plain', 1, 1)
N = ImpMass * v * 1000 / 0.005
# print(N)
ops.load(len(element_length) + 1, -N, 0.0, 0.0)

# ------------------------------
# Analysis generation
# ------------------------------
Tol = 1.0e-1
# Create the constraint handler, the transformation method
ops.constraints('Transformation')
# Create the system of equation, a sparse solver with partial pivoting
ops.system('BandGen')
# Create the DOF numberer, the reverse Cuthill-McKee algorithm
ops.numberer('RCM')
# Create the convergence test, the norm of the residual with a tolerance of 1e-5 and a max number of iterations of 100
ops.test('NormDispIncr', Tol, 5000, 2)
# Create the solution algorithm, a Newton-Raphson algorithm
ops.algorithm('Newton', '-initial')
# Create the integration scheme, the LoadControl scheme using steps of 0.1
ops.integrator('Newmark', 0.5, 0.25)
# Create the analysis object
ops.analysis('Transient')
# Run analysis
ops.analyze(20000, 0.00001)


