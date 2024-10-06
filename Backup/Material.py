import math
import opsvis as opsv
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
from Units import *


def generate_cyclic_load(duration=6, sampling_rate=50, max_displacement=3):
    # Generate a constant time array
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Calculate the displacement slope to achieve the desired max_displacement
    displacement_slope = (max_displacement / 2) / (duration / 2)

    # Generate the cyclic load with displacement varying over time
    cyclic_load = (displacement_slope * t) * np.sin(2 * np.pi * t)

    return cyclic_load


# Define the material properties of the steel rod in MPa
Fy = 434 * MPa  # Yield strength in MPa
E = 200 * GPa  # Young's modulus in MPa
fc = 40 * MPa  # Yield strength in MPa 41.75

# Define the geometry of the steel rod in mm
L = 10000 * mm  # Length of the rod in mm
D = 360 * mm  # Diameter of the rod in mm
A = np.pi * (D / 2) ** 2  # Cross-sectional area of the rod in mm^2

# Calculate the second moment of area about the local z-axis
Iz = (np.pi * (D ** 4)) / 64

# Create an OpenSees model
ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)  # Model of 2 dimensions, 3 dof per node

# ---------------------------------------------------------------------------------------
# Define Steel uni-axial materials
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Define "ConcreteCM" uni-axial materials
# ---------------------------------------------------------------------------------------
concWeb = 4
concBE = 5

# ----- unconfined concrete for WEB
fc0 = abs(fc) * MPa  # Initial concrete strength
Ec0 = 8200.0 * (fc0 ** 0.375)  # Initial elastic modulus
fcU = -fc0 * MPa  # Unconfined concrete strength
ecU = -(fc0 ** 0.25) / 1150  # Unconfined concrete strain
EcU = Ec0  # Unconfined elastic modulus
ftU = 0.45 * (fc0 ** 0.5) * MPa  # Unconfined tensile strength
etU = 2.0 * ftU / EcU  # Unconfined tensile strain
xpU = 2.0
xnU = 2.3
rU = -1.9 + (fc0 / 5.2)  # Shape parameter
# ----- confined concrete for BE
fl1 = -1.58 * MPa  # Lower limit of confined concrete strength
fl2 = -1.87 * MPa  # Upper limit of confined concrete strength
q = fl1 / fl2
x = (fl1 + fl2) / (2.0 * fcU)
A = 6.8886 - (0.6069 + 17.275 * q) * math.exp(-4.989 * q)
B = (4.5 / (5 / A * (0.9849 - 0.6306 * math.exp(-3.8939 * q)) - 0.1)) - 5.0
k1 = A * (0.1 + 0.9 / (1 + B * x))
# Check the strength of transverse reinforcement and set k2 accordingly
if abs(Fy) <= 413.8 * MPa:  # Normal strength transverse reinforcement (<60ksi)
    k2 = 5.0 * k1
else:  # High strength transverse reinforcement (>60ksi)
    k2 = 3.0 * k1
# Confined concrete properties
fcC = fcU * (1 + k1 * x)
ecC = ecU * (1 + k2 * x)  # confined concrete strain
EcC = Ec0
ftC = ftU
etC = etU
xpC = xpU
xnC = 30.0
ne = EcC * ecC / fcC
rC = ne / (ne - 1)

ru = 7.0  # shape parameter - compression
xcrnu = 1.035  # cracking strain - compression
rc = 7.3049  # shape parameter - compression
xcrnc = 1.0125  # cracking strain - compression
et = 0.00008  # strain at peak tensile stress (0.00008)
rt = 1.2  # shape parameter - tension
xcrp = 10000  # cracking strain - tension

# -------------------------- ConcreteCM model --------------------------
ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
# -------------------------- Concrete7 model --------------------------------------------
# ops.uniaxialMaterial('Concrete07', concWeb, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)  # Web (unconfined concrete)
# print('Concrete07', concWeb, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)  # Web (unconfined concrete)

# ---------------------------------------------------------------------------------------
# Define "SteelMPF" uni-axial materials
# ---------------------------------------------------------------------------------------
sY = 1
sYb = 11

# STEEL Y BE (boundary element)
fyYbp = Fy  # fy - tension
fyYbn = Fy  # fy - compression
bybp = 0.0185  # strain hardening - tension
bybn = 0.02  # strain hardening - compression
R0 = 20  # initial value of curvature parameter
Bs = 0.01  # strain-hardening ratio
cR1 = 0.925  # control the transition from elastic to plastic branches
cR2 = 0.0015  # control the transition from elastic to plastic branches

# SteelMPF model
ops.uniaxialMaterial('SteelMPF', sY, fyYbp, fyYbn, E, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
# ---------------------------------------------------------------------------------------

# Create a node to represent the fixed end of the rod
ops.node(1, 0, 0)
ops.node(2, 0, L)
# Fix the fixed end of the rod in all directions
ops.fix(1, 1, 1, 1)

# Create a recorder for element stress and strain
ops.recorder('Element', '-file', 'element_output.out', '-ele', 1, 'section', str(1), 'fiber', str(D / 2), str(D / 2), 'stressStrain')

# Create a uniaxial material using a section tag
section_tag = 1
ops.section('Fiber', section_tag)
ops.patch('circ', concWeb, 36, 12, *[0, 0], *[0, D], *[0, 360])

# fib_sec_1 = [['section', 'Fiber', section_tag],
#              ['patch', 'circ', concWeb, 36, 12, *[0, 0], *[0, D], *[0, 360]]  # noqa: E501
#             ]
#
# matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
# opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
# plt.axis('equal')
# plt.show()

integrationTag = 1
ops.beamIntegration('Lobatto', integrationTag, section_tag, 5)

transformation_tag = 1
ops.geomTransf('Linear', transformation_tag)  #  Corotational
# ops.element("nonlinearBeamColumn", 1, *[1, 2], 5, section_tag, transformation_tag)
ops.element('forceBeamColumn', 1, *[1, 2], transformation_tag, integrationTag)
# ops.element('zeroLength', 1, *[1, 2], '-mat', sY, '-dir', 1, 2, 3)

# Define load pattern (applying tension)
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
ops.load(2, *[0.0, 1.0, 0.0])
ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
ops.numberer('RCM')
ops.system("BandGen")
ops.test('NormDispIncr', 1e-10, 1000, 0)
ops.algorithm('Newton')

# Define analysis parameters
DisplacementStep = generate_cyclic_load(duration=8, sampling_rate=20, max_displacement=100)

maxUnconvergedSteps = 1
unconvergeSteps = 0
Nsteps = len(DisplacementStep)
print(Nsteps)
finishedSteps = 0
dispData = np.zeros(Nsteps + 1)
baseShearData = np.zeros(Nsteps + 1)

# Perform cyclic analysis
D0 = 0.0
for j in range(Nsteps):
    D1 = DisplacementStep[j]
    Dincr = D1 - D0

    print(f'Step {j} -------->', f'Dincr = ', Dincr)
    if unconvergeSteps > maxUnconvergedSteps:
        break
    ops.integrator("DisplacementControl", 2, 2, Dincr)
    ops.analysis('Static')
    ok = ops.analyze(1)
    if ok != 0:
        # ------------------------ If not converged, reduce the increment -------------------------
        unconvergeSteps += 1
        # Dts = 10  # Analysis loop with 10x smaller increments
        # smallDincr = Dincr / Dts
        # for k in range(1, Dts):
        #     print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
        #     ops.integrator("DisplacementControl", 2, 2, smallDincr)
        #     ok = ops.analyze(1)
        # ------------------------ If not converged --------------------------------------------
        if ok != 0:
            print("Problem running Cyclic analysis for the model : Ending analysis ")
    D0 = D1  # move to next step
    finishedSteps = j + 1
    disp = ops.nodeDisp(2, 2)
    axial_force = ops.getLoadFactor(1) / 1000  # Convert to from N to kN
    dispData[j + 1] = disp
    baseShearData[j + 1] = axial_force

    print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
    print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')
    print('CYCLIC ANALYSIS DONE')

# Extract recorded data, specifying columns
# data = np.loadtxt('element_output.out')

# Extract time, element stress, and element strain
# element_stress = data[:, 0]  # 2nd column as stress
# element_strain = data[:, 1]  # 3rd column as strain

# Plot Force vs. Displacement
plt.figure(figsize=(7, 6), dpi=100)
# plt.plot(element_stress, element_strain, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
plt.plot(dispData, baseShearData, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
# plt.plot(element_strain, element_stress, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
plt.axhline(0, color='black', linewidth=0.4)
plt.axvline(0, color='black', linewidth=0.4)
plt.grid(linestyle='dotted')
font_settings = {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14}
plt.xlabel('Displacement (mm)', fontdict=font_settings)
plt.ylabel('Base Shear (kN)', fontdict=font_settings)
plt.yticks(fontname='Cambria', fontsize=14)
plt.xticks(fontname='Cambria', fontsize=14)
plt.title(f'Specimen', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
plt.tight_layout()
plt.legend()
plt.show()
