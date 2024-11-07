import math
import opsvis as opsv
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
from Units import *
from functions import *


def generate_cyclic_load(duration=8, sampling_rate=100, max_displacement=3):
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
# ops.timeSeries("Linear", 1)
# ops.pattern("Plain", 1, 1)
# ops.load(2, *[0.0, 1.0, 0.0])
# ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
# ops.numberer('RCM')
# ops.system("BandGen")
# ops.test('NormDispIncr', 1e-10, 1000, 0)
# ops.algorithm('Newton')

# Define analysis parameters
DisplacementStep = generate_cyclic_load(duration=8, sampling_rate=50, max_displacement=120)


# define parameters for adaptive time-step
max_factor = 1  # 1.0 -> don't make it larger than initial time step
min_factor = 1e-06  # at most initial/1e6
max_factor_increment = 1.5  # define how fast the factor can increase
min_factor_increment = 1e-08  # define how fast the factor can decrease
max_iter = 5000
desired_iter = int(max_iter / 2)  # should be higher than the desired number of iterations

# -------------CYCLIC-----------------
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
RefLoad = 1000e3
ops.load(2, *[0.0, RefLoad, 0.0])
ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
ops.numberer('RCM')
ops.system("ProfileSPD")  # UmfPack 19
ops.test('NormDispIncr', 1e-6, desired_iter, 0)
ops.algorithm('KrylovNewton')  # KrylovNewton
ops.analysis("Static")

Nsteps = len(DisplacementStep)
dispData = np.zeros(Nsteps + 1)
loadData = np.zeros(Nsteps + 1)

finishedSteps = 0
D0 = 0.0
for j in range(Nsteps):
    D1 = DisplacementStep[j]
    Dincr = D1 - D0
    # start with 1 step per Dincr
    n_sub_steps = 1
    # compute the actual displacement increment
    dU = Dincr / n_sub_steps
    dU_tolerance = abs(dU) * 1.0e-8
    factor = 1.0
    old_factor = factor
    dU_cumulative = 0.0
    increment_done = False
    while True:
        # Check if the target displacement has been reached
        # are we done with this cycle?
        if abs(dU_cumulative - Dincr) <= dU_tolerance:
            print("Target displacement has been reached. Current Dincr = {:.3g}".format(dU_cumulative))
            increment_done = True
            break
        # adapt the current displacement increment
        dU_adapt = dU * factor
        if abs(dU_cumulative + dU_adapt) > (abs(Dincr) - dU_tolerance):
            dU_adapt = Dincr - dU_cumulative
        # update integrator
        ops.integrator("DisplacementControl", 2, 2, dU_adapt)
        ok = ops.analyze(1)
        # timeV[j] = ops.getTime()
        # for el_i, ele_tag in enumerate(el_tags):
        #     nd1, nd2 = ops.eleNodes(ele_tag)
        #     Eds[j, el_i, :] = [ops.nodeDisp(nd1)[0],
        #                        ops.nodeDisp(nd1)[1],
        #                        ops.nodeDisp(nd1)[2],
        #                        ops.nodeDisp(nd2)[0],
        #                        ops.nodeDisp(nd2)[1],
        #                        ops.nodeDisp(nd2)[2]]
        # adapt if necessary
        if ok == 0: # Convergence achieved
            num_iter = ops.testIter()
            norms = ops.testNorms()
            error_norm = norms[num_iter - 1] if num_iter > 0 else 0.0
            print(f"Increment: {j:6d} | Iterations: {num_iter:4d} | Norm: {error_norm:8.3e} | Progress: {j / Nsteps * 100.0:7.3f} %")

            # update adaptive factor (increase)
            factor_increment = min(max_factor_increment, desired_iter / num_iter)
            factor *= factor_increment
            if factor > max_factor:
                factor = max_factor
            if factor > old_factor:
                print("Increasing increment factor due to faster convergence. Factor = {:.3g}".format(factor))
            old_factor = factor
            dU_cumulative += dU_adapt
        else: # Convergence failed, reduce factor
            num_iter = max_iter
            factor_increment = max(min_factor_increment, desired_iter / num_iter)
            factor *= factor_increment
            print("Reducing increment factor due to non convergence. Factor = {:.3g}".format(factor))
            if factor < min_factor:
                print("ERROR: current factor is less then the minimum allowed ({:.3g} < {:.3g})".format(factor, min_factor))
                print("ERROR: the analysis did not converge")
                break
    if not increment_done:
        break
    else:
        D0 = D1  # move to next step

    # Record results
    finishedSteps = j + 1
    disp = ops.nodeDisp(2, 1)
    baseLoad = ops.getLoadFactor(1) / 1000 * RefLoad  # patternTag(2) Convert to from N to kN
    # eleForce = ops.eleForce(2, 1) / 1000
    dispData[j + 1] = disp
    loadData[j + 1] = baseLoad

# Plot Force vs. Displacement
plt.figure(figsize=(7, 6), dpi=100)
# plt.plot(element_stress, element_strain, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
plt.plot(dispData, loadData, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
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


'''
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

'''


