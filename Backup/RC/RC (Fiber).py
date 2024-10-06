print("======================================================")
print("<<<<< Start RC Shear wall ground motion Analysis >>>>>")

import openseespy.postprocessing.ops_vis as opsv
import openseespy.postprocessing.Get_Rendering as opsplt
from openseespy.opensees import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import ReadRecord
from PEERGM import processNGAfile

# SET UP ---------------------------------------------------------------------------------------------
wipe()  # Clear opensees model
model('basic', '-ndm', 2, '-ndf', 3)  # Model of 2 dimensions, 3 dof per node

# ------------------- Material Properties ------------------------------------------------------------
# ----------------------- CONCRETE -------------------------------------------------------------------
IDconcCore = 1  # material ID tag -- confined core concrete
IDconcCover = 2  # material ID tag -- unconfined cover concrete
# nominal concrete compressive strength
fc = -26.8  # CONCRETE Compressive Strength, MPa   (+Tension, -Compression)
Ec = 32500  # Concrete Elastic Modulus
# confined concrete
Kfc = 1.3  # ratio of confined to unconfined concrete strength
fc1C = Kfc * fc  # CONFINED concrete (mander model), maximum stress
eps1C = 2 * fc1C / Ec  # strain at maximum stress
fc2C = 0.2 * fc1C  # ultimate stress
eps2C = 5 * eps1C  # strain at ultimate stress
# unconfined concrete
fc1U = fc  # UNCONFINED concrete (todeschini parabolic model), maximum stress
eps1U = -0.003  # strain at maximum strength of unconfined concrete
fc2U = 0.2 * fc1U  # ultimate stress
eps2U = -0.01  # strain at ultimate stress
lam = 0.1  # ratio between unloading slope at $eps2 and initial slope $Ec
# tensile-strength properties
ftC = -0.14 * fc1C  # tensile strength +tension
ftU = -0.14 * fc1U  # tensile strength +tension
Ets = ftU / 0.002  # tension softening stiffness

# ----------------------- STEEL ----------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
IDreinf = 5  # material ID tag -- reinforcement
IDreinfMinMax = 6
Fy = 360  # STEEL yield stress
Es = 200000  # modulus of steel
Bs = 0.01  # strain-hardening ratio
R0 = 18.5  # control the transition from elastic to plastic branches
cR1 = 0.925  # control the transition from elastic to plastic branches
cR2 = 0.15  # control the transition from elastic to plastic branches

# ------------------- Define materials for nonlinear columns --------------------------------------
# -------------------------------------------------------------------------------------------------
# CORE CONCRETE  (confined)
uniaxialMaterial('Concrete02', IDconcCore, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)
# COVER CONCRETE  (unconfined)
uniaxialMaterial('Concrete02', IDconcCover, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)
# ------------------------------------------------------------------------------------------------
# STEEL Reinforcing steel
uniaxialMaterial('Steel02', IDreinf, Fy, Es, Bs, R0, cR1, cR2)
uniaxialMaterial('MinMax', IDreinfMinMax, IDreinf, '-min', -0.06, '-max', 0.06)
# END MATERIAL parameters ------------------------------------------------------------------------

Unite = 10 ** 3  # ----- m to mm
# -------------------------------------------------------------------------------------------------
# ----------------------- Different GEOMETRY Coordinate------------- Unit : m ---------------------
# -------------------------------------------------------------------------------------------------
member_length = 10  # The total length of the member (SHEAR WALL OR COLUMN)
number_element = 20  # Discretized
element_length = member_length / number_element  # Number of element 10
# Section Depth and Width-----------------------
HSec = 2 * Unite
BSec = 0.3 * Unite
CoverSec = 0.05 * Unite  # Concrete Cover thickness
# Reinforcement-----------------------
RebarNum = 16  # number of longitudinal-reinforcement bars in column. (symmetric top & bottom)
RebarArea = 225  # area of longitudinal-reinforcement bars mm

# -------------------------------------------------------------------------------------------------------
# ----------------------- Define FIBER SECTION properties -----------------------------------------------
# -------------------------------------------------------------------------------------------------------
# symmetric section
#                                          Y
#                                          ^
#                                          |
#            K-------------------------------------------------------------L  -     -
#             |   o     o     o     o      o      o      o      o     o   |   |     | CoverY
#    Z <------|                            +                              |   | H   -
#             |   o     o     o     o      o      o      o      o     o   |   |
#            J-------------------------------------------------------------I  -
#             |--------------------------- B -----------------------------|
#                                          | -----------------------------|
#                                                     CoverZ
# RC section:
CoverY = HSec / 2.0  # The distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
CoverZ = BSec / 2.0  # The distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
CoreY = CoverY - CoverSec
CoreZ = CoverZ - CoverSec
nfY = int(HSec / Unite * 20)  # number of fibers for concrete in y-direction
nfZ = 1  # int(BSec/Unite*20)		# number of fibers for concrete in z-direction

for i in range(1, number_element):
    section('Fiber', i)
    # Create the concrete core fibers
    patch('rect', IDconcCore, nfY, nfZ, -CoreY, -CoreZ, CoreY, CoreZ)
    # Create the concrete cover fibers (top, bottom, left, right)
    patch('rect', IDconcCover, nfY, nfZ, -(CoreY + CoverSec), -(CoreZ + CoverSec), CoreY + CoverSec, -CoreZ)
    patch('rect', IDconcCover, nfY, nfZ, -(CoreY + CoverSec), (CoreZ + CoverSec), CoreY + CoverSec, CoreZ)
    patch('rect', IDconcCover, nfY, nfZ, -(CoreY + CoverSec), -CoreZ, -CoreY, CoreZ)
    patch('rect', IDconcCover, nfY, nfZ, (CoreY + CoverSec), -CoreZ, CoreY, CoreZ)
    # Create the reinforcing fibers (left, middle, right)
    layer('straight', IDreinf, RebarNum, RebarArea, -CoreY, -CoreZ, CoreY, -CoreZ)
    layer('straight', IDreinf, RebarNum, RebarArea, -CoreY, CoreZ, CoreY, CoreZ)

# -------------------------------------------------------------------------------------------------------
# ---- Define Shear Aggregator ---- including elastic shear component for each section ------------------
# -------------------------------------------------------------------------------------------------------
area = np.zeros(number_element)
for i in range(1, number_element):
    area[i] = HSec * BSec
    uniaxialMaterial('Elastic', i + 100, ((Ec / 2.4) * area[i]) / 2)  # CONCRETE  (Shear Model) for Section Aggregator
    section('Aggregator', i + 100, i + 100, 'Vy', '-section', i)

# -------- Plote Section ----------------------------------------------------------------------------
for i in range(1, 2):
    IDconcCover = IDconcCover
    IDconcCore = IDconcCore
    fib_sec = [['section', 'Fiber', i, '-GJ', 1.0e6],
               ['patch', 'rect', IDconcCore, nfY, nfZ, -CoreY, -CoreZ, CoreY, CoreZ],
               ['patch', 'rect', IDconcCover, nfY, nfZ, -(CoreY + CoverSec), -(CoreZ + CoverSec), CoreY + CoverSec, -(CoreZ)],
               ['patch', 'rect', IDconcCover, nfY, nfZ, -(CoreY + CoverSec), (CoreZ + CoverSec), CoreY + CoverSec, (CoreZ)],
               ['patch', 'rect', IDconcCover, nfZ, nfZ, -(CoreY + CoverSec), -CoreZ, -CoreY, CoreZ],
               ['patch', 'rect', IDconcCover, nfZ, nfZ, (CoreY + CoverSec), -CoreZ, CoreY, CoreZ],
               ['layer', 'straight', IDreinf, RebarNum, RebarArea, -CoreY, -CoreZ, CoreY, -CoreZ],
               ['layer', 'straight', IDreinf, RebarNum, RebarArea, -CoreY, CoreZ, CoreY, CoreZ]]
    # print(fib_sec)
    matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
    opsv.plot_fiber_section(fib_sec, matcolor=matcolor)
    plt.axis('equal')
    # plt.savefig('fibsec_rc.png')
    plt.show()

# -------------------------------------------------------------------------------------------------------
# ----------------------- Define NODES properties & BOUNDARY --------------------------------------------
# -------------------------------------------------------------------------------------------------------
nodeX = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
nodeY = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

Node_Coordinate = 0
for i in range(0, number_element):
    Node_Coordinate += element_length
    node(i + 1, 0, Node_Coordinate)
    print('node', i + 1, 0, Node_Coordinate)
fix(1, 1, 1, 1)  # Boundary Conditions node DX DY RZ

# -------------------------------------------------------------------------------------------------------
# ----------------------- Define ELEMENTS properties ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# define geometric transformation: performs a linear geometric transformation of beam
ColTransfTag = 1
geomTransf('Corotational', ColTransfTag)  # associate a tag to transformation

Iz = np.zeros(number_element)
for i in range(1, number_element):
    # Iz[i] = (math.pi / 4) * ((ro[i] ** 4) - (ri[i] ** 4))
    beamIntegration('Lobatto', i, i, 5)
    element('forceBeamColumn', i, *[i, i + 1], ColTransfTag, i)
    # print('forceBeamColumn', i, *[i, i + 1], ColTransfTag, i)
    # element('nonlinearBeamColumn', i, *[i, i + 1], 5, i, ColTransfTag)
    # element('elasticBeamColumn', i, *[i, i + 1], area[i], Ec * 0.9, Iz[i], ColTransfTag)
    # print('elasticBeamColumn', i, *[i, i + 1], area[i], IDconc * 0.9, Iz[i], ColTransfTag)

# -------------------------------------------------------------------------------------------------------
# ----------------------- Define MASS properties --------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
Density = 2.500 * 1e-9  # ---------- Concrete Density * 1e-9 > t/mm3
Mass_Sec = np.zeros(number_element)
Vol_Sec = np.zeros(number_element)
# ----------------------------------
# Additional Mass due to Steel Tubes
# ----------------------------------
for i in range(1, number_element):
    Vol_Sec[i] = (BSec * HSec * 1000)
    Mass_Sec[i] = Vol_Sec[i] * Density
    Mass_Sec[i] += 0  # 3454 / 30  # 3454
    mass(i + 1, Mass_Sec[i], Mass_Sec[i], Mass_Sec[i])
    print('mass', i + 1, Mass_Sec[i], Mass_Sec[i], Mass_Sec[i])
print('\n===========================================================')
print(repr(Mass_Sec), end=", ")
print('\nTotal Mass of the structure Section', sum(Mass_Sec))
# print('\nTotal Mass of the structure Node', sum(Mass_Node))

# -------------------------------------------------------------------------------------------------------
# ----------------------- SAVE ODB MODEL FOR PLOTTING ---------------------------------------------------
# -------------------------------------------------------------------------------------------------------
ModelName = 'Chimney'
LoadCaseName = 'Transient'
# opsplt.createODB(ModelName, LoadCaseName, deltaT=0.02, Nmodes=10)
# -------------------------------------------------------------------------------------------------------
# ----------------------- Define Gravity ANALYSIS -------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
timeSeries('Linear', 400)
pattern('Plain', 400, 400, )
# load(25, 0, -Total_Mass*9806.65, 0)			    # #    nd,  FX,  FY, MZ --  superstructure-weight
for i in range(1, number_element):
    load(i + 1, 0, -Mass_Sec[i] * 9806.65, 0)  # #    nd,  FX,  FY, MZ --  superstructure-weight* 9806.65
constraints('Plain')  # how it handles boundary conditions
numberer('Plain')  # renumber dof's to minimize band-width (optimization), if you want to
system('BandGeneral')  # how to store and solve the system of equations in the analysis
algorithm('Linear')  # use Linear algorithm for linear analysis
integrator('LoadControl', 0.1)  # determine the next time step for an analysis, # apply gravity in 10 steps
analysis('Static')  # define type of analysis static or transient
analyze(10)  # perform gravity analysis
loadConst('-time', 0.0)  # hold gravity constant and restart time

# -------------------------------------------------------------------------------------------------------
# ----------------------- Define GROUND Motion ----------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# Set some parameters
# ------------------------ SELECTED GM ------------------------->   0,   1,   2
both = {
    "folder": "VHGM",
    "title": "Horizontal + Vertical",
    "plot": "Horizontal + Vertical GM plot"}
vertical = {  # done
    "folder": "VGM",
    "title": "Vertical",
    "plot": "Vertical GM plot"}
horizontal = {
    "folder": "HGM",
    "title": "Horizontal",
    "plot": "Horizontal GM plot"}
# ------------------------------------------- Select the direction to be Plotted -------------------------------------------
direction = horizontal

# -------------------------------------------------------------------------------------------------------
# ----------------------- Define damping  ---------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# Perform an eigenvalue analysis

xDamp = 0.05  # 5% damping ratio
numEigen = 10  # Number of eigenvalue modes

if direction == both:
    mode_i = 0  # mode 1
    mode_j = 2  # mode 3
elif direction == horizontal:
    mode_i = 0  # mode 1
    mode_j = 1  # mode 2
elif direction == vertical:
    mode_i = 2  # mode 3
    mode_j = 5  # mode 6
else:
    pass
print("mode_i = ", mode_i)
print("mode_j = ", mode_j)
Lambda = eigen('-fullGenLapack', numEigen)  # eigenvalue modes Lambda
Omega = np.zeros(numEigen)
# -------------------- Elemental Rayleigh Damping
for i in range(0, numEigen):
    Omega[i] = math.pow(Lambda[i], 0.5)
MpropSwitch = 1
KcurrSwitch = 0
KinitSwitch = 0
KcommSwitch = 1
# ----------------------------------------
alphaM = MpropSwitch * xDamp * (2 * Omega[mode_i] * Omega[mode_j]) / (Omega[mode_i] + Omega[mode_j])  # M-prop. damping; D = alphaM*M
betaKcurr = KcurrSwitch * 2 * xDamp / (Omega[mode_i] + Omega[mode_j])  # K-proportional damping;      +beatKcurr*KCurrent
betaKinit = KinitSwitch * 2 * xDamp / (Omega[mode_i] + Omega[mode_j])  # initial-stiffness proportional damping      +beatKinit*Kini
betaKcomm = KcommSwitch * 2 * xDamp / (Omega[mode_i] + Omega[mode_j])  # last-committed K;   +betaKcomm*KlastCommitt

#       (alpha_m, beta_k, beta_k_init, beta_k_comm) # RAYLEIGH  D = αM∗M + βK∗Kcurr + βKinit∗Kinit + βKcomm∗Kcommit
rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)
# rayleigh(0.0, 0.0, 0.0, 0.000625)

# -------------------------------------------------------------------------------------------------------
# ----------------------- Plot Model and Deformation ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# plt.plot(reaction, Time)
# plt.title(filename)
# plt.grid(linestyle='--', linewidth=0.5)
# plt.ylabel('Displacement H (cm/s2)')
# plt.xlabel('Time (s)')
# plt.show()

# plt.figure()
opsplt.plot_model("nodes", "elements")
opsplt.plot_modeshape(1, 100)
opsplt.plot_modeshape(2, 100)
opsplt.plot_modeshape(3, 100)
opsplt.plot_deformedshape(Model=ModelName, LoadCase=LoadCaseName, scale=100, overlap="yes")  # tstep=100.0,
plt.show()
# opsv.plot_mode_shape(3, sfac=200000000, unDefoFlag=0, endDispFlag=0)
# ani = opsplt.animate_deformedshape(ModelName, LoadCaseName, dt=dt, scale = 100)
# plt.show()

# opsplt.plot_modeshape(2, 200, Model = ModelName)
# opsplt.plot_deformedshape(Model = ModelName, LoadCase = LoadCaseName)
# anime = opsplt.animate_deformedshape(Model = ModelName, LoadCase = LoadCaseName, dt = dt, tStart=10.0, tEnd=20.0, scale=200, Movie="Dynamic")
