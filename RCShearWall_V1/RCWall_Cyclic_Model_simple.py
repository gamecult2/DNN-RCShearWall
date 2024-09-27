import math
import time
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
# import vfo.vfo as vfo
# import opsvis as opsv
from Units import *


def analysisLoopDisp(ok, step, Dincr, ControlNode, ControlNodeDof):
    # The displacement control analysis loop.
    if ok != 0:
        print("Trying Newton with Initial Tangent ..")
        ops.integrator('DisplacementControl', ControlNode, ControlNodeDof, Dincr)
        ops.test('NormDispIncr', 1e-8, 1000, 0)
        ops.algorithm('Newton', '-initial')
        ok = ops.analyze(1)

    if ok != 0:
        print("Trying RaphsonNewton ..")
        ops.integrator('DisplacementControl', ControlNode, ControlNodeDof, Dincr)
        ops.test('NormDispIncr', 1e-8, 1000, 0)
        ops.algorithm('RaphsonNewton')
        ok = ops.analyze(1)

    ops.integrator('DisplacementControl', ControlNode, ControlNodeDof, Dincr)
    ops.test('NormDispIncr', 1e-8, 1000, 0)
    ops.algorithm('KrylovNewton')

    return ok


def reset_analysis():
    """
    Resets the analysis by setting time to 0,
    removing the recorders and wiping the analysis.
    """
    ops.setTime(0.0)  # Set the time in the Domain to zero
    ops.loadConst()  # Set the loads constant in the domain
    ops.remove('recorders')  # Remove all recorder objects.
    ops.wipeAnalysis()  # destroy all components of the Analysis object
    ops.wipe()


def plotting(x_data, y_data, x_label, y_label, title, save_fig=None, plotValidation=True):
    plt.rcParams.update({'font.size': 14, "font.family": ["Cambria", "Times New Roman"]})

    # Plot Force vs. Displacement
    plt.figure(figsize=(7, 6), dpi=100)
    plt.plot(x_data, y_data, color='red', linewidth=1.2, label='Numerical test')
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')

    font_settings = {'fontname': 'Cambria', 'size': 14}
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.title(title, fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.tight_layout()
    plt.legend()
    plt.savefig('CyclicValidation/' + title, format='svg', dpi=300, bbox_inches='tight')
    plt.show()



def build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, eleH=16, eleL=8, printProgression=True):
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)  # Model of 2 dimensions, 3 dof per node

    # ----------------------------------------------------------------------------------------
    # Set geometry, ops.nodes, boundary conditions
    # ----------------------------------------------------------------------------------------
    wall_thickness = tw  # Wall thickness
    wall_height = hw  # Wall height
    wall_length = lw  # Wall width
    length_be = lbe  # Length of the Boundary Element
    length_web = lweb = lw - (2 * lbe)  # Length of the Web

    # ----------------------------------------------------------------------------------------
    # Discretization of the wall geometry
    # ----------------------------------------------------------------------------------------
    m = eleH
    n = eleL
    eleBE = 2
    eleWeb = eleL - eleBE
    elelweb = lweb / eleWeb

    # ----------------------------------------------------------------------------------------
    # Define Nodes (for MVLEM)
    # ----------------------------------------------------------------------------------------
    # Loop through the list of node values
    for i in range(1, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))
        # print(i, 0, (i - 1) * (hw / eleH))

    ops.fix(1, 1, 1, 1)  # Fixed condition at node 1

    # ---------------------------------------------------------------------------------------
    # Define Control Node and DOF                                                   ||
    # ---------------------------------------------------------------------------------------
    global ControlNode, ControlNodeDof
    ControlNode = eleH + 1  # Control Node (TopNode)
    ControlNodeDof = 1  # Control DOF 1 = X-direction

    # ---------------------------------------------------------------------------------------
    # Define Axial Load on Top Node
    # ---------------------------------------------------------------------------------------
    global Aload  # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    Aload = 0.85 * abs(fc) * tw * lw * loadCoeff
    # print('Axial load fc(kN) = ', Aload / 1000)

    # ---------------------------------------------------------------------------------------
    # Define Steel uni-axial materials
    # ---------------------------------------------------------------------------------------
    sYw = 1
    sYb = 2
    sX = 3

    # STEEL misc
    Es = 200 * GPa  # Young's modulus

    # STEEL Y BE (boundary element)
    fyYbp = fyb  # fy - tension
    fyYbn = fyb  # fy - compression
    bybp = 0.02  # strain hardening - tension
    bybn = 0.02  # strain hardening - compression

    # STEEL Y WEB
    fyYwp = fyw  # fy - tension
    fyYwn = fyw  # fy - compression
    bywp = 0.02  # strain hardening - tension
    bywn = 0.02  # strain hardening - compression

    # STEEL X
    fyXp = fyw  # fy - tension
    fyXn = fyw  # fy - compression
    bXp = 0.01  # strain hardening - tension
    bXn = 0.01  # strain hardening - compression

    # STEEL misc
    Bs = 0.01  # strain-hardening ratio
    R0 = 20.0  # initial value of curvature parameter
    cR1 = 0.925  # control the transition from elastic to plastic branches
    cR2 = 0.0015  # control the transition from elastic to plastic branches

    # SteelMPF model
    ops.uniaxialMaterial('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    ops.uniaxialMaterial('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    # ops.uniaxialMaterial('SteelMPF', sX, fyXp, fyXn, Es, bXp, bXn, R0, cR1, cR2)  # Steel X
    # print('--------------------------------------------------------------------------------------------------')
    # print('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    # print('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    # print('SteelMPF', sX, fyYwp, fyYwn, Es, bXp, bXn, R0, cR1, cR2)  # Steel X

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
    if abs(fyb) <= 413.8 * MPa:  # Normal strength transverse reinforcement (<60ksi)
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

    # -------------------------- ConcreteCM model --------------------------------------------
    # ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
    # ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 1)  # BE (confined concrete)
    # print('--------------------------------------------------------------------------------------------------')
    # print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
    # print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 1)  # BE (confined concrete)

    # -------------------------- Concrete7 model --------------------------------------------
    ops.uniaxialMaterial('Concrete07', concWeb, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)  # Web (unconfined concrete)
    ops.uniaxialMaterial('Concrete07', concBE, fcC, ecC, EcC, ftC, etC, xpC, xnC, rC)  # BE (confined concrete)
    # print('--------------------------------------------------------------------------------------------------')
    #  print('Concrete07', concWeb, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)  # Web (unconfined concrete)
    #  print('Concrete07', concBE, fcC, ecC, EcC, ftC, etC, xpC, xnC, rC)  # BE (confined concrete)

    # ----------------------------Shear spring for MVLEM-------------------------------------
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    ops.uniaxialMaterial('Elastic', 6, Kshear)

    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    # print('rouYb =', rouYb)
    # print('rouYw =', rouYw)

    # ---- Steel in X direction (BE + Web) -------------------------------------------
    # print('rouXb =', rouXb)
    # print('rouXw =', rouXw)

    # --------------------------------------------------------------------------------
    #  Define 'MVLEM' elements
    # --------------------------------------------------------------------------------
    # Set 'MVLEM' parameters thick, width, rho, matConcrete, matSteel
    MVLEM_thick = [tw] * n
    # tbe = 200 * mm
    # tweb = tw
    # MVLEM_thick = [tbe if i in (0, n - 1) else tweb for i in range(n)]
    MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
    MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
    MVLEM_matConcrete = [concBE if i in (0, n - 1) else concWeb for i in range(n)]
    MVLEM_matSteel = [sYb if i in (0, n - 1) else sYw for i in range(n)]

    for i in range(eleH):
        # ------------------ MVLEM ----------------------------------------------
        ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 6)
        # print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 6)

    parameter_values = [tw, hw, lw, lbe, fc, fyb, fyw, round(rouYb, 4), round(rouYw, 4), loadCoeff]

    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print("\033[92mModel Built Successfully --> Using the following parameters :", parameter_values, "\033[0m")
        print('--------------------------------------------------------------------------------------------------')


def run_gravity(steps=10, printProgression=True):
    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")

    ops.timeSeries('Linear', 1, '-factor', 1.0)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 1, 1)
    ops.load(ControlNode, *[0.0, -Aload, 0.0])  # apply vertical load
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1 / steps)
    ops.analysis('Static')
    ops.analyze(steps)
    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time for further analysis
    if printProgression:
        print("GRAVITY ANALYSIS DONE!")


def run_cyclic(DisplacementStep, plotResults=True, printProgression=True, recordData=False):
    if printProgression:
        tic = time.time()
        print("RUNNING CYCLIC ANALYSIS")

    if recordData:
        print("RECORDING SHEAR LOAD VS DISPLACEMENT DATA")
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Reaction.out', '-closeOnWrite', '-node', 1, '-dof', ControlNodeDof, 'reaction')
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Displacement.out', '-closeOnWrite', '-node', ControlNode, '-dof', ControlNodeDof, 'disp')

    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(ControlNode, *[1.0, 0.0, 0.0])  # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('RCM')
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-8, 1000, 0)
    ops.algorithm('KrylovNewton')
    ops.analysis('Static')

    # Define analysis parameters
    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    dispData = np.zeros(Nsteps + 1)
    baseShearData = np.zeros(Nsteps + 1)

    # Perform cyclic analysis
    D0 = 0.0
    for j in range(Nsteps):
        D1 = DisplacementStep[j]
        Dincr = D1 - D0
        if printProgression:
            print(f'Step {j} -------->', f'Dincr = ', Dincr)
        # ------------------------- first analyze command ---------------------------------------------
        ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, Dincr)
        ok = ops.analyze(1)
        # ------------------------ If not converged -------------------------
        # if ok != 0:
        #     ok = analysisLoopDisp(ok, j, Dincr, controlNode, controlNodeDof)
        if ok != 0:
            print("Problem running Cyclic analysis for the model : Ending analysis ")
            break
        else:
            D0 = D1  # move to next step

        finishedSteps = j + 1
        disp = ops.nodeDisp(ControlNode, ControlNodeDof)
        baseShear = -ops.getLoadFactor(2) / 1000  # Convert to from N to kN
        dispData[j + 1] = disp
        baseShearData[j + 1] = baseShear

        if printProgression:
            print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
            print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')

    if printProgression:
        toc = time.time()
        print('CYCLIC ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    if plotResults:
        # force_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Reaction.out')
        # displacement_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Displacement.out')
        # Plot Force vs. Displacement
        plt.figure(figsize=(7, 6), dpi=100)
        # plt.plot(DisplacementStep, -baseShearData, color='blue', linewidth=1.2, label='Numerical test')
        plt.plot(dispData, -baseShearData, color="red", linestyle="-", linewidth=2.2, label='Output Displacement vs Shear Load')
        # plt.plot([row[0] for row in dataCyc[:finishedSteps]], [-row[1] for row in dataCyc[:finishedSteps]], color="black", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
        plt.plot(DisplacementStep[0:finishedSteps], -baseShearData[1:finishedSteps + 1], color="blue", linestyle="-", linewidth=1.2, label='Input Displacement vs Shear Load')
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

    return [dispData[0:finishedSteps], -baseShearData[0:finishedSteps]]


def run_pushover(MaxDisp=75, dispIncr=1, plotResults=True, printProgression=True, recordData=False):
    if printProgression:
        tic = time.time()
        print('--------------------------------------------------------------------------------------------------')
        print("\033[92m RUNNING PUSHOVER ANALYSIS --> Using the following parameters :", MaxDisp, " and ", dispIncr, "\033[0m")
        print('--------------------------------------------------------------------------------------------------')
    if recordData:
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Reaction.out', '-node', 1, '-dof', ControlNodeDof, 'reaction')
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Displacement.out', '-node', ControlNode, '-dof', ControlNodeDof, 'disp')

    ops.timeSeries('Linear', 3)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 3, 3)
    ops.load(ControlNode, *[1.0, 0.0, 0.0])  # Apply a unit reference load in DOF=1 (nd    FX  FY  MZ)

    NstepsPush = round(MaxDisp / dispIncr)

    if printProgression:
        print("Starting pushover analysis...")
        print("   total steps: ", NstepsPush)
    ops.constraints('Transformation')
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-8, 1000)
    ops.algorithm('KrylovNewton')
    ops.analysis("Static")

    finishedSteps = 0
    dataPush = np.zeros((NstepsPush + 1, 2))
    dispImpo = np.zeros(NstepsPush + 1)

    # Perform pushover analysis
    for j in range(NstepsPush):
        ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, dispIncr)  # Target node is controlNode and dof is 1
        ok = ops.analyze(1)
        # ------------------------ If not converged -------------------------
        # if ok != 0:
        #     ok = analysisLoopDisp(ok, j, dispIncr, controlNode, controlNodeDof)
        if ok != 0:
            print("Problem running Pushover analysis for the model : Ending analysis ")
            break
        else:
            dispImpo += dispIncr
        finishedSteps = j + 1
        disp = ops.nodeDisp(ControlNode, ControlNodeDof)
        baseShear = -ops.getLoadFactor(3) / 1000  # Convert to from N to kN
        dataPush[j + 1, 0] = disp
        dataPush[j + 1, 1] = baseShear

        if printProgression:
            print("step", j + 1, "/", NstepsPush, "   ", "Impos disp = ", round(dispImpo[j], 2), "---->  Real disp = ", str(round(disp, 2)), "---->  dispIncr = ", dispIncr)

    if printProgression:
        toc = time.time()
        print('PUSHOVER ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    if plotResults:
        plt.rcParams.update({'font.size': 14, "font.family": ["Times New Roman", "Cambria"]})
        plt.figure(figsize=(6, 4), dpi=100)
        plt.plot(dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1], color="red", linewidth=1.2, linestyle="-", label='Pushover Analysis')
        plt.axhline(0, color='black', linewidth=0.4)
        plt.axvline(0, color='black', linewidth=0.4)
        plt.grid(linestyle='dotted')
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Base Shear (N)')
        plt.tight_layout()
        plt.legend()
        plt.show()

    return [dataPush[0:finishedSteps, 0], -dataPush[0:finishedSteps, 1]]
