import math
import os
import time
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import vfo.vfo as vfo
import opsvis as opsv
from Units import *
from GenerateCyclicLoading import *

tw = 400
hw = 2600
lw = 1600
lbe = 200
fc = 40
fyb = 340
fyw = 340
rouYb = 0.002
rouYw = 0.002
rouXb = 0.002
rouXw = 0.002
loadCoeff = 0.1
numED = 4
numPT = 2
areaED = 200
areaPT = 200
fyPT = 1860
fyED = 291

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
    if save_fig:
        plt.savefig('CyclicValidation/' + title, format='svg', dpi=300, bbox_inches='tight')
    plt.show()


def build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numED, numPT, areaED, areaPT, fyPT, fyED, eleH=12, eleL=8, printProgression=True):
    ops.wipe()
    ops.model('Basic', '-ndm', 2, '-ndf', 3)

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
    for i in range(2, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))
        # print('Node', i, 0, (i - 1) * (hw / eleH))

    # ---------------------------------------------------------------------------------------
    # Define Control Node and DOF                                                   ||
    # ---------------------------------------------------------------------------------------
    global ControlNode, ControlNodeDof
    ControlNode = eleH + 1  # Control Node (TopNode)
    ControlNodeDof = 1  # Control DOF 1 = X-direction
    # print('controlNode', controlNode)

    # Wall Base ENT Interface
    numENT = 9
    # [96,  97,  98,  99,  100, 101, 102, 103, 104]
    #   |    |    |    |    |    |    |    |    |
    # [86,  87,  88,  89,  90,  91,  92,  93,  94]
    botENT = [int(86 + i) for i in range(numENT)]
    topENT = [int(86 + numENT + 1 + i) for i in range(numENT)]
    # print('botENT', botENT)
    # print('topENT', topENT)

    # Pre-calculate positions for efficiency
    ENT_positions = np.linspace(-lw / 2, lw / 2, numENT)
    # print('ENT_positions', ENT_positions)

    for i in range(numENT):
        ops.node(botENT[i], ENT_positions[i], 0)
        ops.node(topENT[i], ENT_positions[i], 0)
        # print('nodeENT', botENT[i], ENT_positions[i], 0)
        # print('nodeENT', topENT[i], ENT_positions[i], 0)

    WallNode = [topENT[(numENT - 1) // 2]]
    WallNode.extend(range(2, eleH + 2))
    # print('WallNode', WallNode)
    # WallNode = [topENT[(numENT - 1) // 2]] + list(range(2, eleH + 2))
    # print('WallNode', WallNode)

    unblED = hw / eleH
    anclED = 0
    # Calculate locED based on numED and lw
    locED = [((-lw / 3) / 2) + ((lw / 3) / (numED - 1)) * i for i in range(numED)]
    print("locED", locED)
    # Initialize lists for node IDs
    botED = []
    topED = []
    # Energy Dissipating Rebars Node
    for j in range(numED):
        node_id_bottom = 200 + j
        node_id_top = 300 + j
        ops.node(node_id_bottom, locED[j], -anclED)
        ops.node(node_id_top, locED[j], unblED)
        botED.append(node_id_bottom)
        topED.append(node_id_top)
        # print('EDnode', node_id_bottom, locED[j], -anclED)
        # print('EDnode', node_id_top, locED[j], unblED)
    print('botED', botED)
    print('topED', topED)

    anclPT = 600
    # Calculate locED based on numED and lw
    locPT = [((-lw / 4) / 2) + ((lw / 4) / (numPT - 1)) * i for i in range(numPT)]
    # print("locPT", locPT)
    botPT = []
    topPT = []
    # PT Node
    for j in range(0, numPT):
        node_id_bottom = 400 + j
        node_id_top = 500 + j
        ops.node(node_id_bottom, locPT[j], -anclPT)
        ops.node(node_id_top, locPT[j], hw)
        botPT.append(node_id_bottom)
        topPT.append(node_id_top)
        # print('PTnode', 400 + j, locPT[j], -anclPT)
        # print('PTnode', 500 + j, locPT[j], hw)
    print('botPT', botPT)
    print('topPT', topPT)

    # ------------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------------
    # Constraints of base spring bottom nodes of the Wall
    for i in range(0, len(botENT)):
        ops.fix(botENT[i], 1, 1, 1)
        # print('fix', botENT[i], 1, 1, 1)

    # Constraints of base spring top nodes of the Wall
    for i in range(0, 1):
        ops.fix(topENT[i], 1, 0, 0)
        # print('fix', topENT[i], 1, 0, 0)

    # Constraints of ED truss bottom nodes of the Wall
    for j in range(0, numED):
        ops.fix(200 + j, 1, 1, 1)
        # print('fix', 200 + j, 1, 1, 1)

    # Constraints of PT truss bottom nodes of the Wall
    for j in range(0, numPT):
        ops.fix(400 + j, 1, 1, 1)
        # print('fix', 400 + j, 1, 1, 1)

    # ---------------------------------------------------------------------------------------
    # Define Steel uni-axial materials
    # ---------------------------------------------------------------------------------------
    sYb = 1
    sYw = 2

    # STEEL misc
    Es = 200 * GPa  # Young's modulus

    # STEEL Y BE (boundary element)
    fyYbp = fyb  # fy - tension
    fyYbn = fyb  # fy - compression
    bybp = 0.01  # strain hardening - tension
    bybn = 0.01  # strain hardening - compression

    # STEEL Y WEB
    fyYwp = fyw  # fy - tension
    fyYwn = fyw  # fy - compression
    bywp = 0.02  # strain hardening - tension
    bywn = 0.02  # strain hardening - compression

    # STEEL X
    fyXp = fyw  # fy - tension
    fyXn = fyw  # fy - compression
    bXp = 0.02  # strain hardening - tension
    bXn = 0.02  # strai n hardening - compression

    # STEEL misc
    Bs = 0.01  # strain-hardening ratio
    R0 = 20.0  # initial value of curvature parameter
    cR1 = 0.925  # control the transition from elastic to plastic branches
    cR2 = 0.0015  # control the transition from elastic to plastic branches

    # Steel ED
    matED = 3
    fyED = fyED
    b = 0.01

    # Steel PT
    matPT = 4
    fyPT = fyPT
    a1 = 0
    a2 = 1
    a3 = 0
    a4 = 1
    sigInit = 750

    # SteelMPF model
    ops.uniaxialMaterial('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    ops.uniaxialMaterial('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    ops.uniaxialMaterial('Steel02', matED, fyED, Es, b, R0, cR1, cR2)  # Steel ED
    ops.uniaxialMaterial('Steel02', matPT, fyPT, Es, b, R0, cR1, cR2, a1, a2, a3, a4, sigInit)  # Steel PT

    # ---------------------------------------------------------------------------------------
    # Define "ConcreteCM" uni-axial materials
    # ---------------------------------------------------------------------------------------
    concWeb = 5
    concBE = 6

    # ----- unconfined concrete for WEB
    fc0 = abs(fc) * MPa  # Initial concrete strength
    Ec0 = 8200.0 * (fc0 ** 0.375)  # Initial elastic modulus
    fcU = -fc0 * MPa  # Unconfined concrete strength
    ecU = -(fc0 ** 0.25) / 1150  # Unconfined concrete strain
    EcU = Ec0  # Unconfined elastic modulus
    ftU = 0.35 * (fc0 ** 0.5)  # Unconfined tensile strength
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
    # Check the strength of transverse reinforcement and set k2 accordingly10.47
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

    ru = 7.3049  # shape parameter - compression
    xcrnu = 1.0125  # cracking strain - compression
    rc = 7  # shape parameter - compression
    xcrnc = 1.039  # cracking strain - compression
    et = 0.00008  # strain at peak tensile stress (0.00008)
    rt = 1.2  # shape parameter - tension
    xcrp = 10000  # cracking strain - tension

    # -------------------------- ConcreteCM model --------------------------------------------
    ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, rU, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rC, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    # print('--------------------------------------------------------------------------------------------------')
    # print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    # print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)

    # ----------------------------Shear spring for MVLEM-------------------------------------
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    matSpring = 7
    ops.uniaxialMaterial('Elastic', matSpring, Kshear)

    # Define ENT Material
    matENT = 9
    EENT = 2.5 * 32500.0 * lw * tw / (hw * numENT)
    ops.uniaxialMaterial('ENT', matENT, EENT)

    # ------------------------------
    #  Define SFI_MVLEM elements
    # ------------------------------
    MVLEM_thick = [tw] * n
    MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
    MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
    MVLEM_matConcrete = [concBE if i in (0, n - 1) else concWeb for i in range(n)]
    MVLEM_matSteel = [sYb if i in (0, n - 1) else sYw for i in range(n)]

    for i in range(eleH):
        # ------------------ MVLEM ----------------------------------------------
        ops.element('MVLEM', i + 101, 0.0, WallNode[i], WallNode[i + 1], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', matSpring)
        # print('element', 'MVLEM', i + 101, WallNode[i], WallNode[i + 1], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', matSpring)

    # Define Element for ENT (Zero-Length element)
    for i in range(0, numENT):
        ops.element('zeroLength', i + 601, botENT[i], topENT[i], '-mat', matENT, '-dir', 2)
        print('zeroLength', i + 601, botENT[i], topENT[i], '-mat', matENT, '-dir', 2)

    A = 1e5
    E = 3.0e11
    I = 9.0e4
    geomTransfTag_PDelta = 1
    ops.geomTransf('PDelta', geomTransfTag_PDelta)

    # Define Element for ENT (Rigid element between ENT top nodes)
    for i in range(0, numENT - 1):
        ops.element('elasticBeamColumn', i + 1001, topENT[i], topENT[i + 1], A, E, I, geomTransfTag_PDelta)
        # print('elasticBeamColumn', i + 1601, topENT[i], topENT[i + 1], A, E, I, geomTransfTag_PDelta)

    # Define Elements for ED (Truss element between ED top nodes)
    for i in range(numED):
        ops.element('truss', 201 + i, 200 + i, 300 + i, areaED, matED)
        print('ED-truss', 201 + i, 200 + i, 300 + i, areaED, matED)

    # Define ED elements
    topED.insert(numED // 2, WallNode[1])
    for i in range(numED):
        ops.element('elasticBeamColumn', 211 + i, topED[i], topED[i + 1], A, E, I, geomTransfTag_PDelta)
        print('ED-elasticBeamColumn', 211 + i, topED[i], topED[i + 1], A, E, I, geomTransfTag_PDelta)

    # Define PT elements
    topPT.insert(numPT // 2, WallNode[-1])
    print(topPT)
    for i in range(numPT):
        ops.element('truss', 301 + i, 400 + i, 500 + i, areaPT, matPT)
        print('PT-truss', 301 + i, 400 + i, 500 + i, areaPT, matPT)
        ops.element('elasticBeamColumn', 311 + i, topPT[i], topPT[i + 1], A, E, I, geomTransfTag_PDelta)
        print('PT-rigid', 311 + i, topPT[i], topPT[i + 1], A, E, I, geomTransfTag_PDelta)

    # ---------------------------------------------------------------------------------------
    # Define Axial Load on Top Node
    # ---------------------------------------------------------------------------------------
    global Aload  # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    Aload = 0.85 * abs(fc) * tw * lw * loadCoeff
    # print('Axial load fc(kN) = ', Aload / 1000)
    # opsv.plot_model(node_supports=False)
    opsv.plot_defo(sfac=1.5)
    plt.show()

def run_gravity(steps=10, printProgression=True):
    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(ControlNode, 0, -Aload, 0)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1 / steps)
    ops.analysis('Static')
    ops.analyze(steps)
    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time for further analysis
    if printProgression:
        print("GRAVITY ANALYSIS DONE!")


def run_pushover(MaxDisp=75, dispIncr=1, plotResults=True, printProgression=True, recordData=False):
    if printProgression:
        tic = time.time()
        print("RUNNING PUSHOVER ANALYSIS")

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

    maxUnconvergedSteps = 1
    unconvergeSteps = 0
    finishedSteps = 0
    dataPush = np.zeros((NstepsPush + 1, 2))
    dispImpo = np.zeros(NstepsPush + 1)

    # Perform pushover analysis
    for j in range(NstepsPush):
        if unconvergeSteps > maxUnconvergedSteps:
            break
        ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, dispIncr)  # Target node is controlNode and dof is 1
        ok = ops.analyze(1)
        if ok != 0:
            # ------------------------ If not converged, reduce the increment -------------------------
            unconvergeSteps += 1
            Dts = 20  # Try 50x smaller increments
            smallDincr = dispIncr / Dts
            for k in range(1, Dts):
                if printProgression:
                    print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
                ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, smallDincr)
                ok = ops.analyze(1)
            # ------------------------ If not converged --------------------------------------------
            if ok != 0:
                if printProgression:
                    print("Problem running Pushover analysis for the model : Ending analysis ")

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


def run_cyclic(DisplacementStep, plotResults=True, printProgression=True, recordData=False):
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(ControlNode, *[1.0, 0.0, 0.0])  # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('RCM')
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-8, 1000, 0)
    ops.algorithm('KrylovNewton')
    ops.analysis('Static')

    # for k in range(0, len(displacement)):
    #     disp = displacement[k]
    #     ops.integrator('DisplacementControl', controlNode, 1, 0.1)
    #     for i in range(0, disp):
    #         ops.analyze(1)
    #         # save_result(RESULTS, numENT, element_length)
    #         f = ops.nodeDisp(controlNode, 1)
    #         p = i / disp * 25
    #         print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
    #     ops.integrator('DisplacementControl', controlNode, 1, -0.1)
    #     for i in range(0, disp):
    #         ops.analyze(1)
    #         # save_result(RESULTS, numENT, element_length)
    #         f = ops.nodeDisp(controlNode, 1)
    #         p = i / disp * 25 + 25
    #         print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
    #     ops.integrator('DisplacementControl', controlNode, 1, -0.1)
    #     for i in range(0, disp):
    #         ops.analyze(1)
    #         # save_result(RESULTS, numENT, element_length)
    #         f = ops.nodeDisp(controlNode, 1)
    #         p = i / disp * 25 + 50
    #         print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
    #     ops.integrator('DisplacementControl', controlNode, 1, 0.1)
    #     for i in range(0, disp):
    #         ops.analyze(1)
    #         # save_result(RESULTS, numENT, element_length)
    #         f = ops.nodeDisp(controlNode, 1)
    #         p = i / disp * 25 + 75
    #         print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")

    print("Done Cyclic analysis")

    # Define analysis parameters
    maxUnconvergedSteps = 2
    unconvergeSteps = 0
    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    dispData = np.zeros(Nsteps + 1)
    baseShearData = np.zeros(Nsteps + 1)

    el_tags = ops.getEleTags()
    nels = len(el_tags)
    Eds = np.zeros((Nsteps, nels, 6))
    timeV = np.zeros(Nsteps)
    # Perform cyclic analysis
    D0 = 0.0
    for j in range(Nsteps):
        # collect disp for element nodes

        D1 = DisplacementStep[j]
        Dincr = D1 - D0
        if printProgression:
            print(f'Step {j} -------->', f'Dincr = ', Dincr)
        if unconvergeSteps > maxUnconvergedSteps:
            break
        # ------------------------- first analyze command ---------------------------------------------
        ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, Dincr)
        ok = ops.analyze(1)
        timeV[j] = ops.getTime()
        for el_i, ele_tag in enumerate(el_tags):
            nd1, nd2 = ops.eleNodes(ele_tag)
            Eds[j, el_i, :] = [ops.nodeDisp(nd1)[0],
                                  ops.nodeDisp(nd1)[1],
                                  ops.nodeDisp(nd1)[2],
                                  ops.nodeDisp(nd2)[0],
                                  ops.nodeDisp(nd2)[1],
                                  ops.nodeDisp(nd2)[2]]

        if ok != 0:
            # ------------------------ If not converged, reduce the increment -------------------------
            unconvergeSteps += 1
            Dts = 20  # Analysis loop with 10x smaller increments
            smallDincr = Dincr / Dts
            for k in range(1, Dts):
                if printProgression:
                    print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
                ops.integrator("DisplacementControl", ControlNode, ControlNodeDof, smallDincr)
                ok = ops.analyze(1)
                timeV[j] = ops.getTime()
            # ------------------------ If not converged --------------------------------------------
            if ok != 0:
                print("Problem running Cyclic analysis for the model : Ending analysis ")

        D0 = D1  # move to next step
        finishedSteps = j + 1
        disp = ops.nodeDisp(ControlNode, ControlNodeDof)
        baseShear = -ops.getLoadFactor(2) / 1000  # Convert to from N to kN
        dispData[j + 1] = disp
        baseShearData[j + 1] = baseShear

        if printProgression:
            print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
            print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')

    # if printProgression:
    # toc = time.time()
    # print('CYCLIC ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
                'marker': '', 'markersize': 6}
    # 1. animate the deformated shape
    anim = opsv.anim_defo(Eds, timeV, 1, fmt_defo=fmt_defo, xlim=[-2300, 2300], ylim=[-750, 3000])

    plt.show()

    plt.show()
    return [dispData[0:finishedSteps], -baseShearData[0:finishedSteps]]


build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, numED, numPT, areaED, areaPT, fyPT, fyED, eleH=16, eleL=8, printProgression=True)
DisplacementStep = generate_increasing_cyclic_loading(num_cycles=6, initial_displacement=7, max_displacement=80, num_points=200, repetition_cycles=1)
run_gravity()
[x, y] = run_cyclic(DisplacementStep, plotResults=False, printProgression=True, recordData=False)
# plotting(x, y, 'Displacement (mm)', 'Base Shear (kN)', f'test', save_fig=False, plotValidation=False)
