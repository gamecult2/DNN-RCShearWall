import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from utils import *

import openseespy.postprocessing.Get_Rendering as opsplt
# import vfo.vfo as vfo
# import opsvis as opsv


def reset_analysis():

    ops.setTime(0.0)  # Set the time in the Domain to zero
    ops.loadConst()  # Set the loads constant in the domain
    ops.remove('recorders')  # Remove all recorder objects.
    ops.wipeAnalysis()  # destroy all components of the Analysis object
    ops.wipe()


def build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, loadF, numED, numPT, areaED, areaPT, fyPT, fyED, tenPT, eleH=3, eleL=10, printProgression=True):

    ops.wipe()
    ops.model('Basic', '-ndm', 2, '-ndf', 3)

    # Set geometry, ops.nodes, boundary conditions
    lweb = lw - (2 * lbe)  # Web length
    eleBE = 2
    eleWeb = eleL - eleBE
    elelweb = lweb / eleWeb
    Ag = tw * lweb + 2 * (tb * lbe)  # Wall area

    # Define Nodes (for MVLEM)
    for i in range(2, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))

    # Define control node and DOF
    global controlNode, controlNodeDof, eH, eL
    controlNode = eleH + 1  # Control Node (TopNode)
    controlNodeDof = 1  # Control DOF 1 = X-direction
    eH, eL = eleH, eleL

    # Define Axial Load on Top Node in N according to (ACI 318-19 Section 22.4.2.2)
    global Aload
    Aload = 0.85 * abs(fc) * Ag * loadF # Axial load
    if printProgression:
        print('Axial load = ', Aload/1000)

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

    unbED = hw / eleH
    ancED = 0
    # Calculate locED based on numED and lw
    locED = [((-lw / 6) / 2) + ((lw / 6) / (numED - 1)) * i for i in range(numED)]
    locED = [10 if x == 0 else x for x in locED]
    # print("locED", locED)
    # Initialize lists for node IDs
    botED = []
    topED = []
    # Energy Dissipating Rebars Node
    for j in range(numED):
        node_id_bottom = 200 + j
        node_id_top = 300 + j
        ops.node(node_id_bottom, locED[j], -ancED)
        ops.node(node_id_top, locED[j], unbED)
        botED.append(node_id_bottom)
        topED.append(node_id_top)
        # print('EDnode', node_id_bottom, locED[j], -ancED)
        # print('EDnode', node_id_top, locED[j], unbED)
    # print('botED', botED)
    # print('topED', topED)

    ancPT = 600
    # Calculate locED based on numED and lw
    locPT = [((-lw / 3) / 2) + ((lw / 3) / (numPT - 1)) * i for i in range(numPT)]
    locPT = [10 if x == 0 else x for x in locPT]
    # print("locPT", locPT)
    botPT = []
    topPT = []
    # PT Node
    for i in range(0, numPT):
        node_id_bottom = 400 + i
        node_id_top = 500 + i
        ops.node(node_id_bottom, locPT[i], -ancPT)
        ops.node(node_id_top, locPT[i], hw)
        botPT.append(node_id_bottom)
        topPT.append(node_id_top)
        # print('PTnode', node_id_bottom, locPT[i], -ancPT)
        # print('PTnode', node_id_top, locPT[i], hw)
    # print('botPT', botPT)
    # print('topPT', topPT)

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
    IDsYb, IDsYw = 1, 2  # Steel ID
    Es = 200 * GPa  # Young's modulus
    Bs, R0, cR1, cR2 = 0.01, 20.0, 0.925, 0.0015  # Strain-hardening and curve parameters
    byb = 0.01  # STEEL Y BE - strain hardening - tension & compression
    byw = 0.02  # STEEL Y WEB - strain hardening - tension & compression
    bX = 0.02   # STEEL X - strain hardening - tension & compression

    # Steel ED
    IDmatED = 3
    fyED = fyED

    # Steel PT
    IDmatPT = 4
    fyPT = fyPT
    a1, a2, a3, a4 = 0, 1, 0, 1
    sigInit = tenPT * MPa

    # SteelMPF model
    ops.uniaxialMaterial('SteelMPF', IDsYb, fyb, fyb, Es, byb, byb, R0, cR1, cR2)  # Steel Y boundary
    ops.uniaxialMaterial('SteelMPF', IDsYw, fyw, fyw, Es, byw, byw, R0, cR1, cR2)  # Steel Y web
    ops.uniaxialMaterial('Steel02', IDmatED, fyED, Es, Bs, R0, cR1, cR2)  # Steel ED
    ops.uniaxialMaterial('Steel02', IDmatPT, fyPT, Es, Bs, R0, cR1, cR2, a1, a2, a3, a4, sigInit)  # Steel PT

    # Define "ConcreteCM" uni-axial materials
    IDconcWeb = 5
    IDconcBE = 6

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
    # fcC = 89.6318 * MPa
    # -------------------------- ConcreteCM model --------------------------------------------
    ops.uniaxialMaterial('ConcreteCM', IDconcWeb, fcU, ecU, EcU, rU, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    ops.uniaxialMaterial('ConcreteCM', IDconcBE, fcC, ecC, EcC, rC, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    # print('--------------------------------------------------------------------------------------------------')
    # print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    # print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    IDmatSpring = 7
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6
    ops.uniaxialMaterial('Elastic', IDmatSpring, Kshear)

    # Define ENT Material
    IDmatENT = 9
    # EENT = 5 * 32500.0 * lw * tw / (hw * numENT)
    # print('EENT', EENT)
    EENT = (4 * EcC * lw * tw / (lw / 4)) / numENT
    # print('EENT', EENT)
    ops.uniaxialMaterial('ENT', IDmatENT, EENT)

    # ------------------------------
    #  Define SFI_MVLEM elements
    # ------------------------------
    MVLEM_thick = [tw] * eleL
    MVLEM_width = [lbe if i in (0, eleL - 1) else elelweb for i in range(eleL)]
    MVLEM_rho = [rouYb if i in (0, eleL - 1) else rouYw for i in range(eleL)]
    MVLEM_matConcrete = [IDconcBE if i in (0, eleL - 1) else IDconcWeb for i in range(eleL)]
    MVLEM_matSteel = [IDsYb if i in (0, eleL - 1) else IDsYw for i in range(eleL)]

    for i in range(eleH):
        # ------------------ MVLEM ----------------------------------------------
        ops.element('MVLEM', (101 + i), 0.0, WallNode[i], WallNode[i + 1], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', IDmatSpring)
        # print('element', 'MVLEM', (101+i), 0.0, WallNode[i], WallNode[i + 1], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', matSpring)

    # Define Element for ENT (Zero-Length element)
    for i in range(0, numENT):
        ops.element('zeroLength', (601 + i), botENT[i], topENT[i], '-mat', IDmatENT, '-dir', 2)
        # print('zeroLength', i + 601, botENT[i], topENT[i], '-mat', matENT, '-dir', 2)

    A = 1e5
    E = 3.0e11
    Iz = 9.0e4
    geomTransfTag_PDelta = 1
    ops.geomTransf('PDelta', geomTransfTag_PDelta)
    # Define Element for ENT (Rigid element between ENT top nodes)
    for i in range(0, numENT - 1):
        ops.element('elasticBeamColumn', (1001 + i), topENT[i], topENT[i + 1], A, E, Iz, geomTransfTag_PDelta)
        # print('ENT-rigidColumn', i + 1001, topENT[i], topENT[i + 1], A, E, I, geomTransfTag_PDelta)

    # Define Elements for ED (Truss element between ED top and bottom nodes)
    for i in range(numED):
        ops.element('truss', (201 + i), (200 + i), (300 + i), areaED, IDmatED)
        # print('ED-truss', 201+i, 200+i, 300+i, areaED, matED)

    # Define ED elements
    topED.insert(numED // 2, WallNode[1])
    for i in range(numED):
        ops.element('elasticBeamColumn', (221 + i), topED[i], topED[i + 1], A, E, Iz, geomTransfTag_PDelta)
        # print('ED-rigidColumn', 221 + i, topED[i], topED[i + 1], A, E, Iz, geomTransfTag_PDelta)

    # Define PT elements
    for i in range(numPT):
        ops.element('truss', (301 + i), botPT[i], topPT[i], areaPT, IDmatPT)
        # print('PT-truss', 301+i, botPT[i], topPT[i], areaPT, matPT)

    # Define PT top rigid elements
    topPT.insert(numPT // 2, WallNode[-1])
    for i in range(numPT):
        ops.element('elasticBeamColumn', (311 + i), topPT[i], topPT[i + 1], A, E, Iz, geomTransfTag_PDelta)
        # print('PT-rigidColumn', 311 + i, topPT[i], topPT[i + 1], A, E, Iz, geomTransfTag_PDelta)

    opsv.plot_model()
    plt.show()
    # opsplt.plot_model('nodes', 'elements')
    plt.show()


def run_gravity(steps=10, printProgression=False):
    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(controlNode, 0, -Aload, 0)
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


def run_analysis(DisplacementStep, analysis='cyclic', printProgression=False):
    if printProgression:
        print(f"RUNNING {analysis.upper()} ANALYSIS")

    # For pushover analysis, generate linear displacement steps
    if analysis == 'pushover':
        MaxDisp = max(DisplacementStep)
        dispIncr = (max(DisplacementStep) / len(DisplacementStep))
        DisplacementStep = [dispIncr * i for i in range(int(MaxDisp / dispIncr))]

    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(controlNode, *[1.0, 0.0, 0.0])  # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('RCM')
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1e-8, 300, 0)
    ops.algorithm('KrylovNewton')
    ops.analysis('Static')

    print("Done Cyclic analysis")

    # Define analysis parameters
    maxUnconvergedSteps = 2
    unconvergeSteps = 0
    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    dispData = np.zeros(Nsteps + 1)
    baseShearData = np.zeros(Nsteps + 1)

    # el_tags = ops.getEleTags()
    # nels = len(el_tags)
    # Eds = np.zeros((Nsteps, nels, 6))
    # timeV = np.zeros(Nsteps)
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
        ops.integrator("DisplacementControl", controlNode, controlNodeDof, Dincr)
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

        if ok != 0:
            # ------------------------ If not converged, reduce the increment -------------------------
            unconvergeSteps += 1
            Dts = 20  # Analysis loop with 10x smaller increments
            smallDincr = Dincr / Dts
            for k in range(1, Dts):
                if printProgression:
                    print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
                ops.integrator("DisplacementControl", controlNode, controlNodeDof, smallDincr)
                ok = ops.analyze(1)
                # timeV[j] = ops.getTime()
            # ------------------------ If not converged --------------------------------------------
            if ok != 0:
                print("Problem running Cyclic analysis for the model : Ending analysis ")

        D0 = D1  # move to next step
        finishedSteps = j + 1
        disp = ops.nodeDisp(controlNode, controlNodeDof)
        baseShear = -ops.getLoadFactor(2) / 1000  # Convert to from N to kN
        dispData[j + 1] = disp
        baseShearData[j + 1] = baseShear

        # if printProgression:
        #     print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
        #     print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')

    # if printProgression:
    # toc = time.time()
    # print('CYCLIC ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    # fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
    #             'marker': '', 'markersize': 6}
    # # 1. animate the deformated shape
    # anim = opsv.anim_defo(Eds, timeV, 50, fmt_defo=fmt_defo, xlim=[-2300, 2300], ylim=[-750, 3000])
    #
    return [dispData[0:finishedSteps], -baseShearData[0:finishedSteps]]
