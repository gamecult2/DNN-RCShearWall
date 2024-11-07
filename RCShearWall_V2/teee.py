import math
import os
import time

import h5py
import numpy as np
import openseespy.opensees as ops
# import opsvis
# import opsvis as opsv
#import vfo.vfo as vfo
import matplotlib.pyplot as plt
# import NewGeneratePeaks
from collections import defaultdict
from scipy.io import savemat


def hertz_impact_material(e=0.7, r1=0.064, fmax=250000, v0=4.8, t=0.005):
    miu1 = 0.3  # Impactor possion ratio
    miu2 = 0.2  # Beam possion ratio
    e1 = 1.5 * 10 ** 5 * 10 ** 6  # Impactor modulus (in Pa)
    e2 = 3.25 * 10 ** 4 * 10 ** 6  # Beam modulus (in Pa)
    # print(e2)
    lamda1 = (1 - miu1 ** 2) / e1
    lamda2 = (1 - miu2 ** 2) / e2
    kh = 4 / 3 * math.sqrt(r1) * (1 / (lamda1 + lamda2))  # Hertz contact stiffness
    # Calculate deltam (maximum penetration)
    deltam = (fmax / kh) ** (2 / 3)
    deltae = (kh * deltam ** 2.5 * (1 - e ** 2)) / 2.5
    keff = kh * math.sqrt(deltam)
    kt1 = keff + (deltae / (0.1 * deltam ** 2))
    kt2 = keff - (deltae / (0.9 * deltam ** 2))
    kt1 = kt1 / 1000
    kt2 = kt2 / 1000
    deltam = deltam * 1000
    deltay = - 0.1 * deltam
    gap = - 1000 * v0 * t / 2
    return kt1, kt2, deltay, gap


def save_result(results: defaultdict, numENT=9, element_length=[]):
    #MVLEM_D_impact_point = ops.nodeDisp(int(BeamImpArg), 1)
    #results["MVLEM_D_impact_point"].append(MVLEM_D_impact_point)
    #MVLEM_D_impact_point = ops.nodeAccel(int(BeamImpArg), 1)
    #results["MVLEM_D_impact_point"].append(MVLEM_D_impact_point)
    #MVLEM_F_impact = ops.eleResponse(101, "force")
    #results["MVLEM_F_impact"].append(MVLEM_F_impact)
    # pier 1
    for m in range(0, numENT):
        DRocking_P1Base = ops.eleResponse(m + 601, "deformation")
        results[f"DRocking_P1Base_ele_{m + 601}"].append(DRocking_P1Base)
        DRocking_P1Top = ops.eleResponse(m + 701, "deformation")
        results[f"DRocking_P1Top_ele_{m + 701}"].append(DRocking_P1Top)
        DRocking_P2Base = ops.eleResponse(m + 801, "deformation")
        results[f"DRocking_P2Base_ele_{m + 801}"].append(DRocking_P2Base)
        DRocking_P2Top = ops.eleResponse(m + 901, "deformation")
        results[f"DRocking_P2Top_ele_{m + 901}"].append(DRocking_P2Top)
    MVLEM_DX = ops.nodeDisp(1000, 1)
    results[f"MVLEM_DX_node{501}"].append(MVLEM_DX)  #当作501节点记录，方便后处理循环读取位移数据
    MVLEM_DY = ops.nodeDisp(1000, 2)
    results[f"MVLEM_DY_node{501}"].append(MVLEM_DY)
    MVLEM_DX = ops.nodeDisp(3000, 1)
    results[f"MVLEM_DX_node{601}"].append(MVLEM_DX)
    MVLEM_DY = ops.nodeDisp(3000, 2)
    results[f"MVLEM_DY_node{601}"].append(MVLEM_DY)
    for k in range(1, len(element_length) - 1):
        # Pier 1
        MVLEM_DX = ops.nodeDisp(500 + k + 1, 1)
        results[f"MVLEM_DX_node{500 + k + 1}"].append(MVLEM_DX)
        MVLEM_DY = ops.nodeDisp(500 + k + 1, 2)
        results[f"MVLEM_DY_node{500 + k + 1}"].append(MVLEM_DY)
        # Pier 2
        MVLEM_DX = ops.nodeDisp(600 + k + 1, 1)
        results[f"MVLEM_DX_node{600 + k + 1}"].append(MVLEM_DX)
        MVLEM_DY = ops.nodeDisp(600 + k + 1, 2)
        results[f"MVLEM_DY_node{600 + k + 1}"].append(MVLEM_DY)
    MVLEM_DX = ops.nodeDisp(1990, 1)
    results[f"MVLEM_DX_node{546}"].append(MVLEM_DX)  # 当作546节点记录，方便后处理循环读取位移数据
    MVLEM_DY = ops.nodeDisp(1990, 2)
    results[f"MVLEM_DY_node{546}"].append(MVLEM_DY)
    MVLEM_DX = ops.nodeDisp(3990, 1)
    results[f"MVLEM_DX_node{646}"].append(MVLEM_DX)
    MVLEM_DY = ops.nodeDisp(3990, 2)
    results[f"MVLEM_DY_node{646}"].append(MVLEM_DY)
    for i in range(0, len(element_length) - 1):
        #pier 1
        MVLEM_C = ops.eleResponse(i + 101, "Curvature")
        results[f"MVLEM_Curvature_ele_{i + 101}"].append(MVLEM_C)
        #MVLEM_F = ops.eleResponse(i + 101, "force")
        #results[f"MVLEM_F_ele_{i + 101}"].append(MVLEM_F)
        #MVLEM_ShearD = ops.eleResponse(i + 101, "ShearDef")
        #results[f"MVLEM_F_ele_{i + 101}"].append(MVLEM_ShearD)
        #pier 2
        MVLEM_C = ops.eleResponse(i + 151, "Curvature")
        results[f"MVLEM_Curvature_ele_{i + 151}"].append(MVLEM_C)
        #MVLEM_F = ops.eleResponse(i + 151, "force")
        #results[f"MVLEM_F_ele_{i + 151}"].append(MVLEM_F)
        #MVLEM_ShearD = ops.eleResponse(i + 151, "ShearDef")
        #results[f"MVLEM_F_ele_{i + 151}"].append(MVLEM_ShearD)
        for j in range(0, 10):
            #pier 1
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "strain_stress_steelX")
            results[f'MVLEM_steelX_ele_{i + 101}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "strain_stress_steelY")
            results[f'MVLEM_steelY_ele_{i + 101}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "strain_stress_concrete1")
            results[f'MVLEM_concrete1_ele_{i + 101}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "strain_stress_concrete2")
            results[f'MVLEM_concrete2_ele_{i + 101}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "strain_stress_interlock1")
            results[f'MVLEM_interlock1_ele_{i + 101}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "strain_stress_interlock2")
            results[f'MVLEM_interlock2_ele_{i + 101}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 101, "RCPanel", j + 1, "cracking_angles")
            results[f'MVLEM_cracking_angles_ele_{i + 101}_panel_{j + 1}'].append(resp)
            #pier 2
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "strain_stress_steelX")
            results[f'MVLEM_steelX_ele_{i + 151}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "strain_stress_steelY")
            results[f'MVLEM_steelY_ele_{i + 151}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "strain_stress_concrete1")
            results[f'MVLEM_concrete1_ele_{i + 151}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "strain_stress_concrete2")
            results[f'MVLEM_concrete2_ele_{i + 151}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "strain_stress_interlock1")
            results[f'MVLEM_interlock1_ele_{i + 151}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "strain_stress_interlock2")
            results[f'MVLEM_interlock2_ele_{i + 151}_panel_{j + 1}'].append(resp)
            resp = ops.eleResponse(i + 151, "RCPanel", j + 1, "cracking_angles")
            results[f'MVLEM_cracking_angles_ele_{i + 151}_panel_{j + 1}'].append(resp)


def Run_DCRP1_cyclic(displacement):
    ops.wipe()
    ops.model('Basic', '-ndm', 2, '-ndf', 3)
    pielen = 2250
    piewid = 2300
    unblED = 400
    anclED = 0
    anclPT = 600
    pieH = 540
    pieB = 400
    numENT = 9
    numED = 4
    numPT = 2
    locED = [-135, -120, 120, 135]
    locPT = [-100, 100]
    t = 400
    # base = 0
    # pier1 = 0
    # Top = 1000
    # pier2 = 2000
    baspier1 = 0
    toppier1 = 1000
    baspier2 = 2000
    toppier2 = 3000
    # Create nodes
    # ------------------------------------------------------------------------
    # node nodeId xCrd yCrd..
    # ------------------------------------------------------------------------

    Impact_coordinate_Y = 600
    #                              1   2   3   4   5   6   7   8  9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45
    element_length = np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
    # element_length = np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 400, 400, 450])
    #                              1   2   3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39    40    41    42    43    44    45    46
    Nodes_coordinate_Y = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250])
    # Nodes_coordinate_Y = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1400, 1800, 2250])

    # Pier 1
    # ops.node(1000, 0, 0)
    # print('node', 1000, 0, 0)
    for i in range(1, len(element_length) - 1):
        ops.node(500 + i + 1, 0, int(Nodes_coordinate_Y[i]))
        print('node', 500 + i + 1, 0, int(Nodes_coordinate_Y[i]))
    # Pier 2
    # ops.node(3000, piewid, 0)
    # print('node', 3000, 0, 0)
    for i in range(1, len(element_length) - 1):
        ops.node(600 + i + 1, piewid, int(Nodes_coordinate_Y[i]))
        print('node', 600 + i + 1, piewid, int(Nodes_coordinate_Y[i]))
    # Bent Beam
    ops.node(13, 0, pielen + 300)
    ops.node(14, piewid, pielen + 300)
    print('node', 13, 0, pielen + 300)
    print('node', 14, piewid, pielen + 300)
    # Rocking Nodes (Pier 1 and Pier 2)

    # Pier 1 top
    # [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004]
    #   |     |     |     |     |     |     |     |     |
    # [1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994]

    # Pier 1 base
    # [996,  997,  998,  999,  1000, 1001, 1002, 1003, 1004]
    #   |     |     |     |     |     |     |     |     |
    # [986,  987,  988,  989,  990,  991,  992,  993,  994]

    # Pier 2 top
    # [3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004]
    #   |     |     |     |     |     |     |     |     |
    # [3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994]

    # Pier 2 base
    # [2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004]
    #   |     |     |     |     |     |     |     |     |
    # [2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994]

    bas1botnodes = []
    bas1topnodes = []
    top1botnodes = []
    top1topnodes = []
    bas2botnodes = []
    bas2topnodes = []
    top2botnodes = []
    top2topnodes = []
    for i in range(0, numENT):
        bas1botnodes.append(int(986 + baspier1 + i))
        bas1topnodes.append(int(986 + baspier1 + numENT + 1 + i))
        top1botnodes.append(int(986 + toppier1 + i))
        top1topnodes.append(int(986 + toppier1 + numENT + 1 + i))
        bas2botnodes.append(int(986 + baspier2 + i))
        bas2topnodes.append(int(986 + baspier2 + numENT + 1 + i))
        top2botnodes.append(int(986 + toppier2 + i))
        top2topnodes.append(int(986 + toppier2 + numENT + 1 + i))
    # print(bas1botnode)
    # print(bas1topnode)
    # print(top1botnode)
    # print(top1topnode)
    # print(bas2botnode)
    # print(bas2topnode)
    # print(top2botnode)
    # print(top2topnode)
    for i in range(0, numENT):
        ops.node(bas1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
        ops.node(bas1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
        ops.node(top1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
        ops.node(top1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
        ops.node(bas2botnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), 0)
        ops.node(bas2topnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), 0)
        ops.node(top2botnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
        ops.node(top2topnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
        print('node', bas1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
        print('node', bas1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), 0)
        print('node', top1botnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
        print('node', top1topnodes[i], -pieH / 2 + i * pieH / (numENT - 1), pielen)
        print('node', bas2botnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), 0)
        print('node', bas2topnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), 0)
        print('node', top2botnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
        print('node', top2topnodes[i], piewid + (-pieH) / 2 + i * pieH / (numENT - 1), pielen)
    pie1node = []
    pie2node = []
    pie1node.append(bas1topnodes[int((numENT - 1) / 2)])
    pie2node.append(bas2topnodes[int((numENT - 1) / 2)])
    for i in range(1, len(element_length) - 1):
        pie1node.append(500 + i + 1)
        pie2node.append(600 + i + 1)
    pie1node.append(top1botnodes[int((numENT - 1) / 2)])
    pie2node.append(top2botnodes[int((numENT - 1) / 2)])
    print(pie1node)
    print(pie2node)

    # Energy Dissipating Rebars Node
    for i in range(0, 2):
        for j in range(0, numED):
            ops.node(100 + numED * i + j, locED[j] + i * piewid, -anclED)
            ops.node(200 + numED * i + j, locED[j] + i * piewid, unblED)
            print('node', 100 + numED * i + j, locED[j] + i * piewid, -anclED, ';')
            print('node', 200 + numED * i + j, locED[j] + i * piewid, unblED, ';')
    # PT Node
    for i in range(0, 2):
        for j in range(0, numPT):
            ops.node(300 + numPT * i + j, locPT[j] + i * piewid, -anclPT)
            ops.node(400 + numPT * i + j, locPT[j] + i * piewid, pielen)
            print('node', 300 + numPT * i + j, locPT[j] + i * piewid, -anclPT)
            print('node', 400 + numPT * i + j, locPT[j] + i * piewid, pielen)

    # ------------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------------
    # Constraints of base spring bottom nodes of pier1 and pier2
    for i in range(0, len(bas1botnodes)):
        ops.fix(bas1botnodes[i], 1, 1, 1)
        ops.fix(bas2botnodes[i], 1, 1, 1)
        print('fix', bas1botnodes[i], 1, 1, 1)
        print('fix', bas2botnodes[i], 1, 1, 1)
    # Constraints of base spring top nodes of pier1 and pier2
    for i in range(0, 1):
        ops.fix(bas1topnodes[i], 1, 0, 0)
        ops.fix(bas2topnodes[i], 1, 0, 0)
        print('fix', bas1topnodes[i], 1, 0, 0)
        print('fix', bas2topnodes[i], 1, 0, 0)
    # Constraints of ED truss bottom nodes of pier1 and pier2
    for i in range(0, 2):
        for j in range(0, numED):
            ops.fix(100 + numED * i + j, 1, 1, 1)
            print('fix', 100 + numED * i + j, 1, 1, 1)
    # Constraints of PT truss bottom nodes of pier1 and pier2
    for i in range(0, 2):
        for j in range(0, numPT):
            ops.fix(300 + numPT * i + j, 1, 1, 1)
            print('fix', 300 + numPT * i + j, 1, 1, 1)
    # EqualDOF of top spring bottom and top nodes of pier1 and pier2
    for i in range(0, len(top1botnodes)):
        ops.equalDOF(top1topnodes[i], top1botnodes[i], 1)
        ops.equalDOF(top2topnodes[i], top2botnodes[i], 1)
        print('equalDOF', top1topnodes[i], top1botnodes[i], 1)
        print('equalDOF', top2topnodes[i], top2botnodes[i], 1)
    # ------------------------------------------------------------------------
    # Define uniaxial materials for 2D RC Panel Constitutive Model (FSAM)
    # ------------------------------------------------------------------------
    # Steel X
    fyx = 425  # yield strength of transverse reinforcement in ksi
    bx = 0.01  # strain hardening coefficient of transverse reinforcement
    # Steel Y
    fyY = 455  # yield strength of longitudinal reinforcement in ksi
    by = 0.01  # strain hardening coefficient of longitudinal reinforcement
    # Steel misc
    Esy = 200000  # Young's modulus (199947.9615MPa)
    Esx = 200000  # Young's modulus (199947.9615MPa)
    R0 = 20  # Initial value of curvature parameter
    A1 = 0.925  # Curvature degradation parameter
    A2 = 0.15  # Curvature degradation parameter
    # Steel ED
    IDED = 3
    fyED = 291
    EED = 200000
    b = 0.01
    # Steel PT
    IDPT = 4
    fyPT = 1860
    EPT = 200000
    a1 = 0
    a2 = 1
    a3 = 0
    a4 = 1
    sigInit = 750
    # Build SteelMPF material
    ops.uniaxialMaterial('SteelMPF', 1, fyx, fyx, Esx, bx, bx, R0, A1, A2)  #Steel X
    ops.uniaxialMaterial('SteelMPF', 2, fyY, fyY, Esy, by, by, R0, A1, A2)  #Steel Y
    ops.uniaxialMaterial('Steel02', IDED, fyED, EED, b, R0, A1, A2)  # Steel ED
    ops.uniaxialMaterial('Steel02', IDPT, fyPT, EPT, b, R0, A1, A2, a1, a2, a3, a4, sigInit)  # Steel PT
    # Concrete
    # unconfined concrete
    fpc = 38.46  # peak compressive stress(5.578)
    ec0 = -0.002161  # strain at peak compressive stress
    ft = 3.8621  # peak tensile stress
    et = 0.000238  # strain at peak tensile stress
    Ec = 32405.9362  # Young's modulus
    xcrnu = 1.035  # cracking strain (compression)
    xcrp = 10000  # cracking strain (tension)
    ru = 5.5375  # shape parameter (compression)
    rt = 25  # shape parameter (tension)
    # confined concrete
    fpcc = 57.69  # peak compressive stress
    ec0c = -0.00231  # strain at peak compressive stress
    ftc = 4.4034  # peak tensile stress
    etc = 0.0002463  # strain at peak tensile stress
    Ecc = 35756.3623  # Young's modulus
    xcrnc = 1.035  # cracking strain (compression)
    rc = 7.7687  # shape parameter (compression)
    # Build ConcreteCM material
    ops.uniaxialMaterial('ConcreteCM', 5, -fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp, '-GapClose', 0)  # unconfined concrete
    ops.uniaxialMaterial('ConcreteCM', 6, -fpcc, ec0c, Ecc, rc, xcrnc, ftc, etc, rt, xcrp, '-GapClose', 0)  # confined concrete
    # ---------------------------------------
    #  Define 2D RC Panel Material (FSAM)
    # ---------------------------------------

    # Reinforcing ratios
    rouX = 0.011  # Reinforcing ratio of transverse rebar
    rouYc = 0.036  # Reinforcing ratio of cover
    rouY1 = 0.036  # Reinforcing ratio of longitudinal rebar(12f20: different from Han 2019)
    nu = 0.2  # Friction coefficient
    alfadow = 0.012  # Dowel action stiffness parameter

    # Build ndMaterial FSAM
    ops.nDMaterial('FSAM', 7, 0.0, 1, 2, 5, rouX, rouYc, nu, alfadow)
    ops.nDMaterial('FSAM', 8, 0.0, 1, 2, 6, rouX, rouY1, nu, alfadow)
    # Define ENT Material
    IDENT = 9
    EENT = 2.5 * 32500.0 * pieH * pieB / (pielen * numENT)
    # print(EENT)

    # print(EENT)
    ops.uniaxialMaterial('ENT', IDENT, EENT)

    # ------------------------------
    #  Define SFI_MVLEM elements
    # ------------------------------
    t = 400
    for i in range(0, len(element_length) - 1):
        ops.element('E_SFI', i + 101, pie1node[i], pie1node[i + 1], 10, 0.4, '-thick', t, t, t, t, t, t, t, t, t, t, '-width', 70, 50, 50, 50, 50, 50, 50, 50, 50, 70, '-mat', 7, 8, 8, 8, 8, 8, 8, 8, 8, 7)
        ops.element('E_SFI', i + 151, pie2node[i], pie2node[i + 1], 10, 0.4, '-thick', t, t, t, t, t, t, t, t, t, t, '-width', 70, 50, 50, 50, 50, 50, 50, 50, 50, 70, '-mat', 7, 8, 8, 8, 8, 8, 8, 8, 8, 7)
        print('element', 'E_SFI', i + 101, pie1node[i], pie1node[i + 1], 10, 0.4, '-thick', t, t, t, t, t, t, t, t, t, t, '-width', 70, 50, 50, 50, 50, 50, 50, 50, 50, 70, '-mat', 7, 8, 8, 8, 8, 8, 8, 8, 8, 7)
        print('element', 'E_SFI', i + 151, pie2node[i], pie2node[i + 1], 10, 0.4, '-thick', t, t, t, t, t, t, t, t, t, t, '-width', 70, 50, 50, 50, 50, 50, 50, 50, 50, 70, '-mat', 7, 8, 8, 8, 8, 8, 8, 8, 8, 7)

    # Define Zero-Length Element
    for i in range(0, numENT):
        ops.element('zeroLength', i + 601, bas1botnodes[i], bas1topnodes[i], '-mat', IDENT, '-dir', 2)
        ops.element('zeroLength', i + 701, top1botnodes[i], top1topnodes[i], '-mat', IDENT, '-dir', 2)
        ops.element('zeroLength', i + 801, bas2botnodes[i], bas2topnodes[i], '-mat', IDENT, '-dir', 2)
        ops.element('zeroLength', i + 901, top2botnodes[i], top2topnodes[i], '-mat', IDENT, '-dir', 2)
        print('zeroLength', i + 601, bas1botnodes[i], bas1topnodes[i], '-mat', IDENT, '-dir', 2)
        print('zeroLength', i + 701, top1botnodes[i], top1topnodes[i], '-mat', IDENT, '-dir', 2)
        print('zeroLength', i + 801, bas2botnodes[i], bas2topnodes[i], '-mat', IDENT, '-dir', 2)
        print('zeroLength', i + 901, top2botnodes[i], top2topnodes[i], '-mat', IDENT, '-dir', 2)
    geomTransfTag_PDelta = 1
    ops.geomTransf('PDelta', geomTransfTag_PDelta)
    for i in range(0, numENT - 1):
        ops.element('elasticBeamColumn', i + 1601, bas1topnodes[i], bas1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        ops.element('elasticBeamColumn', i + 1701, top1botnodes[i], top1botnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        ops.element('elasticBeamColumn', i + 1711, top1topnodes[i], top1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        ops.element('elasticBeamColumn', i + 1801, bas2topnodes[i], bas2topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        ops.element('elasticBeamColumn', i + 1901, top2botnodes[i], top2botnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        ops.element('elasticBeamColumn', i + 1911, top2topnodes[i], top2topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1601, bas1topnodes[i], bas1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1701, top1botnodes[i], top1botnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1711, top1topnodes[i], top1topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1801, bas2topnodes[i], bas2topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1901, top2botnodes[i], top2botnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
        print('elasticBeamColumn', i + 1911, top2topnodes[i], top2topnodes[i + 1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # Define ED elements
    ops.element('truss', 201, 100, 200, 314.11 * 2, IDED)
    ops.element('truss', 202, 101, 201, 314.11 * 2, IDED)
    ops.element('truss', 203, 102, 202, 353.37 * 1, IDED)
    ops.element('truss', 204, 103, 203, 353.37 * 1, IDED)
    ops.element('elasticBeamColumn', 211, 200, 201, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 212, 201, 500 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 213, 500 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 202, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 214, 202, 203, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'truss', 201, 100, 200, 314.11 * 2, IDED)
    print('element', 'truss', 202, 101, 201, 314.11 * 2, IDED)
    print('element', 'truss', 203, 102, 202, 353.37 * 1, IDED)
    print('element', 'truss', 204, 103, 203, 353.37 * 1, IDED)
    print('element', 'elasticBeamColumn', 211, 200, 201, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 212, 201, 500 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 213, 500 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 202, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 214, 202, 203, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('truss', 221, 104, 204, 314.11 * 2, IDED)
    ops.element('truss', 222, 105, 205, 314.11 * 2, IDED)
    ops.element('truss', 223, 106, 206, 353.37 * 1, IDED)
    ops.element('truss', 224, 107, 207, 353.37 * 1, IDED)
    ops.element('elasticBeamColumn', 231, 204, 205, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 232, 205, 600 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 233, 600 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 206, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 234, 206, 207, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'truss', 221, 104, 204, 314.11 * 2, IDED)
    print('element', 'truss', 222, 105, 205, 314.11 * 2, IDED)
    print('element', 'truss', 223, 106, 206, 353.37 * 1, IDED)
    print('element', 'truss', 224, 107, 207, 353.37 * 1, IDED)
    print('element', 'elasticBeamColumn', 231, 204, 205, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 232, 205, 600 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 233, 600 + int(np.argwhere(Nodes_coordinate_Y == unblED)) + 1, 206, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 234, 206, 207, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # Define PT elements
    ops.element('truss', 301, 300, 400, 280, IDPT)
    ops.element('truss', 302, 301, 401, 280, IDPT)
    ops.element('elasticBeamColumn', 311, 400, pie1node[-1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 312, pie1node[-1], 401, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('truss', 321, 302, 402, 280, IDPT)
    ops.element('truss', 322, 303, 403, 280, IDPT)
    ops.element('elasticBeamColumn', 331, 402, pie2node[-1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 332, pie2node[-1], 403, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'truss', 301, 300, 400, 280, IDPT)
    print('element', 'truss', 302, 301, 401, 280, IDPT)
    print('element', 'elasticBeamColumn', 311, 400, pie1node[-1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 312, pie1node[-1], 401, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'truss', 321, 302, 402, 280, IDPT)
    print('element', 'truss', 322, 303, 403, 280, IDPT)
    print('element', 'elasticBeamColumn', 331, 402, pie2node[-1], 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('element', 'elasticBeamColumn', 332, pie2node[-1], 403, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    # Define Rigid Beam
    ops.element('elasticBeamColumn', 401, 2000, 13, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 402, 13, 14, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    ops.element('elasticBeamColumn', 403, 14, 4000, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', 401, 2000, 2 * len(element_length) + 1, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', 402, 2 * len(element_length) + 1, 2 * len(element_length) + 2, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)
    print('elasticBeamColumn', 403, 2 * len(element_length) + 2, 4000, 1e5, 3.0e11, 9.0e4, geomTransfTag_PDelta)

    N = 666400  # in N
    IDctrlNode1 = 13
    IDctrlNode2 = 14
    # print('IDctrlNode1 =', IDctrlNode1, 'IDctrlNode2 =', IDctrlNode2,)
    IDctrlDOF = 1

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(IDctrlNode1, 0, -N, 0)
    ops.load(IDctrlNode2, 0, -N, 0)

    # ------------------------------
    # Analysis generation
    # ------------------------------
    # Create the integration scheme, the LoadControl scheme using steps of 0.1
    ops.wipeAnalysis()
    ops.integrator('LoadControl', 0.1)
    # Create the system of equation, a sparse solver with partial pivoting
    ops.system('BandGen')
    # Create the convergence test, the norm of the residual with a tolerance of 1e-5 and a max number of iterations of 100
    ops.test('NormDispIncr', 1.0e-6, 1000)
    # Create the DOF numberer, the reverse Cuthill-McKee algorithm
    ops.numberer('RCM')
    # Create the constraint handler, the transformation method
    ops.constraints('Penalty', 1.0e20, 1.0e20)
    # Create the solution algorithm, a Newton-Raphson algorithm
    ops.algorithm('Newton')
    # Create the analysis object
    ops.analysis('Static')
    # Run analysis

    ops.analyze(10)
    print("Gravity Analysis is completed, return value is 0")

    ops.loadConst('-time', 0.0)

    # -----------------------Begin Cyclic Analysis--------------------------
    print('-----------------------Begin stage2: Impact_Cyclic Analysis--------------------------')
    time.sleep(1)
    #opsvis.plot_model(node_labels=0, element_labels=0)
    # opsvis.plot_mode_shape(1)
    #plt.show()

    dataDir = 'DCRP_1_cyclic'
    # os.chdir('..')
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    os.chdir(dataDir)

    ops.recorder('Node', '-file', f'MVLEM_Dtop.txt', '-time', '-node', IDctrlNode1, '-dof', 1, 'disp')
    # pier 1 Base
    ops.recorder('Element', '-file', f'MVLEM_DRocking_601.txt', '-time', '-ele', 601, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_602.txt', '-time', '-ele', 602, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_603.txt', '-time', '-ele', 603, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_604.txt', '-time', '-ele', 604, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_605.txt', '-time', '-ele', 605, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_606.txt', '-time', '-ele', 606, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_607.txt', '-time', '-ele', 607, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_608.txt', '-time', '-ele', 608, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_609.txt', '-time', '-ele', 609, "deformation")
    # pier 1 Top
    ops.recorder('Element', '-file', f'MVLEM_DRocking_701.txt', '-time', '-ele', 701, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_702.txt', '-time', '-ele', 702, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_703.txt', '-time', '-ele', 703, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_704.txt', '-time', '-ele', 704, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_705.txt', '-time', '-ele', 705, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_706.txt', '-time', '-ele', 706, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_707.txt', '-time', '-ele', 707, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_708.txt', '-time', '-ele', 708, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_709.txt', '-time', '-ele', 709, "deformation")
    # pier 2 Base
    ops.recorder('Element', '-file', f'MVLEM_DRocking_801.txt', '-time', '-ele', 801, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_802.txt', '-time', '-ele', 802, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_803.txt', '-time', '-ele', 803, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_804.txt', '-time', '-ele', 804, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_805.txt', '-time', '-ele', 805, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_806.txt', '-time', '-ele', 806, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_807.txt', '-time', '-ele', 807, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_808.txt', '-time', '-ele', 808, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_809.txt', '-time', '-ele', 809, "deformation")
    # pier 2 Top
    ops.recorder('Element', '-file', f'MVLEM_DRocking_901.txt', '-time', '-ele', 901, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_902.txt', '-time', '-ele', 902, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_903.txt', '-time', '-ele', 903, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_904.txt', '-time', '-ele', 904, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_905.txt', '-time', '-ele', 905, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_906.txt', '-time', '-ele', 906, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_907.txt', '-time', '-ele', 907, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_908.txt', '-time', '-ele', 908, "deformation")
    ops.recorder('Element', '-file', f'MVLEM_DRocking_909.txt', '-time', '-ele', 909, "deformation")
    # ED Rebars
    ops.recorder('Element', '-file', f'MVLEM_F_ED_201.txt', '-time', '-ele', 201, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_202.txt', '-time', '-ele', 202, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_203.txt', '-time', '-ele', 203, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_204.txt', '-time', '-ele', 204, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_221.txt', '-time', '-ele', 221, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_222.txt', '-time', '-ele', 222, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_223.txt', '-time', '-ele', 223, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_ED_224.txt', '-time', '-ele', 224, "force")

    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_201.txt', '-time', '-ele', 201, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_202.txt', '-time', '-ele', 202, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_203.txt', '-time', '-ele', 203, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_204.txt', '-time', '-ele', 204, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_221.txt', '-time', '-ele', 221, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_222.txt', '-time', '-ele', 222, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_223.txt', '-time', '-ele', 223, "strainstress")
    ops.recorder('Element', '-file', f'MVLEM_F_EDSS_224.txt', '-time', '-ele', 224, "strainstress")
    # PT Tendons
    ops.recorder('Element', '-file', f'MVLEM_F_PT_301.txt', '-time', '-ele', 301, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_PT_302.txt', '-time', '-ele', 302, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_PT_321.txt', '-time', '-ele', 321, "force")
    ops.recorder('Element', '-file', f'MVLEM_F_PT_322.txt', '-time', '-ele', 322, "force")

    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(IDctrlNode1, 1000, 0, 0)
    ops.constraints('Penalty', 1e20, 1e20)
    ops.numberer('RCM')
    ops.system('BandGen')
    ops.test('NormDispIncr', 1e-3, 2000)
    ops.algorithm('KrylovNewton')
    ops.analysis('Static')

    RESULTS = defaultdict(lambda: [])

    # int(50), int(100), int(150), int(200), int(300), int(400), int(500), int(600), int(800), int(1000)

    for k in range(0, len(displacement)):
        disp = displacement[k]
        ops.integrator('DisplacementControl', IDctrlNode1, 1, 0.1)
        for i in range(0, disp):
            ops.analyze(1)
            save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(IDctrlNode1, 1)
            p = i / disp * 25
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
        ops.integrator('DisplacementControl', IDctrlNode1, 1, -0.1)
        for i in range(0, disp):
            ops.analyze(1)
            save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(IDctrlNode1, 1)
            p = i / disp * 25 + 25
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
        ops.integrator('DisplacementControl', IDctrlNode1, 1, -0.1)
        for i in range(0, disp):
            ops.analyze(1)
            save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(IDctrlNode1, 1)
            p = i / disp * 25 + 50
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")
        ops.integrator('DisplacementControl', IDctrlNode1, 1, 0.1)
        for i in range(0, disp):
            ops.analyze(1)
            save_result(RESULTS, numENT, element_length)
            f = ops.nodeDisp(IDctrlNode1, 1)
            p = i / disp * 25 + 75
            print("1Cycle", k + 1, "/", len(displacement), "Disp = ", '{:.2f}'.format(f), "mm", "------- Processing", '{:.2f}'.format(p), "%-------")

    print("Done Cyclic analysis")

    # print(RESULTS.keys())
    for key, value in RESULTS.items():
        RESULTS[key] = np.array(value)

    file_path = "Result.hdf5"
    with h5py.File(file_path, "w") as f:
        for key, value in RESULTS.items():
            f.create_dataset(key, data=value)

    file_path2 = "Result.mat"
    savemat(file_path2, RESULTS)
    print("Done Saving Data")
    os.chdir('..')


Run_DCRP1_cyclic(displacement=[int(50), int(100), int(150), int(200), int(300), int(400), int(500), int(600), int(800), int(1000)])
