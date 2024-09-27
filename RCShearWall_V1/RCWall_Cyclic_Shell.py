import math
import time

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
# import vfo.vfo as vfo
# import opsvis as opsv

from Units import *


def getCurrentNode():
    if len(ops.getNodeTags()) == 0:
        return 0
    else:
        return ops.getNodeTags()[-1]


def getCurrentElement():
    if len(ops.getEleTags()) == 0:
        return 0
    else:
        return ops.getEleTags()[-1]


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
    plt.plot(x_data, y_data, color='blue', linewidth=1.2, label='Numerical test')
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


def RebarArea(RebarDiametermm):
    a = math.pi * (RebarDiametermm / 2) ** 2  # compute area
    return a


# this function is used to create the MULTI-LAYERED-SHELL section
def defineWallSecion_ratios_LongAndTransv(secID, matConc, mreinfV, mreinfH, length, height, thick, cover, ratioLong, ratioTransv):
    # wall properties
    t = thick
    r = cover
    reinfMaterialV = mreinfV
    reinfMaterialH = mreinfH
    print('reinfMaterialV ', reinfMaterialV)
    print('reinfMaterialH ', reinfMaterialH)
    # vertical reinforcement
    totalSreqLong = length * t * ratioLong
    reqPerSideLong = totalSreqLong * 0.5
    layerThickV = reqPerSideLong / length
    layerThickV2 = layerThickV / 2
    print('layerThickV2 ', layerThickV2)

    # horizontal reinforcement
    totalSreqTransv = height * t * ratioTransv
    reqPerSideTransv = totalSreqTransv * 0.5
    layerThickH = reqPerSideTransv / height
    layerThickH2 = layerThickH / 2
    print('layerThickH2 ', layerThickH2)

    middleConcreteThickWall = t - r * 2 - layerThickV * 2 - layerThickH * 2
    conct = middleConcreteThickWall / 4

    # each row, starting from the second, is a layer
    # each row contains the thickness value and the reference to the corresponding material model
    ops.section('LayeredShell', secID, 14,
                matConc, r,  # conc cover top
                reinfMaterialV, layerThickV2,  # vertical Reinf layer top 1
                reinfMaterialV, layerThickV2,  # vertical Reinf layer top 2
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer top 1
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer top 2
                matConc, conct,  # cconcrete core layer 1
                matConc, conct,  # cconcrete core layer 2
                matConc, conct,  # cconcrete core layer 3
                matConc, conct,  # cconcrete core layer 4
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer bot 1
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer bot 2
                reinfMaterialV, layerThickV2,  # vertical Reinf layer bot 1
                reinfMaterialV, layerThickV2,  # vertical Reinf layer bot 2
                matConc, r)  # conc cover bottom
    print('LayeredShell', r+layerThickV2+layerThickV2+layerThickH2+layerThickH2+conct+conct+conct+conct+layerThickV2+layerThickV2+layerThickH2+layerThickH2+r)
    print('LayeredShell', secID, 14,
                matConc, r,  # conc cover top
                reinfMaterialV, layerThickV2,  # vertical Reinf layer top 1
                reinfMaterialV, layerThickV2,  # vertical Reinf layer top 2
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer top 1
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer top 2
                matConc, conct,  # cconcrete core layer 1
                matConc, conct,  # cconcrete core layer 2
                matConc, conct,  # cconcrete core layer 3
                matConc, conct,  # cconcrete core layer 4
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer bot 1
                reinfMaterialH, layerThickH2,  # horizontal Reinf layer bot 2
                reinfMaterialV, layerThickV2,  # vertical Reinf layer bot 1
                reinfMaterialV, layerThickV2,  # vertical Reinf layer bot 2
                matConc, r)  # conc cover bottom


def build_model(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadcoef, eleH=10, eleL=8, printProgression=True):
    ops.wipe()
    # ops.model('basic', '-ndm', 2, '-ndf', 3)  # Model of 2 dimensions, 3 dof per node
    ops.model('basic', '-ndm', 3, '-ndf', 6)
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
    # for i in range(1, eleH + 2):
    #     ops.node(i, 0, (i - 1) * (hw / eleH))
    # ops.fix(1, 1, 1, 1)  # Fixed condition at node 1

    # ----------------------------------------------------------------------------------------
    # Define Nodes (for SHELL)
    # ----------------------------------------------------------------------------------------
    # Number of elements in the vertical and horizontal direction
    vSpaces = eleH
    hSpaces = eleL  # must be even number so that there is a node in the middle
    discBE = 1  # number of elements to discretize each boundary element
    nNodes = (hSpaces + 1) * (vSpaces + 1)

    # number of shells horizontally to discretize the wall segment
    discWeb = hSpaces - eleBE

    # shell element horizontal dimensions
    vSpacing = hw / vSpaces
    hSpacingEB = lbe / discBE
    hSpacingWeb = lweb / discWeb

    nShellElements = hSpaces * vSpaces
    vLines = vSpaces + 1
    hLines = hSpaces + 1

    # array with all the required vertical coordinates
    vSpacing = np.ones(vSpaces + 1) * vSpacing
    vSpacing = np.concatenate((np.zeros(1), vSpacing), axis=0)

    # array with all the required vertical coordinates
    hSpacing = np.zeros(hLines)
    hSpacing[0:discBE] = np.ones(discBE) * hSpacingEB
    hSpacing[discBE:hSpaces - discBE] = np.ones(hSpaces - 2 * discBE) * hSpacingWeb
    hSpacing[hSpaces - discBE:hSpaces] = np.ones(discBE) * hSpacingEB
    hSpacing = np.concatenate((np.zeros(1), hSpacing), axis=0)
    # index of the top middle node

    posY = 0
    for i in range(vLines):
        posX = 0
        posY = posY + vSpacing[i]
        for j in range(hLines):
            nodeIndex = getCurrentNode() + 1
            posX = posX + hSpacing[j]
            ops.node(nodeIndex, posX, posY, 0)
            # print('NodeTag', nodeIndex, ': ', posX, ' , ', posY)
            # fix ground nodes
            if i == 0:
                # Fix supports at base of columns
                ops.fix(nodeIndex, 1, 1, 1, 1, 1, 1)  # tag, DX, DY, RZ
                # print('FixedNodeTag', nodeIndex, 1, 1, 1, 1, 1, 1)

    # ---------------------------------------------------------------------------------------
    # Define Control Node and DOF
    # ---------------------------------------------------------------------------------------
    global IDctrlNode, IDctrlDOF
    # controlNode = eleH + 1  # Control Node (TopNode)
    IDctrlNode = int((vLines - 1) * hLines + 1 + hSpaces / 2)
    IDctrlDOF = 1  # Control DOF 1 = X-direction
    print('controlNode = ', IDctrlNode)

    # ---------------------------------------------------------------------------------------
    # Define Axial Load on Top Node
    # ---------------------------------------------------------------------------------------
    global Aload  # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    Aload = 0.85 * abs(fc) * tw * lw * loadcoef
    print('Axial load (kN) = ', Aload / 1000)

    # ---------------------------------------------------------------------------------------
    # Define Steel uni-axial materials
    # ---------------------------------------------------------------------------------------
    sYb = 1
    sYw = 2
    sX = 3

    # STEEL Y BE (boundary element)
    fyYbp = fyb  # fy - tension
    fyYbn = fyb  # fy - compression
    bybp = 0.0185  # strain hardening - tension
    bybn = 0.02  # strain hardening - compression

    # STEEL Y WEB
    fyYwp = fyw  # fy - tension
    fyYwn = fyw  # fy - compression
    bywp = 0.025  # strain hardening - tension
    bywn = 0.02  # strain hardening - compression

    # STEEL X
    fyXp = fyw  # fy - tension
    fyXn = fyw  # fy - compression
    bXp = 0.025  # strain hardening - tension
    bXn = 0.02  # strain hardening - compression

    # STEEL misc
    Es = 200 * GPa  # Young's modulus
    Bs = 0.01  # strain-hardening ratio
    R0 = 20.0  # initial value of curvature parameter
    cR1 = 0.925  # control the transition from elastic to plastic branches
    cR2 = 0.15  # control the transition from elastic to plastic branches

    # SteelMPF model
    ops.uniaxialMaterial('Steel02', sYb, fyYbp, Es, Bs, R0, cR1, cR2)
    ops.uniaxialMaterial('Steel02', sYw, fyYwp, Es, Bs, R0, cR1, cR2)
    ops.uniaxialMaterial('Steel02', sX, fyXp, Es, Bs, R0, cR1, cR2)
    # SteelMPF model
    # ops.uniaxialMaterial('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    # ops.uniaxialMaterial('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    # ops.uniaxialMaterial('SteelMPF', sX, fyXp, fyYwn, Es, bXp, bXn, R0, cR1, cR2)  # Steel X
    print('--------------------------------------------------------------------------------------------------')
    print('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    print('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    print('SteelMPF', sX, fyYwp, fyYwn, Es, bXp, bXn, R0, cR1, cR2)  # Steel X

    # ---------------------------------------------------------------------------------------
    # Define "ConcreteCM" uni-axial materials
    # ---------------------------------------------------------------------------------------
    concWeb = 4
    concBE = 5

    # ----- unconfined concrete for WEB
    fc0 = abs(fc) * MPa  # Initial concrete strength
    Ec0 = 8200.0 * (fc0 ** 0.375)  # Initial elastic modulus
    fcU = -fc0 * MPa  # Unconfined concrete strength
    ecU = -(fc0 ** 0.25) / 1152.7  # Unconfined concrete strain
    EcU = Ec0  # Unconfined elastic modulus
    ftU = 0.5 * (fc0 ** 0.5)  # Unconfined tensile strength
    etU = 2.0 * ftU / EcU  # Unconfined tensile strain
    xpU = 2.0
    xnU = 2.3
    rU = -1.9 + (fc0 / 5.2)

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
    et = 0.00008463  # strain at peak tensile stress (0.00008)
    rt = 1.2  # shape parameter - tension
    xcrp = 100000  # cracking strain - tension

    ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
    ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 1)  # BE (confined concrete)
    print('--------------------------------------------------------------------------------------------------')
    print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 1)  # Web (unconfined concrete)
    print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 1)  # BE (confined concrete)
    print('--------------------------------------------------------------------------------------------------')

    # -------------------------- Concrete7 model --------------------------
    # ops.uniaxialMaterial('Concrete07', concWeb, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)  # Web (unconfined concrete)
    # ops.uniaxialMaterial('Concrete07', concBE, fcC, ecC, EcC, ftC, etC, xpC, xnC, rC)  # BE (confined concrete)
    # print('--------------------------------------------------------------------------------------------------')
    # print('Concrete07', concWeb, fcU, ecU, EcU, ftU, etU, xpU, xnU, rU)  # Web (unconfined concrete)
    # print('Concrete07', concBE, fcC, ecC, EcC, ftC, etC, xpC, xnC, rC)  # BE (confined concrete)

    # ----------------------------Shear spring for MVLEM-------------------------------
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    Kshear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    ops.uniaxialMaterial('Elastic', 6, Kshear)

    # ---- Steel in Y direction (BE + Web) -------------------------------------------
    print('rouYb =', rouYb)
    print('rouYw =', rouYw)

    # ---- Steel in X direction (BE + Web) -------------------------------------------
    rouXb = 0.0105450
    rouXw = 0.0030169
    print('rouXb =', rouXb)
    print('rouXw =', rouXw)

    # ---------------------------------------------------------------------------------------
    # Define FSAM nDMaterial
    # ---------------------------------------------------------------------------------------
    matBE = 7
    matWeb = 8

    # FSAM model
    ops.nDMaterial('FSAM', matBE, 0.0, sX, sYb, concBE, rouXb, rouYb, 1.2, 0.012)  # Boundary (confined concrete)
    ops.nDMaterial('FSAM', matWeb, 0.0, sX, sYw, concWeb, rouXw, rouYw, 1.2, 0.012)  # Web (unconfined concrete)
    print('--------------------------------------------------------------------------------------------------')
    print('FSAM', matBE, 0.0, sX, sYb, concBE, rouXb, rouYb, 0.2, 0.012)  # Boundary (confined concrete)
    print('FSAM', matWeb, 0.0, sX, sYw, concWeb, rouXw, rouYw, 0.2, 0.012)  # Web (unconfined concrete)
    print('--------------------------------------------------------------------------------------------------')

    # --------------------------------------------------------------------------------
    #  Define 'material' for SHELL
    # --------------------------------------------------------------------------------
    # Define PSUMAT and convert it to plane stress material
    # IDPlaneStress = 9
    # IDPlateStress = 10
    # fcu = 0.2 * fc0
    # ecr = 5 * ecU  # strain at ultimate stress
    # srf = 0.3
#
    # # out of plane behaviour incorporated to the plane stress material
    # ops.nDMaterial('PlaneStressUserMaterial', IDPlaneStress, 40, 7, fc0, ftU, fcu, ecU, ecr, etU, srf)
    # ops.nDMaterial('PlateFromPlaneStress', IDPlateStress, IDPlaneStress, 12500)
#
    # # Convert rebar material to plane stress/plate rebar
    # # angle=90 longitudinal reinforced steel
    # # angle=0 transverse reinforced steel
    # IDRebarYb = 11
    # IDRebarYw = 12
    # IDRebarX = 13
    # ops.nDMaterial('PlateRebar', IDRebarYb, sYb, 90)  # vertical
    # ops.nDMaterial('PlateRebar', IDRebarYw, sYw, 90)  # vertical
    # ops.nDMaterial('PlateRebar', IDRebarX, sX, 0)  # horizontal

    # shell sections
    sectionID_web = 1
    sectionID_BE = 2

    cover = 12
    ShellType = "ShellNLDKGQ"

    # NON LINEAR CONCRETE MATERIAL MODEL
    # the fracture strength ft is 10% of fc
    ft = 0.10 * fc
    # the crushing strength fcu is 20% of fc
    fcu = -0.20 * fc
    # the strain at maxium compressive strength eco is -0.002
    eco = -0.002
    # the strain at the crushing strength
    ecu = -0.01
    # the ultimate tensile strain is etu is 0.001
    etu = 0.001
    # shear retention factor is 0.3
    srf = 0.3

    ops.nDMaterial('PlaneStressUserMaterial', 1, 40, 7, fc, ft, fcu, eco, ecu, etu, srf)

    # figure, ax = plt.subplots(2, 3)
    # out of plane behaviour incorporated to the plane stress material
    ops.nDMaterial('PlateFromPlaneStress', 14, 1, 12830)

    # NON LINEAR STEEL
    # elastic moduli for common reinforcement steel
    steelElasticMod = 202.7e9
    # strain hardening ratio for renforcement steel
    strainHardeningRatio = 0.01
    # yield strength is a user parameter
    fy = fyb
    ops.uniaxialMaterial('Steel02', 8, fy, steelElasticMod, strainHardeningRatio, 20.0, 0.925, 0.15)

    # Convert rebar material to plane stress/plate rebar
    # angle=90 longitudinal reinforced steel
    # angle=0 transverse reinforced steel
    ops.nDMaterial('PlateRebar', 10, 8, 90.0)  # vertical
    ops.nDMaterial('PlateRebar', 11, 8, 0.0)  # horizontal
    # Define LayeredShell sections
    # shell with smeared rebar layer in both directions
    # this section is used for the WEB
    defineWallSecion_ratios_LongAndTransv(sectionID_web,  # sectionID,
                                          14, 10, 11,  # concID, #steelReinfLongID(vert), #steelReinfTransvID(horz)
                                          lw,  # length (of the wall)
                                          hw,  # height
                                          tw,  # thick
                                          cover,  # cover
                                          rouYw,  # long reinf ratio (minimum from code)
                                          rouXw)  # transv reinf ratio (minimum from code)

    #  shell with smeared rebar layer only in the horizontal direction (vertical rebar is defined with truss elements)
    # this section is used for the Boundary Elements
    defineWallSecion_ratios_LongAndTransv(sectionID_BE,  # sectionID,
                                          14, 10, 11,  # concID, #steelReinfLongID(vert), #steelReinfTransvID(horz)
                                          lbe,  # length (of the special boundary element)
                                          hw,  # height
                                          tw,  # thick
                                          cover,  # cover
                                          rouYb,  # long reinf ratio
                                          rouXb)  # transv reinf ratio

    shellSections = np.zeros(hSpaces)
    shellSections[0:discBE] = np.ones(discBE) * sectionID_BE
    shellSections[discBE:hSpaces - discBE] = np.ones(hSpaces - 2 * discBE) * sectionID_web
    shellSections[hSpaces - discBE:hSpaces] = np.ones(discBE) * sectionID_BE

    # create the elements
    # the numbering starts from the bottom left corner and goes up
    # this numbering facilitates creationg of the border elements
    for k in range(hSpaces):
        section = shellSections[k]
        for i in range(vSpaces):
            eIndex = getCurrentElement() + 1
            n1 = 1 + hLines * (i) + k
            n2 = 1 + hLines * (i) + k + 1
            n3 = 1 + hLines * (i + 1) + k + 1
            n4 = 1 + hLines * (i + 1) + k
            ops.element(ShellType, eIndex, n1, n2, n3, n4, int(section))
            print(ShellType, eIndex, n1, n2, n3, n4, int(section))

    # BEAM IN THE TOP TO STABILIZE DEFORMATION IN TOP NODES
    a = tw * 10
    b = tw * 10
    E = 35e9
    Iz = ((a * b * b * b) / 12)
    Iy = ((a * b * b * b) / 12)
    Jxx = 0.141 * a * b * b * b
    G = E / (2 * (1 + 0.2))
    A = a * b
    ops.geomTransf('Linear', 1, 0, 1, 0)
    nodei = midNode = (hSpaces + 1) * (vSpaces) + 1
    for j in range(hSpaces):
        ops.element('elasticBeamColumn', getCurrentElement() + 1, nodei + j, nodei + j + 1, A, E, G, Jxx, Iy, Iz, 1)
    # --------------------------------------------------------------------------------
    #  Define 'MVLEM' elements
    # --------------------------------------------------------------------------------
    # Set 'MVLEM' parameters thick, width, rho, matConcrete, matSteel
    MVLEM_thick = [tw] * n
    MVLEM_width = [lbe if i in (0, n - 1) else elelweb for i in range(n)]
    MVLEM_rho = [rouYb if i in (0, n - 1) else rouYw for i in range(n)]
    MVLEM_matConcrete = [concBE if i in (0, n - 1) else concWeb for i in range(n)]
    MVLEM_matSteel = [sYb if i in (0, n - 1) else sYw for i in range(n)]

    MVLEM_mat = [matBE if i in (0, n - 1) else matWeb for i in range(n)]

    # for i in range(eleH):
    # ------------------ MVLEM ----------------------------------------------
    # ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 6)
    # print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', 6)

    # ---------------- SFI_MVLEM -------------------------------------------
    # ops.element('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
    # print('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

    # ops.element('E_SFI', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
    # print('E_SFI', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

    # ---------------- SFI_MVLEM_3D -------------------------------------------
    # print('SFI_MVLEM_3D', i + 1, *[i + 2, i + 1, i + 41, i + 42], eleL, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
    # ops.element('SFI_MVLEM_3D', i + 1, *[i + 2, i + 1, i + 41, i + 42], eleL, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

    parameter_values = [tw, hw, lw, lbe, fc, fyb, fyw, round(rouYb, 4), round(rouYw, 4), loadcoef]

    # ---------------------------------------------------------------------------------------
    # Plot the model
    # ---------------------------------------------------------------------------------------
    vfo.plot_model(show_nodetags="yes")

    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print("\033[92mModel Built Successfully --> Using the following parameters :", parameter_values, "\033[0m")
        print('--------------------------------------------------------------------------------------------------')


def run_gravity(steps=10, printProgression=True):
    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")
    ops.timeSeries('Linear', 1, '-factor', 1.0)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 1, 1)
    # ops.load(controlNode, *[0.0, -Aload, 0.0])  # apply vertical load
    ops.load(IDctrlNode, *[0.0, -Aload, 0.0, 0.0, 0.0, 0.0])  # apply vertical load
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


def run_cyclic(DisplacementStep, plotResults=True, printProgression=True, recordData=False):
    if printProgression:
        tic = time.time()
        print("RUNNING CYCLIC ANALYSIS")
    if recordData:
        print("RECORDING SHEAR LOAD VS DISPLACEMENT DATA")
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Reaction.out', '-closeOnWrite', '-node', 1, '-dof', 1, 'reaction')
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Cyclic_Displacement.out', '-closeOnWrite', '-node', IDctrlNode, '-dof', 1, 'disp')
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    # ops.load(controlNode, *[1.0, 0.0, 0.0])  # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.load(IDctrlNode, *[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('RCM')
    ops.system("BandGeneral")
    ops.test('NormDispIncr', 1.0e-8, 1000, 0)
    ops.algorithm('Newton')

    # Define analysis parameters
    maxUnconvergedSteps = 1
    unconvergeSteps = 0
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
        if unconvergeSteps > maxUnconvergedSteps:
            break
        ops.integrator("DisplacementControl", IDctrlNode, 1, Dincr)
        ops.analysis('Static')
        ok = ops.analyze(1)
        if ok != 0:
            # If not converged, reduce the increment
            unconvergeSteps += 1
            ts = 20
            smallDincr = Dincr / ts  # Try 20x smaller increments
            for k in range(1, ts):
                print(f'Small Step {k} -------->', f'smallDincr = ', smallDincr)
                ops.integrator("DisplacementControl", IDctrlNode, 1, smallDincr)
                ops.analysis('Static')
                ok = ops.analyze(1)

        D0 = D1  # move to next step
        finishedSteps = j + 1
        disp = ops.nodeDisp(IDctrlNode, 1)
        baseShear = -ops.getLoadFactor(2) / 1000  # Convert to from N to kN
        dispData[j + 1] = disp
        baseShearData[j + 1] = baseShear

        if printProgression:
            print(f'\033[92m InputDisplacement {j} = {DisplacementStep[j]}\033[0m')
            print(f'\033[91mOutputDisplacement {j} = {dispData[j + 1]}\033[0m')

    if printProgression:
        toc = time.time()
        print('CYCLIC ANALYSIS DONE IN {:1.2f} seconds'.format(toc - tic))

    # if plotResults:
    #     force_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Reaction.out')
    #     displacement_data = np.loadtxt('RunTimeNodalResults/Cyclic_Horizontal_Displacement.out')
    #     # Plot Force vs. Displacement
    #     plt.figure(figsize=(7, 6), dpi=100)
    #     # plt.plot(displacement_data, -force_data, color='blue', linewidth=1.2, label='Numerical test')
    #     plt.plot(dispData, -baseShearData, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
    #     # plt.plot([row[0] for row in dataCyc[:finishedSteps]], [-row[1] for row in dataCyc[:finishedSteps]], color="black", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
    #     # plt.plot(DisplacementStep, [-row[1] for row in dataCyc[:finishedSteps]], color="blue", linestyle="-", linewidth=1.2, label='Input Displacement vs Shear Load')
    #     plt.axhline(0, color='black', linewidth=0.4)
    #     plt.axvline(0, color='black', linewidth=0.4)
    #     plt.grid(linestyle='dotted')
    #     font_settings = {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14}
    #     plt.xlabel('Displacement (mm)', fontdict=font_settings)
    #     plt.ylabel('Base Shear (kN)', fontdict=font_settings)
    #     plt.yticks(fontname='Cambria', fontsize=14)
    #     plt.xticks(fontname='Cambria', fontsize=14)
    #     plt.title(f'Specimen', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()

    return [dispData[0:finishedSteps], -baseShearData[0:finishedSteps]]


def run_pushover(MaxDisp=75, DispIncr=1, plotResults=True, printProgression=True, recordData=False):
    DispIncr = MaxDisp / 500
    if printProgression:
        tic = time.time()
        print("RUNNING PUSHOVER ANALYSIS")

    if recordData:
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Reaction.out', '-node', 1, '-dof', 1, 'reaction')
        ops.recorder('Node', '-file', 'RunTimeNodalResults/Pushover_Displacement.out', '-node', IDctrlNode, '-dof', 1, 'disp')

    ops.timeSeries('Linear', 3)  # create TimeSeries for gravity analysis
    ops.pattern('Plain', 3, 3)
    ops.load(IDctrlNode, *[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Apply a unit reference load in DOF=1 (nd    FX  FY  MZ)

    NstepsPush = int(MaxDisp / DispIncr)

    if printProgression:
        print("Starting pushover analysis...")
        print("   total steps: ", NstepsPush)
    ops.constraints('Transformation')
    ops.numberer("RCM")
    ops.system("BandGen")
    ops.test('NormDispIncr', 1e-5, 100, 0)
    ops.algorithm('NewtonLineSearch')
    ops.integrator("DisplacementControl", IDctrlNode, 1, DispIncr)  # Target node is controlNode and dof is 1
    ops.analysis("Static")

    maxUnconvergedSteps = 10
    unconvergeSteps = 0
    finishedSteps = 0
    dataPush = np.zeros((NstepsPush + 1, 2))
    DispImpo = np.zeros(NstepsPush + 1)
    # Perform pushover analysis
    for j in range(NstepsPush):
        if unconvergeSteps > maxUnconvergedSteps:
            break
        ok = ops.analyze(1)
        if ok < 0:
            unconvergeSteps = unconvergeSteps + 1

        DispImpo += DispIncr
        finishedSteps = j
        disp = ops.nodeDisp(IDctrlNode, 1)
        baseShear = -ops.getLoadFactor(3) / 1000  # Convert to from N to kN
        dataPush[j + 1, 0] = disp
        dataPush[j + 1, 1] = baseShear

        if printProgression:
            print("step", j + 1, "/", NstepsPush, "   ", "Impos disp = ", round(DispImpo[j], 2), "---->  Real disp = ", str(round(disp, 2)))

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
