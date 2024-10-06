import math
import time
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
# import vfo.vfo as vfo
# import opsvis as opsv
from functions import *


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


def build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, rouXb, rouXw, loadF, eleH=10, eleL=8, printProgression=True):
    """
    Builds a model based on the provided parameters.

    Args:
        tw: Thickness of the wall in web
        tb: Thickness of the wall in flange
        hw: Height of the wall
        lw: Length of the wall
        lbe: Length of the flange
        fc: Concrete Compressive Strength
        fyb: Yield strength of the reinforcement in flange
        fyw: Yield strength of the reinforcement in web
        rouYb: Density of the longitudinal reinforcement in flange
        rouYw: Density of the longitudinal reinforcement in web
        rouXb: Density of the transversal reinforcement in flange
        rouXw: Density of the transversal reinforcement in web
        loadF: Axial Load coefficient
        eleH: Number of elements in height (default: 10)
        eleL: Number of elements in length (default: 6)
    """

    ops.wipe()
    ops.model('Basic', '-ndm', 2, '-ndf', 3)

    # ----------------------------------------------------------------------------------------
    # Set geometry, ops.nodes, boundary conditions
    # ----------------------------------------------------------------------------------------
    lweb = lw - (2 * lbe)  # Length of the Web
    eleBE = 2
    eleWeb = eleL - eleBE
    elelweb = lweb / eleWeb

    # ----------------------------------------------------------------------------------------
    # Define Nodes (for MVLEM)
    # ----------------------------------------------------------------------------------------
    for i in range(1, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))
        # print(i, 0, (i - 1) * (hw / eleH))

    ops.fix(1, 1, 1, 1)  # Fixed condition at node 1

    # ---------------------------------------------------------------------------------------
    # Define Control Node and DOF
    # ---------------------------------------------------------------------------------------
    global controlNode, controlNodeDof
    controlNode = eleH + 1  # Control Node (TopNode)
    controlNodeDof = 1  # Control DOF 1 = X-direction

    # ---------------------------------------------------------------------------------------
    # Define Axial Load on Top Node
    # ---------------------------------------------------------------------------------------
    global Aload  # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    Aload = 0.85 * abs(fc) * tw * lw * loadF

    # ---------------------------------------------------------------------------------------
    # Define Steel uni-axial materials
    # ---------------------------------------------------------------------------------------
    sYb = 1
    sYw = 2
    sX = 3

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

    # SteelMPF model
    # ops.uniaxialMaterial('Steel02', sYb, fyYbp, Es, Bs, R0, cR1, cR2)
    # ops.uniaxialMaterial('Steel02', sYw, fyYwp, Es, Bs, R0, cR1, cR2)
    # ops.uniaxialMaterial('Steel02', sX, fyXp, Es, Bs, R0, cR1, cR2)
    # SteelMPF model
    ops.uniaxialMaterial('SteelMPF', sYb, fyYbp, fyYbn, Es, bybp, bybn, R0, cR1, cR2)  # Steel Y boundary
    ops.uniaxialMaterial('SteelMPF', sYw, fyYwp, fyYwn, Es, bywp, bywn, R0, cR1, cR2)  # Steel Y web
    ops.uniaxialMaterial('SteelMPF', sX, fyXp, fyXn, Es, bXp, bXn, R0, cR1, cR2)  # Steel X
    if printProgression:
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
    # ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    # ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    ops.uniaxialMaterial('ConcreteCM', concWeb, fcU, ecU, EcU, rU, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    ops.uniaxialMaterial('ConcreteCM', concBE, fcC, ecC, EcC, rC, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print('ConcreteCM', concWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
        print('ConcreteCM', concBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)

    # ----------------------------Shear spring for MVLEM-------------------------------------
    Ac = lw * tw  # Concrete Wall Area
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    kShear = Ac * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    matSpring = 6
    ops.uniaxialMaterial('Elastic', matSpring, kShear)

    # ---- Steel in Y -------------------------------------------
    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print('rouYb =', rouYb)
        print('rouYw =', rouYw)

    # ---- Steel in X -------------------------------------------
    if printProgression:
        print('rouXb =', rouXb)
        print('rouXw =', rouXw)

    # ---------------------------------------------------------------------------------------
    # Define FSAM nDMaterial
    # ---------------------------------------------------------------------------------------
    matBE = 7
    matWeb = 8

    # FSAM model
    nu = 0.2  # friction coefficient
    alfadow = 0.012  # dowel action stiffness parameter
    ops.nDMaterial('FSAM', matBE, 0.0, sX, sYb, concBE, rouXb, rouYb, nu, alfadow)  # Boundary (confined concrete)
    ops.nDMaterial('FSAM', matWeb, 0.0, sX, sYw, concWeb, rouXw, rouYw, nu, alfadow)  # Web (unconfined concrete)
    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print('FSAM', matBE, 0.0, sX, sYb, concBE, rouXb, rouYb, nu, alfadow)  # Boundary (confined concrete)
        print('FSAM', matWeb, 0.0, sX, sYw, concWeb, rouXw, rouYw, nu, alfadow)  # Web (unconfined concrete)
        print('--------------------------------------------------------------------------------------------------')

    # --------------------------------------------------------------------------------
    #  Define 'MVLEM' elements
    # --------------------------------------------------------------------------------
    # Set 'MVLEM' parameters thick, width, rho, matConcrete, matSteel
    MVLEM_thick = [tb if i in (0, eleL - 1) else tw for i in range(eleL)]
    MVLEM_width = [lbe if i in (0, eleL - 1) else elelweb for i in range(eleL)]
    MVLEM_mat = [matBE if i in (0, eleL - 1) else matWeb for i in range(eleL)]
    # MVLEM_rho = [rouYb if i in (0, eleL - 1) else rouYw for i in range(eleL)]
    # MVLEM_matConcrete = [concBE if i in (0, eleL - 1) else concWeb for i in range(eleL)]
    # MVLEM_matSteel = [sYb if i in (0, eleL - 1) else sYw for i in range(eleL)]

    for i in range(eleH):
        # ------------------ MVLEM ----------------------------------------------
        # ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', matSpring)
        # print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', matSpring)

        # ---------------- SFI_MVLEM -------------------------------------------
        ops.element('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
        if printProgression:
            print('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

        # ---------------- E_SFI -----------------------------------------------
        # ops.element('E_SFI', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
        # if printProgression:
        #     print('E_SFI', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

    parameter_values = [tw, tb, hw, lw, lbe, fc, fyb, fyw, round(rouYb, 4), round(rouYw, 4), round(rouXb, 4), round(rouXw, 4), loadF]
    '''
    ops.recorder('Node', '-file', f'plot/MVLEM_Dx.txt', '-time', '-nodeRange', 1, eleH, '-dof', 1, 'disp')
    for i in range(0, eleH):
        for j in range(0, eleL):
            # Unaxial Steel Recorders for all panels
            # ops.recorder('Element', '-file', f'MVLEM_strain_stress_steelX_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelX')
            # ops.recorder('Element', '-file', f'MVLEM_strain_stress_steelY_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelY')
            # print('recorder', 'Element', '-file', f'plot/MVLEM_strain_stress_steelX_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelX')
            # print('recorder', 'Element', '-file', f'plot/MVLEM_strain_stress_steelY_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelY')

            # Unaxial Concrete Recorders for all panels
            ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_concr1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete1')
            ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_concr2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete2')
            # print('recorder', 'Element', '-file', f'plot/MVLEM_strain_stress_concr1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete1')
            # print('recorder', 'Element', '-file', f'plot/MVLEM_strain_stress_concr2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete2')

            # Shear Concrete Recorders for all panels
            ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_inter1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock1')
            ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_inter2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock2')
            # print('recorder', 'Element', '-file', f'MVLEM_strain_stress_inter1_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock1')
            # print('recorder', 'Element', '-file', f'MVLEM_strain_stress_inter2_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock2')
            ops.recorder('Element', '-file', f'plot/MVLEM_cracking_angle_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'cracking_angles')
            # print('recorder', 'Element', '-file', f'plot/MVLEM_cracking_angle_ele_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'cracking_angles')
            ops.recorder('Element', '-file', f'plot/MVLEM_panel_crack_{i + 1}_panel_{j + 1}.txt', '-time', '-ele', i + 1, 'RCPanel', j + 1, 'panel_crack')
    '''

    # ops.recorder('Node', '-file', f'MVLEM_Dtop.out.txt', '-time', '-node', controlNode, '-dof', 1, 'disp')
    #
    # for i in range(0, eleH):
    #     # Unaxial Steel Recorders for all panels
    #     ops.recorder('Element', '-file', f'MVLEM_Fgl.out_{i + 1}.txt', '-time', '-ele', i + 1, '-dof', 1, 'globalForce')
    #     ops.recorder('Element', '-file', f'MVLEM_Dsh.out_{i + 1}.txt', '-time', '-ele', i + 1, 'ShearDef')

    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print("\033[92mModel Built Successfully --> Using the following parameters :", parameter_values, "\033[0m")
        print('--------------------------------------------------------------------------------------------------')


def run_gravity(steps=10, printProgression=True):
    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")
    ops.timeSeries('Linear', 1, '-factor', 1.0)
    ops.pattern('Plain', 1, 1)
    ops.load(controlNode, *[0.0, -Aload, 0.0])  # apply vertical load
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


def run_analysis(DisplacementStep, pushover=False, printProgression=True):
    if printProgression:
        if pushover:
            print("RUNNING PUSHOVER ANALYSIS")
        else:
            print("RUNNING CYCLIC ANALYSIS")

    if pushover:
        MaxDisp = max(DisplacementStep)
        dispIncr = (max(DisplacementStep) / len(DisplacementStep))
        DisplacementStep = [dispIncr * i for i in range(int(MaxDisp / dispIncr))]

    # define parameters for adaptive time-step
    max_factor = 0.12  # 1.0 -> don't make it larger than initial time step
    min_factor = 1e-06  # at most initial/1e6
    max_factor_increment = 1.5  # define how fast the factor can increase
    min_factor_increment = 1e-06  # define how fast the factor can decrease
    max_iter = 2500
    desired_iter = int(max_iter / 2)  # should be higher than the desired number of iterations

    # -------------CYCLIC-----------------
    ops.timeSeries('Linear', 2, '-factor', 1.0)
    ops.pattern('Plain', 2, 2)
    RefLoad = 1000e3
    ops.load(controlNode, *[RefLoad, 0.0, 0.0])
    ops.constraints('Transformation')  # Transformation 'Penalty', 1e20, 1e20
    ops.numberer('RCM')
    ops.system("ProfileSPD")
    ops.test('NormDispIncr', 1e-6, desired_iter, 0)
    ops.algorithm('KrylovNewton')  # KrylovNewton
    ops.analysis("Static")

    Nsteps = len(DisplacementStep)
    finishedSteps = 0
    dispData = np.zeros(Nsteps + 1)
    loadData = np.zeros(Nsteps + 1)
    forceData = np.zeros(Nsteps + 1)
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
            # are we done with this cycle?
            if abs(dU_cumulative - Dincr) <= dU_tolerance:
                if printProgression:
                    print("Target displacement has been reached. Current Dincr = {:.3g}".format(dU_cumulative))
                increment_done = True
                break
            # adapt the current displacement increment
            dU_adapt = dU * factor
            if abs(dU_cumulative + dU_adapt) > (abs(Dincr) - dU_tolerance):
                dU_adapt = Dincr - dU_cumulative
            # update integrator
            ops.integrator("DisplacementControl", controlNode, 1, dU_adapt)
            ok = ops.analyze(1)
            # adapt if necessary
            if ok == 0:
                num_iter = ops.testIter()
                norms = ops.testNorms()
                error_norm = norms[num_iter - 1] if num_iter > 0 else 0.0
                # update adaptive factor (increase)
                factor_increment = min(max_factor_increment, desired_iter / num_iter)
                factor *= factor_increment
                if factor > max_factor:
                    factor = max_factor
                if factor > old_factor:
                    if printProgression:
                        print("Increasing increment factor due to faster convergence. Factor = {:.3g}".format(factor))
                old_factor = factor
                dU_cumulative += dU_adapt
            else:
                num_iter = max_iter
                factor_increment = max(min_factor_increment, desired_iter / num_iter)
                factor *= factor_increment
                if printProgression:
                    print("Reducing increment factor due to non convergence. Factor = {:.3g}".format(factor))
                if factor < min_factor:
                    if printProgression:
                        print("ERROR: current factor is less then the minimum allowed ({:.3g} < {:.3g})".format(factor, min_factor))
                        print("ERROR: the analysis did not converge")
                    break
        if not increment_done:
            break
        else:
            D0 = D1  # move to next step

        finishedSteps = j + 1

        disp = ops.nodeDisp(controlNode, 1)
        baseLoad = ops.getLoadFactor(2) / 1000 * RefLoad  # Convert to from N to kN
        dispData[j + 1] = disp
        loadData[j + 1] = baseLoad

        # eleForce = ops.eleForce(1, 4) / 1000
        # forceData[j + 1] = eleForce

    return [dispData[0:finishedSteps], loadData[0:finishedSteps]]
