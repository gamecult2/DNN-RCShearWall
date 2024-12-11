import time
import openseespy.opensees as ops
# import vfo.vfo as vfo
# import opsvis as opsv
from functions import *


def reset_analysis():
    ops.setTime(0.0)  # Set the time in the Domain to zero
    ops.loadConst()  # Set the loads constant in the domain
    ops.remove('recorders')  # Remove all recorder objects.
    ops.wipeAnalysis()  # destroy all components of the Analysis object
    ops.wipe()


def build_model(tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, eleH=14, eleL=12, printProgression=True):
    ops.wipe()
    ops.model('Basic', '-ndm', 2, '-ndf', 3)

    # Geometry and material properties
    eleBE = 2*2

    lweb = lw - (2 * lbe)  # Web length

    eleWeb = eleL - eleBE
    elelweb = lweb / eleWeb
    Ag = tw * lweb + 2 * (tb * lbe)  # Wall area

    # Node definition
    for i in range(1, eleH + 2):
        ops.node(i, 0, (i - 1) * (hw / eleH))

    ops.fix(1, 1, 1, 1)  # Fixed condition at base node 1

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

    # ---------------------------------------------------------------------------------------
    # Define Steel uni-axial materials
    IDsYb, IDsYw, IDsX = 1, 2, 3  # Steel ID
    Es = 200 * GPa  # Young's modulus
    R0, cR1, cR2 = 20.0, 0.925, 0.0015  # Strain-hardening and curve parameters
    byb = 0.01  # STEEL Y BE - strain hardening - tension & compression
    byw = 0.02  # STEEL Y WEB - strain hardening - tension & compression
    bX = 0.02   # STEEL X - strain hardening - tension & compression

    # SteelMPF model
    ops.uniaxialMaterial('SteelMPF', IDsYb, fyb, fyb, Es, byb, byb, R0, cR1, cR2)  # Steel Y boundary
    ops.uniaxialMaterial('SteelMPF', IDsYw, fyw, fyw, Es, byw, byw, R0, cR1, cR2)  # Steel Y web
    ops.uniaxialMaterial('SteelMPF', IDsX, fx, fx, Es, bX, bX, R0, cR1, cR2)  # Steel X boundary
    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print('SteelMPF', IDsYb, fyb, fyb, Es, byb, byb, R0, cR1, cR2)  # Steel Y boundary
        print('SteelMPF', IDsYw, fyw, fyw, Es, byw, byw, R0, cR1, cR2)  # Steel Y web
        print('SteelMPF', IDsX, fx, fx, Es, bX, bX, R0, cR1, cR2) # Steel X

    # Define "ConcreteCM" uni-axial materials
    IDconcWeb, IDconcBE = 4, 5  # Concrete ID

    # unconfined concrete for WEB
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
    # confined concrete for BE
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

    ru = 7.3049  # shape parameter - compression
    xcrnu = 1.0125  # cracking strain - compression
    rc = 1.5  # shape parameter - compression
    xcrnc = 1.039  # cracking strain - compression
    et = 0.0001236  # strain at peak tensile stress (0.00008)
    rt = 1.5  # shape parameter - tension
    xcrp = 1000  # cracking strain - tension

    # ConcreteCM model
    # ops.uniaxialMaterial('ConcreteCM', IDconcWeb, fcU, ecU, EcU, ru, xcrnu, ftU, et, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    # ops.uniaxialMaterial('ConcreteCM', IDconcBE, fcC, ecC, EcC, rc, xcrnc, ftC, et, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    ops.uniaxialMaterial('ConcreteCM', IDconcWeb, fcU, ecU, EcU, rU, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
    ops.uniaxialMaterial('ConcreteCM', IDconcBE, fcC, ecC, EcC, rC, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)
    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print('ConcreteCM', IDconcWeb, fcU, ecU, EcU, rU, xcrnu, ftU, etU, rt, xcrp, '-GapClose', 0)  # Web (unconfined concrete)
        print('ConcreteCM', IDconcBE, fcC, ecC, EcC, rC, xcrnc, ftC, etC, rt, xcrp, '-GapClose', 0)  # BE (confined concrete)

    # Shear Model for Section Aggregator to assign for MVLEM element shear spring
    IDmatSpring = 6
    Gc = Ec0 / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    kShear = Ag * Gc * (5 / 6)  # Shear stiffness k * A * G ---> k=5/6
    ops.uniaxialMaterial('Elastic', IDmatSpring, kShear)

    if printProgression:
        #  Steel in Y
        print('--------------------------------------------------------------------------------------------------')
        print('rouYb =', rouYb)
        print('rouYw =', rouYw)
        # Steel in X
        print('rouXb =', rouXb)
        print('rouXw =', rouXw)

    # Define FSAM nDMaterial model
    IDmatBE, IDmatWeb = 7, 8
    nu = 0.2  # friction coefficient
    alfadow = 0.012  # dowel action stiffness parameter

    ops.nDMaterial('FSAM', IDmatBE, 0.0, IDsX, IDsYb, IDconcBE, rouXb, rouYb, nu, alfadow)  # Boundary (confined concrete)
    ops.nDMaterial('FSAM', IDmatWeb, 0.0, IDsX, IDsYw, IDconcWeb, rouXw, rouYw, nu, alfadow)  # Web (unconfined concrete)
    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print('FSAM', IDmatBE, 0.0, IDsX, IDsYb, IDconcBE, rouXb, rouYb, nu, alfadow)  # Boundary (confined concrete)
        print('FSAM', IDmatWeb, 0.0, IDsX, IDsYw, IDconcWeb, rouXw, rouYw, nu, alfadow)  # Web (unconfined concrete)
        print('--------------------------------------------------------------------------------------------------')

    # --------------------------------------------------------------------------------
    #  Define 'SFI-MVLEM' elements
    # --------------------------------------------------------------------------------
    # Set 'SFI-MVLEM' parameters thick, width, rho, matConcrete, matSteel
    # MVLEM_thick = [tb if i in (0, eleL - 1) else tw for i in range(eleL)]
    # MVLEM_width = [lbe if i in (0, eleL - 1) else elelweb for i in range(eleL)]
    # MVLEM_mat = [IDmatBE if i in (0, eleL - 1) else IDmatWeb for i in range(eleL)]

    MVLEM_thick = [tb if i in (0, 1, eleL - 2, eleL - 1) else tw for i in range(eleL)]
    MVLEM_width = [lbe/2 if i in (0, 1, eleL - 2, eleL - 1) else elelweb for i in range(eleL)]
    MVLEM_mat = [IDmatBE if i in (0, 1, eleL - 2, eleL - 1) else IDmatWeb for i in range(eleL)]
    # MVLEM_rho = [rouYb if i in (0, eleL - 1) else rouYw for i in range(eleL)]
    # MVLEM_matConcrete = [IDconcBE if i in (0, eleL - 1) else IDconcWeb for i in range(eleL)]
    # MVLEM_matSteel = [IDsYb if i in (0, eleL - 1) else IDsYw for i in range(eleL)]

    for i in range(eleH):
        # ------------------ MVLEM ----------------------------------------------
        # ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', IDmatSpring)
        # if printProgression:
        #     print('MVLEM', i + 1, 0.0, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-rho', *MVLEM_rho, '-matConcrete', *MVLEM_matConcrete, '-matSteel', *MVLEM_matSteel, '-matShear', IDmatSpring)

        # ---------------- SFI_MVLEM -------------------------------------------
        # ops.element('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
        # if printProgression:
        #     print('SFI_MVLEM', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)

        # ---------------- E_SFI -----------------------------------------------
        ops.element('E_SFI', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
        if printProgression:
            print('E_SFI', i + 1, *[i + 1, i + 2], eleL, 0.4, '-thick', *MVLEM_thick, '-width', *MVLEM_width, '-mat', *MVLEM_mat)
    Ag = tw * (lw - (2 * lbe)) + 2 * (tb * lbe)  # Calculate Ag based on provided formula
    ar = hw/lw
    # parameter_values = [tw, tb, hw, lw, lbe, fc, fyb, fyw, fx, round(rouYb, 4), round(rouYw, 4), round(rouXb, 4), round(rouXw, 4), loadF, Ag]
    parameter_values = [tw, tb, hw, lw, ar, lbe, fc, fyb, fyw, fx, rouYb, rouYw, rouXb, rouXw, loadF, Ag]


    if printProgression:
        print('--------------------------------------------------------------------------------------------------')
        print("\033[92mModel Built Successfully --> Using the following parameters :", parameter_values,  "\033[0m")
        print('--------------------------------------------------------------------------------------------------')


def run_gravity(steps=10, printProgression=False):
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


def run_analysis(displacement_step, analysis='cyclic', printProgression=True, enablePlotting=False):
    if printProgression:
        print(f"RUNNING {analysis.upper()} ANALYSIS")

    # For pushover analysis, generate linear displacement steps
    if analysis == 'pushover':
        MaxDisp = max(displacement_step)
        dispIncr = (max(displacement_step) / len(displacement_step))
        displacement_step = [dispIncr * i for i in range(int(MaxDisp / dispIncr))]

    '''
    for i in range(0, eH):
        for j in range(0, eL):
            # Unaxial Steel Recorders for all panels
            # ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_steelX_ele_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelX')
            # ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_steelY_ele_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_steelY')

            # Unaxial Concrete Recorders for all panels
            ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_concr1_ele_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete1')
            ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_concr2_ele_{i + 1}_panel_{j + 1}.txt',  '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_concrete2')

            # Shear Concrete Recorders for all panels
            # ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_inter1_ele_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock1')
            # ops.recorder('Element', '-file', f'plot/MVLEM_strain_stress_inter2_ele_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'strain_stress_interlock2')

            ops.recorder('Element', '-file', f'plot/MVLEM_cracking_angle_ele_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'cracking_angles')
            ops.recorder('Element', '-file', f'plot/MVLEM_panel_crack_{i + 1}_panel_{j + 1}.txt', '-ele', i + 1, 'RCPanel', j + 1, 'panel_crack')
    # '''  # Recorders

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
    ops.system("ProfileSPD")  # UmfPack 19
    ops.test('NormDispIncr', 1e-6, desired_iter, 0)
    ops.algorithm('KrylovNewton')  # KrylovNewton
    ops.analysis("Static")

    Nsteps = len(displacement_step)
    dispData = np.zeros(Nsteps + 1)
    loadData = np.zeros(Nsteps + 1)
    # forceData = np.zeros(Nsteps + 1)

    # Initialize storage for current step data
    strain_matrix_1 = np.zeros((Nsteps, eH, eL))
    strain_matrix_2 = np.zeros((Nsteps, eH, eL))
    crack_angles_1 = np.zeros((Nsteps, eH, eL))
    crack_angles_2 = np.zeros((Nsteps, eH, eL))

    # Initialize NumPy arrays for maximum values
    max_crack_angle_1 = np.zeros((eH, eL, 2))  # [:,:,0] for strain, [:,:,1] for angle
    max_crack_angle_2 = np.zeros((eH, eL, 2))

    # el_tags = ops.getEleTags()
    # nels = len(el_tags)
    # Eds = np.zeros((Nsteps, nels, 6))
    # timeV = np.zeros(Nsteps)

    finishedSteps = 0
    D0 = 0.0
    for j in range(Nsteps):
        D1 = displacement_step[j]
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
                if printProgression:
                    print("Target displacement has been reached. Current Dincr = {:.3g}".format(dU_cumulative))
                increment_done = True
                break
            # adapt the current displacement increment
            dU_adapt = dU * factor
            if abs(dU_cumulative + dU_adapt) > (abs(Dincr) - dU_tolerance):
                dU_adapt = Dincr - dU_cumulative
            # update integrator
            ops.integrator("DisplacementControl", controlNode, controlNodeDof, dU_adapt)
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
                if printProgression:
                    print(f"Increment: {j:6d} | Iterations: {num_iter:4d} | Norm: {error_norm:8.3e} | Progress: {j / Nsteps * 100.0:7.3f} %")

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
            else: # Convergence failed, reduce factor
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

        # Record results
        finishedSteps = j + 1
        disp = ops.nodeDisp(controlNode, 1)
        baseLoad = ops.getLoadFactor(2) / 1000 * RefLoad  # patternTag(2) Convert to from N to kN
        # eleForce = ops.eleForce(1, 4) / 1000
        dispData[j + 1] = disp
        loadData[j + 1] = baseLoad
        # forceData[j + 1] = eleForce

        # Loop to store the response for each panel at each timestep
        for i in range(eH):
            for k in range(eL):
                panel_key = (i, k)
                # Layer 1: Get strain/stress response and crack angle
                crack_strain_1 = ops.eleResponse(i + 1, "RCPanel", k + 1, "strain_stress_concrete1")[0]
                strain_matrix_1[j, i, k] = crack_strain_1

                # Layer 2: Get strain/stress response and crack angle
                crack_strain_2 = ops.eleResponse(i + 1, "RCPanel", k + 1, "strain_stress_concrete2")[0]
                strain_matrix_2[j, i, k] = crack_strain_2

                crack_angle_1, crack_angle_2 = ops.eleResponse(i + 1, "RCPanel", k + 1, "cracking_angles")
                crack_angles_1[j, i, k] = crack_angle_1
                crack_angles_2[j, i, k] = crack_angle_2

                # maximum strains and corresponding angles if current strain is larger
                if abs(crack_strain_1) > abs(max_crack_angle_1[i, k, 0]):
                    max_crack_angle_1[i, k, 0] = crack_strain_1
                    max_crack_angle_1[i, k, 1] = crack_angle_1
                if abs(crack_strain_2) > abs(max_crack_angle_2[i, k, 0]):
                    max_crack_angle_2[i, k, 0] = crack_strain_2
                    max_crack_angle_2[i, k, 1] = crack_angle_2

    # Convert the arrays to NumPy arrays and flatten them
    max_cracks_1 = max_crack_angle_1[:, :, 0].flatten()
    max_angles_1 = max_crack_angle_1[:, :, 1].flatten()
    max_cracks_2 = max_crack_angle_2[:, :, 0].flatten()
    max_angles_2 = max_crack_angle_2[:, :, 1].flatten()

    # Plotting section
    # if enablePlotting:
        # fig = plot_max_panel_response(eH, eL,
        #                               max_cracks_1,
        #                               max_angles_1,
        #                               max_cracks_2,
        #                               max_angles_2)
        # ani = plot_panel_response_animation(eH, eL, Nsteps,
        #                                     strain_matrix_1,
        #                                     strain_matrix_2,
        #                                     crack_angles_1,
        #                                     crack_angles_2)

        # plt.show()

        # ani2 = plot_deformation_animation(Eds, timeV)
        # plt.show()



    # Stack the arrays vertically (each array as a column)
    # combined_data = np.column_stack((max_cracks_1, max_angles_1, max_cracks_2, max_angles_2))
    # np.savetxt('combined_data.csv', combined_data, delimiter=',', header='max_cracks_1,max_angles_1,max_cracks_2,max_angles_2', comments='')

    # return [dispData[0:finishedSteps], loadData[0:finishedSteps]
    return [dispData[0:finishedSteps], loadData[0:finishedSteps], max_cracks_1, max_angles_1, max_cracks_2, max_angles_2]



