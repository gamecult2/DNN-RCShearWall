import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import openseespy.postprocessing.ops_vis as opsv
import openseespy.postprocessing.Get_Rendering as opsplt
import time
import GeneratePeaks

def build_model(wall_height=2,
                wall_width=1,
                wall_thickness=0.125,
                be_length=0.200,
                rouYb=0.029333,
                rouYw=0.003333,
                compStrength=20.7,
                yieldStrength=306,
                axialForce=246):

    # Units - All results in mm, MPa, N and Sec
    mm = 1.0  # Meters
    N = 1.0  # Newtons
    sec = 1.0  # Seconds
    MPa = N / mm  # Pressure

    KN = 0.001 * N  # KiloNewtons
    m = 1000 * mm  # Millimeters
    cm = 10 * mm  # Centimeters
    ton = KN * (sec ** 2) / m  # mass unit (derived)
    g = 9.81 * (m / sec ** 2)  # gravitational constant

    GPa = MPa * 1000

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # --------------------------------------------
    # Set geometry, ops.nodes, boundary conditions
    # --------------------------------------------
    # Wall Geometry
    wall_height = h = 3.66 * m  # Wall height
    wall_thickness = t = 102 * mm  # Wall thickness
    wall_width = w = 1.22 * m  # Wall width
    num_elements = m = 16
    num_MVLEM = n = 8

    # Loop through the list of node values
    for i in range(1, num_elements + 2):
        ops.node(i, 0, (i - 1) * (h / num_elements))
        # print(f'ops.node({i}, 0, {(i - 1) * (h / num_elements)})')

    # Boundary conditions
    ops.fix(1, 1, 1, 1)  # Fixed condition at node 1

    # Set Control Node and DOF
    IDctrlNode = num_elements + 1
    IDctrlDOF = 1

    # ------------------------------------------------------------------------
    # Define uniaxial materials
    # ------------------------------------------------------------------------
    # STEEL ...........................................................
    # steel Y boundary
    fyYbp = 395 * MPa  # fy - tension
    bybp = 0.0185  # strain hardening - tension
    fyYbn = 434 * MPa  # fy - compression
    bybn = 0.02  # strain hardening - compression

    # steel Y web
    fyYwp = 336 * MPa  # fy - tension
    bywp = 0.035  # strain hardening - tension
    fyYwn = 448 * MPa  # fy - compression
    bywn = 0.02  # strain hardening - compression

    # steel misc
    Es = 200 * GPa  # Young's modulus
    R0 = 20.0  # initial value of curvature parameter
    a1 = 18.5  # curvature degradation parameter
    a2 = 0.0015  # curvature degradation parameter

    # Build steel materials
    ops.uniaxialMaterial('SteelMPF', 1, fyYbp, fyYbn, Es, bybp, bybn, R0, a1, a2)  # steel Y boundary
    ops.uniaxialMaterial('SteelMPF', 2, fyYwp, fyYwn, Es, bywp, bywn, R0, a1, a2)  # steel Y web

    # Set 'MVLEM' Reinforcing Ratios
    rouYb = 0.029333  # Y boundary
    rouYw = 0.003333  # Y web

    # CONCRETE ........................................................
    # unconfined
    fpc = 42.8 * MPa  # peak compressive stress
    Ec = 31.03 * GPa  # Young's modulus

    ec0 = -0.0021  # strain at peak compressive stress
    ft = 2.03 * MPa  # peak tensile stress
    et = 0.00008  # strain at peak tensile stress

    xcrnu = 1.039  # cracking strain - compression
    xcrp = 10000  # cracking strain - tension
    ru = 7  # shape parameter - compression
    rt = 1.2  # shape parameter - tension

    # confined
    fpcc = -47.6 * MPa  # peak compressive stress
    print(fpcc)
    fpcc = 1.3 * fc * MPa  # peak compressive stress
    print(fpcc)
    Ecc = 31.03 * GPa  # Young's modulus
    ec0c = -0.0033  # strain at peak compressive stress

    xcrnc = 1.0125  # cracking strain - compression
    rc = 7.3049  # shape parameter - compression

    # Build concrete materials
    # confined concrete
    ops.uniaxialMaterial('ConcreteCM', 3, fpcc, ec0c, Ecc, rc, xcrnc, ft, et, rt, xcrp, '-GapClose', 1)
    # unconfined concrete
    ops.uniaxialMaterial('ConcreteCM', 4, fpc, ec0, Ec, ru, xcrnu, ft, et, rt, xcrp, '-GapClose', 1)

    # SHEAR ........................................................
    # NOTE: large shear stiffness assigned since only flexural response
    area = w * t
    Gc = Ec / (2 * (1 + 0.2))  # Shear Modulus G = E / 2 * (1 + v)
    shear = Gc * area * 2 / 3  # Shear stiffness k * A * G

    # Build shear material
    ops.uniaxialMaterial('Elastic', 5, shear)  # Shear Model for Section Aggregator CONCRETE C40

    # ------------------------------
    #  Define 'MVLEM' elements
    # ------------------------------
    # element('MVLEM',        Tag  iNode jNode  m  c    '-thick',  fiberThick              '-width', *[fiberWidth                           '-rho', Rho                                            '-matConcrete',  '-matSteel',  '-matShear'
    for i in range(num_elements):
        ops.element('MVLEM', i + 1, 0.0, *[i + 1, i + 2], 8, 0.4, '-thick', *[t] * 8, '-width', *[7.5, 1.5, 7.5, 7.5, 7.5, 7.5, 1.5, 7.5], '-rho', *[rouYb, 0.0, rouYw, rouYw, rouYw, rouYw, 0.0, rouYb], '-matConcrete', *[3, 4, 4, 4, 4, 4, 4, 3], '-matSteel', *[1, 2, 2, 2, 2, 2, 2, 1], '-matShear', 5)

    # -------------------------------------------------------
    # Set parameters for displacement controlled analysis
    # -------------------------------------------------------
    load_value = 0.07 * ((wall_width * wall_thickness) * fpc)

    parameters = [wall_height, wall_width, wall_thickness, be_length, rouYb, rouYw, compStrength, yieldStrength, axialForce, IDctrlNode, IDctrlDOF, load_value]
    opsv.plot_model()
    return parameters

def run_gravity(steps=10, load_value=85, load_node=17):
    """
    :param load_node:
    :param steps: total number of analysis steps
    :param load_value: Static load value
    """
    print("running gravity")
    # Records the response of a number of nodes at every converged step
    # ops.recorder('Node', '-file', 'DataGeneration/Gravity_Reactions.out',
    #              '-time', '-node', *[1, 2], '-dof', *[1, 2, 3], 'reaction')

    ops.timeSeries("Linear", 1)  # create TimeSeries for gravity analysis

    ops.pattern('Plain', 1, 1)

    ops.load(load_node, 0.0, -load_value, 0.0)  # apply vertical load

    # plain constraint handler enforces homogeneous single point constraints
    ops.constraints('Transformation')

    # RCM numberer uses the reverse Cuthill-McKee scheme to order the matrix equations
    ops.numberer('RCM')

    # Constructs a profileSPDSOE (Symmetric Positive Definite) system of equation object
    ops.system('BandGeneral')

    # Uses the norm of the left hand side solution vector of the matrix equation to
    # determine if convergence has been reached
    ops.test('NormDispIncr', 1.0e-5, 100, 0)

    # Uses the Newton-Raphson algorithm to solve the nonlinear residual equation
    ops.algorithm('Newton')

    # Uses LoadControl integrator object
    ops.integrator('LoadControl', 0.1)

    # Constructs the Static Analysis object
    ops.analysis('Static')

    # Records the current state of the model
    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time

    # Performs the analysis
    ops.analyze(steps)
    print("Gravity analysis Done!")


def run_cyclic(steps=5000, load_node=17):

    print("running pushover")
    tic = time.time()
    Fact = build_model()[0]
    load_node = build_model()[9]
    Plateral = 1.0
    IDctrlNode = build_model()[9]  # Control node ID
    IDctrlDOF = build_model()[10]  # Control DOF ID
    Dincr = 0.001

       # ------------------------------
    # Recorder generation
    # ------------------------------
    # Nodal recorders
    ops.recorder('Node', '-file', 'dataDir/MVLEM_Dtop.out', '-time', '-node', IDctrlNode, '-dof', 1, 'disp')
    ops.recorder('Node', '-file', 'dataDir/MVLEM_DOFs.out', '-time', '-node', 1, 2, 3, 4, '-dof', 1, 2, 3, 'disp')

    # Element recorders
    ops.recorder('Element', '-file', 'dataDir/MVLEM_Fgl.out', '-time', '-ele', 1, 2, 3, 'globalForce')
    ops.recorder('Element', '-file', 'dataDir/MVLEM_Curvature.out', '-time', '-ele', 1, 2, 3, 'Curvature')

    # Apply lateral load based on first mode shape in x direction (EC8-1)
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 200, 2)  # define load pattern -- generalized
    ops.load(load_node, Plateral, 0.0, 0.0)  # apply lateral load

    # Constructs a transformation constraint handler,
    ops.constraints('Transformation')

    # RCM numberer uses the reverse Cuthill-McKee scheme to order the matrix equations
    ops.numberer('RCM')

    # Construct a BandGeneralSOE linear system of equation object
    ops.system('BandGen')

    # Uses the norm of the left hand side solution vector of the matrix equation to
    ops.test('NormDispIncr', 1.e-5, 100, 0)

    # Line search increases the effectiveness of the Newton method
    ops.algorithm('ModifiedNewton')  # ops.algorithm('NewtonLineSearch', True, 0.8, 1000, 0.1, 10.0)

    ops.integrator('DisplacementControl', 17, 1, Dincr)
    # Constructs the Static Analysis object
    ops.analysis('Static')

    # Define the parameters and variables
    fmt1 = "%s Cyclic analysis: CtrlNode %.3i, dof %.1i, Disp=%.4f %s"



    # Vector of displacement-cycle peaks in terms of wall drift ratio (flexural displacements)
    iDmax2 = [0.000330792, 0.001104233, 0.002925758, 0.004558709, 0.006625238, 0.010816268, 0.014985823, 0.019655056]
    iDmax = [value * 25.4 for value in iDmax2]
    CycleType = "Full"
    Ncycles = 2
    Tol = 1.0e-5
    LunitTXT = "inch"

    load_step = 1  # Initial load step

    # Loop through each Dmax value
    for Dmax in iDmax:
        iDstep = GeneratePeaks.generate_peaks(Dmax, Dincr, CycleType, Fact)
        zeroD = 0
        D0 = 0.0
        for i in range(1, Ncycles + 1):
            for Dstep in iDstep:
                D1 = Dstep
                Dincr = D1 - D0
                ops.integrator('DisplacementControl', 17, 1, Dincr)
                ops.analysis('Static')

                # First analyze command
                ok = ops.analyze(1)

                D0 = D1
                print("Load Step:", load_step)
                load_step += 1

    # Final output
    if ok != 0:
        print(fmt1 % ("PROBLEM", IDctrlNode, IDctrlDOF, ops.nodeDisp(IDctrlNode, IDctrlDOF), "LunitTXT"))
    else:
        print(fmt1 % ("DONE", IDctrlNode, IDctrlDOF, ops.nodeDisp(IDctrlNode, IDctrlDOF), "LunitTXT"))

    # Performs the analysis

    toc = time.time()
    print('Cyclic Analysis Done in {:1.2f} seconds'.format(toc - tic))

    # Print the state at the control node
    ops.print('node', IDctrlNode)

def reset_analysis():
    """
    Resets the analysis by setting time to 0,
    removing the recorders and wiping the analysis.
    """

    # Reset for next analysis case
    ##  Set the time in the Domain to zero
    ops.setTime(0.0)
    # Set the loads constant in the domain
    ops.loadConst()
    # Remove all recorder objects.
    ops.remove('recorders')
    # destroy all components of the Analysis object
    ops.wipeAnalysis()

build_model()
run_gravity()
run_cyclic()
reset_analysis()
ops.wipe()