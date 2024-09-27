import math
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import ReadRecord
import opsvis as osv

from Units import *
from RCsection import *  # Import the RCsection class specifically


def reset_analysis():
    ops.setTime(0.0)  # Set the time in the Domain to zero
    ops.loadConst()  # Set the loads constant in the domain
    ops.remove('recorders')  # Remove all recorder objects.
    ops.wipeAnalysis()  # destroy all components of the Analysis object
    ops.wipe()


def rebar_area(RebarDiameter):
    a = math.pi * (RebarDiameter / 2) ** 2  # compute area
    return a


def building_model(NStory, NBay, NBayZ, LCol, LBeam, LGird, HCol, BCol, HBeam, BBeam, HGird, BGird, cover, numBarsTopCol, numBarsBotCol, numBarsIntCol, barAreaTopCol, barAreaBotCol, barAreaIntCol,
                   numBarsTopBeam, numBarsBotBeam, numBarsIntBeam, barAreaTopBeam, barAreaBotBeam, barAreaIntBeam, numBarsTopGird, numBarsBotGird, numBarsIntGird, barAreaTopGird, barAreaBotGird, barAreaIntGird,
                   nfCoreY, nfCoreZ, nfCoverY, nfCoverZ, Tslab, printProgression=True):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    data_dir = "Data"  # data folder
    gm_dir = "GMfiles"  # ground motion folder

    # define Building configuration
    NStory = 10  # number of stories above ground level
    NBay = 2  # number of bays in X
    NBayZ = 2  # number of bays in Z
    NFrame = NBayZ + 1
    print("\nNumber of Stories in Y:", NStory, "\nNumber of bays in X:", NBay, "\nNumber of bays in Z:", NBayZ)

    # define GEOMETRY of elements
    LCol = 3657.6 * mm  # column height Y axis)
    LBeam = 7315.2 * mm  # beam length X axis
    LGird = 7315.2 * mm  # girder length Z axis

    # define NODAL COORDINATES
    Dlevel = 10000  # numbering increment for new-level nodes
    Dframe = 100  # numbering increment for new-frame nodes
    # nodes = []
    for frame in range(1, NFrame + 1):
        Z = (frame - 1) * LGird
        for level in range(1, NStory + 2):
            Y = (level - 1) * LCol
            for pier in range(1, NBay + 2):
                X = (pier - 1) * LBeam
                nodeID = level * Dlevel + frame * Dframe + pier
                ops.node(nodeID, X, Y, Z)
                # nodes.append((nodeID, X, Y, Z))

    # rigid diaphragm nodes
    RigidDiaphragm = True
    Xa = (NBay * LBeam) / 2
    Za = (NFrame - 1) * LGird / 2
    iMasterNode = []
    for level in range(2, NStory + 2):
        Y = (level - 1) * LCol
        MasterNodeID = 9900 + level
        # master nodes for rigid diaphragm (implementation specific)
        ops.node(MasterNodeID, Xa, Y, Za)
        # nodes.append((MasterNodeID, Xa, Y, Za))
        ops.fix(MasterNodeID, 0, 1, 0, 1, 0, 1)
        iMasterNode.append(MasterNodeID)
        perpDirn = 2  # perpendicular to the plane of rigid diaphragm
        for frame in range(1, NFrame + 1):
            for pier in range(1, NBay + 2):
                nodeID = level * Dlevel + frame * Dframe + pier
                # define Rigid Diaphragm (implementation specific)
                ops.rigidDiaphragm(perpDirn, MasterNodeID, nodeID)

    # Extract coordinates from node data
    # x = [node[1] for node in nodes]
    # y = [node[2] for node in nodes]
    # z = [node[3] for node in nodes]
    # node_numbers = [node[0] for node in nodes]  # Extract node numbers
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for node, number in zip(nodes, node_numbers):
    #     ax.plot([node[1]], [node[3]], [node[2]], 'o', markersize=5, color='red')  # Plot the node
    #     ax.text(node[1], node[3], node[2], number, ha='center', va='center', fontsize=8)  # Add node number text
    #
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # ax.set_title('3D Node Plot')
    # ax.view_init(elev=15, azim=-60)  # Adjust elevation (elev) and azimuth (azim) for different views
    # plt.show()
    # --------------------------------------------------
    # determine support nodes where ground motions are input (multiple-support excitation)
    iSupportNode = []
    for frame in range(1, NFrame + 1):
        level = 1
        for pier in range(1, NBay + 2):
            nodeID = level * Dlevel + frame * Dframe + pier
            iSupportNode.append(nodeID)

    # --------------------------------------------------
    # Boundary Conditions
    ops.fixY(0.0, *[1, 1, 1, 0, 1, 0])

    # calculated MODEL PARAMETERS, particular to this model
    # Set up parameters that are particular to the model for displacement control
    global IDctrlNode, IDctrlDOF, LBuilding
    IDctrlNode = int((NStory + 1) * Dlevel + (1 * Dframe) + 1)  # node where displacement is read for displacement control
    IDctrlDOF = 1  # degree of freedom of displacement read for displacement control
    LBuilding = NStory * LCol  # total building height
    # print('LBuilding', LBuilding)

    # --------------------------------------------------
    # Define SECTIONS
    SectionType = "FiberSection"

    # define section tags:
    ColSecTag = 1
    BeamSecTag = 2
    GirdSecTag = 3
    ColSecTagFiber = 4
    BeamSecTagFiber = 5
    GirdSecTagFiber = 6
    SecTagTorsion = 70

    # Section Properties:
    HCol = 457.2 * mm  # square-Column width
    BCol = HCol
    HBeam = 609.6 * mm  # Beam depth -- perpendicular to bending axis
    BBeam = 457.2 * mm  # Beam width -- parallel to bending axis
    HGird = 609.6 * mm  # Girder depth -- perpendicular to bending axis
    BGird = 457.2 * mm  # Girder width -- parallel to bending axis

    '''
    if SectionType == "Elastic":
        fc = 25 * MPa
        Ec = 37 * GPa
        nu = 0.2
        Gc = Ec / (2 * (1 + nu))
        J = 1e12
        # column section properties:
        AgCol = HCol * BCol  # rectangular-Column cross-sectional area
        IzCol = 0.5 * (1 / 12) * BCol * HCol ** 3  # about-local-z Rect-Column gross moment of inertia
        IyCol = 0.5 * (1 / 12) * HCol * BCol ** 3  # about-local-z Rect-Column gross moment of inertia
        # beam sections:
        AgBeam = HBeam * BBeam  # rectangular-Beam cross-sectional area
        IzBeam = 0.5 * (1 / 12) * BBeam * HBeam ** 3  # about-local-z Rect-Beam cracked moment of inertia
        IyBeam = 0.5 * (1 / 12) * HBeam * BBeam ** 3  # about-local-y Rect-Beam cracked moment of inertia
        # girder sections:
        AgGird = HGird * BGird  # rectangular-Girder cross-sectional area
        IzGird = 0.5 * (1 / 12) * BGird * HGird ** 3  # about-local-z Rect-Girder cracked moment of inertia
        IyGird = 0.5 * (1 / 12) * HGird * BGird ** 3  # about-local-y Rect-Girder cracked moment of inertia

        ops.section("Elastic", ColSecTag, Ec, AgCol, IzCol, IyCol, Gc, J)
        ops.section("Elastic", BeamSecTag, Ec, AgBeam, IzBeam, IyBeam, Gc, J)
        ops.section("Elastic", GirdSecTag, Ec, AgGird, IzGird, IyGird, Gc, J)

        IDconcCore = 1  # material numbers for ops.recorder (this stress-strain ops.recorder will be blank, as this is an elastic section)
        IDSteel = 2  # material numbers for ops.recorder (this stress-strain ops.recorder will be blank, as this is an elastic section)
    '''

    if SectionType == "FiberSection":
        import Materials  # MATERIAL parameters
        # FIBER SECTION properties
        # Column section geometry:
        GJ = 1.e12 * 1.0

        cover = 63.5 * mm  # rectangular-RC-Column cover (m)
        numBarsTopCol = 8  # number of reinforcement bars on top layer
        numBarsBotCol = 8  # number of reinforcement bars on bottom layer
        numBarsIntCol = 6  # number of reinforcement bars on intermediate layers
        barAreaTopCol = rebar_area(12)  #
        barAreaBotCol = rebar_area(12)  # reinforcement bar area (m2)
        barAreaIntCol = rebar_area(12)  #

        numBarsTopBeam = 6  # number of reinforcement bars on top layer
        numBarsBotBeam = 6  # number of reinforcement bars on bottom layer
        numBarsIntBeam = 2  # number of reinforcement bars on intermediate layers
        barAreaTopBeam = rebar_area(12)  #
        barAreaBotBeam = rebar_area(12)  # reinforcement bar area (m2)
        barAreaIntBeam = rebar_area(12)  #

        numBarsTopGird = 6  # number of reinforcement bars on top layer
        numBarsBotGird = 6  # number of reinforcement bars on bottom layer
        numBarsIntGird = 2  # number of reinforcement bars on the intermediate layers
        barAreaTopGird = rebar_area(12)  #
        barAreaBotGird = rebar_area(12)  # reinforcement bar area (m2)
        barAreaIntGird = rebar_area(12)  #

        nfCoreY = 20  # number of fibers in the core patch in the y direction
        nfCoreZ = 20  # number of fibers in the core patch in the z direction
        nfCoverY = 20  # number of fibers in the cover patches with long sides in the y direction
        nfCoverZ = 20  # number of fibers in the cover patches with long sides in the z direction

        # rectangular section with one layer of steel evenly distributed around the perimeter and a confined core.
        BuildRCrectSection(ColSecTagFiber, HCol, BCol, cover, cover, Materials.IDconcCore, Materials.IDconcCover, Materials.IDSteel, numBarsTopCol, barAreaTopCol, numBarsBotCol, barAreaBotCol, numBarsIntCol, barAreaIntCol, nfCoreY, nfCoreZ, nfCoverY, nfCoverZ)
        BuildRCrectSection(BeamSecTagFiber, HBeam, BBeam, cover, cover, Materials.IDconcCore, Materials.IDconcCover, Materials.IDSteel, numBarsTopBeam, barAreaTopBeam, numBarsBotBeam, barAreaBotBeam, numBarsIntBeam, barAreaIntBeam, nfCoreY, nfCoreZ, nfCoverY, nfCoverZ)
        BuildRCrectSection(GirdSecTagFiber, HGird, BGird, cover, cover, Materials.IDconcCore, Materials.IDconcCover, Materials.IDSteel, numBarsTopGird, barAreaTopGird, numBarsBotGird, barAreaBotGird, numBarsIntGird, barAreaIntGird, nfCoreY, nfCoreZ, nfCoverY, nfCoverZ)

        # assign torsional Stiffness for 3D Model
        ops.uniaxialMaterial("Elastic", SecTagTorsion, GJ)
        ops.section("Aggregator", ColSecTag, *[SecTagTorsion, 'T'], "-section", ColSecTagFiber)
        ops.section("Aggregator", BeamSecTag, *[SecTagTorsion, 'T'], "-section", BeamSecTagFiber)
        ops.section("Aggregator", GirdSecTag, *[SecTagTorsion, 'T'], "-section", GirdSecTagFiber)
    else:
        print("No section has been defined")

    GammaConcrete = 2.40e-9 * t_mm3  # Density of concrete in pounds per cubic foot kilograms per cubic metre
    QdlCol = GammaConcrete * HCol * BCol  # self weight of Column, weight per length
    QBeam = GammaConcrete * HBeam * BBeam  # self weight of Beam, weight per length
    QGird = GammaConcrete * HGird * BGird  # self weight of Gird, weight per length

    # --------------------------------------------------------------------------------------------------------------------------------
    # define ELEMENTS
    # set up geometric transformations of element separate columns and beams, in case of P-Delta analysis for columns
    IDColTransf = 1  # all columns
    IDBeamTransf = 2  # all beams
    IDGirdTransf = 3  # all girders
    ColTransfType = "Linear"  # options, "Linear" or "PDelta" or "Corotational"

    ops.geomTransf(ColTransfType, IDColTransf, *[0, 0, 1])  # only columns can have PDelta effects (gravity effects)
    ops.geomTransf('Linear', IDBeamTransf, *[0, 0, 1])
    ops.geomTransf('Linear', IDGirdTransf, *[1, 0, 0])

    # Define Beam-Column Elements
    numIntgrPts = 5  # number of Gauss integration points for nonlinear curvature distribution
    all_elements = []  # Use a descriptive variable name
    # columns
    N0col = 10000 - 1  # column element numbers
    # ops.beamIntegration('Lobatto', 1, ColSecTag, numIntgrPts)
    for frame in range(1, NFrame + 1):
        for level in range(1, NStory + 1):
            for pier in range(1, NBay + 2):
                elemID = N0col + level * Dlevel + frame * Dframe + pier
                all_elements.append(elemID)  # Correct append syntax
                nodeI = level * Dlevel + frame * Dframe + pier
                nodeJ = (level + 1) * Dlevel + frame * Dframe + pier
                ops.element("nonlinearBeamColumn", elemID, *[nodeI, nodeJ], numIntgrPts, ColSecTag, IDColTransf)  # columns
                # ops.element("dispBeamColumn", elemID, *[nodeI, nodeJ], IDColTransf, 1)  # columns
                # ops.element("dispBeamColumn", elemID, *[nodeI, nodeJ], IDColTransf, 1)  # columns

    # beams -- parallel to X-axis
    N0beam = 1000000  # beam element numbers
    # ops.beamIntegration('Lobatto', 2, BeamSecTag, numIntgrPts)
    for frame in range(1, NFrame + 1):
        for level in range(2, NStory + 2):
            for bay in range(1, NBay + 1):
                elemID = N0beam + level * Dlevel + frame * Dframe + bay
                all_elements.append(elemID)  # Correct append syntax
                nodeI = level * Dlevel + frame * Dframe + bay
                nodeJ = level * Dlevel + frame * Dframe + bay + 1
                ops.element("nonlinearBeamColumn", elemID, *[nodeI, nodeJ], numIntgrPts, BeamSecTag, IDBeamTransf)  # beam
                # ops.element("dispBeamColumn", elemID, *[nodeI, nodeJ], IDBeamTransf, 2)  # beam

    # girders -- parallel to Z-axis
    N0gird = 2000000  # gird element numbers
    # ops.beamIntegration('Lobatto', 3, GirdSecTag, numIntgrPts)
    for frame in range(1, NFrame):
        for level in range(2, NStory + 2):
            for bay in range(1, NBay + 2):
                elemID = N0gird + level * Dlevel + frame * Dframe + bay
                all_elements.append(elemID)  # Correct append syntax
                nodeI = level * Dlevel + frame * Dframe + bay
                nodeJ = level * Dlevel + (frame + 1) * Dframe + bay
                ops.element("nonlinearBeamColumn", elemID, *[nodeI, nodeJ], numIntgrPts, GirdSecTag, IDGirdTransf)  # Girds
                # ops.element("dispBeamColumn", elemID, *[nodeI, nodeJ], IDGirdTransf, 3)  # Girds

    # Define GRAVITY LOADS, weight and masses
    # calculate dead load of frame, assume this to be an internal frame (do LL in a similar manner)
    # calculate distributed weight along the beam length
    Tslab = 152.4 * mm  # slab thickness
    Lslab = LGird / 2  # slab extends a distance of $LGird/2 in/out of plane
    DLfactor = 1.0  # scale dead load up a little
    Qslab = GammaConcrete * Tslab * Lslab * DLfactor
    QdlBeam = Qslab + QBeam  # dead load distributed along beam (one-way slab)
    QdlGird = QGird  # dead load distributed along girder
    WeightCol = QdlCol * LCol  # total Column weight
    WeightBeam = QdlBeam * LBeam  # total Beam weight
    WeightGird = QdlGird * LGird  # total Beam weight

    # assign masses columns nodes taking 1/2 mass of each element framing into it (mass=weight/g)
    iFloorWeight = []
    WeightTotal = 0.0
    sumWiHi = 0.0  # Sum of storey weight times height, for lateral-load distribution

    # Loop through frames, levels, and piers
    for frame in range(1, NFrame + 1):
        if frame == 1 or frame == NFrame:
            GirdWeightFact = 1  # 1x1/2 girder on exterior frames
        else:
            GirdWeightFact = 2  # 2x1/2 girder on interior frames
        for level in range(2, NStory + 2):
            FloorWeight = 0.0
            if level == NStory + 1:
                ColWeightFact = 1  # One column in top story
            else:
                ColWeightFact = 2  # Two columns elsewhere
            for pier in range(1, NBay + 2):
                if pier == 1 or pier == NBay + 1:
                    BeamWeightFact = 1  # One beam at exterior nodes
                else:
                    BeamWeightFact = 2  # Two beams elsewhere
                WeightNode = ColWeightFact * WeightCol / 2 + BeamWeightFact * WeightBeam / 2 + GirdWeightFact * WeightGird / 2
                MassNode = WeightNode / g
                nodeID = level * Dlevel + frame * Dframe + pier
                ops.mass(nodeID, *[MassNode, 0, MassNode, 0., 0., 0.])  # Define mass
                FloorWeight += WeightNode  # SUm of all nodes mass in the same floor

            iFloorWeight.append(FloorWeight)
            WeightTotal += FloorWeight
            sumWiHi += FloorWeight * (level - 1) * LCol  # sum of storey weight times height, for lateral-load distribution

    MassTotal = WeightTotal / g  # Total mass

    # LATERAL-LOAD distribution for static pushover analysis
    # calculate distribution of lateral load based on mass/weight distributions along building height
    # Fj = WjHj/sum(WiHi)  * Weight   at each floor j
    iFj = []
    for level in range(2, NStory + 2):
        FloorWeight = iFloorWeight[level - 2]
        FloorHeight = (level - 1) * LCol
        Fj = (FloorWeight * FloorHeight) / sumWiHi * WeightTotal
        iFj.append(Fj)  # per floor

    global iNodePush, iFPush
    iNodePush = iMasterNode  # nodes for pushover/cyclic, vectorized
    iFPush = iFj  # lateral load for pushover, vectorized

    # Define ops.recorderS -------------------------------------------------------------
    FreeNodeID = NFrame * Dframe + (NStory + 1) * Dlevel + (NBay + 1)  # ID: free node
    SupportNodeFirst = iSupportNode[0]  # ID: first support node
    SupportNodeLast = iSupportNode[-1]  # ID: last support node
    FirstColumn = N0col + 1 * Dframe + 1 * Dlevel + 1  # ID: first column

    # ------------------------ Nodes Recorder
    ops.recorder('Node', '-file', f"{data_dir}/DFree.out", '-time', '-node', FreeNodeID, "-dof", *[1, 2, 3], "disp")  # displacements of free node
    ops.recorder("Node", "-file", f"{data_dir}/DBase.out", "-time", "-nodeRange", SupportNodeFirst, SupportNodeLast, "-dof", *[1, 2, 3], "disp")  # displacements of support nodes
    ops.recorder("Node", "-file", f"{data_dir}/RBase.out", "-time", "-nodeRange", SupportNodeFirst, SupportNodeLast, "-dof", *[1, 2, 3], "reaction")  # support reaction

    # ------------------------ Nodes Recorder
    # ops.recorder("Drift", "-file", f"{data_dir}/DrNode.out", "-time", "-iNode", SupportNodeFirst, "-jNode", FreeNodeID, "-dof", 1, "-perpDirn", 2)  # lateral drift

    # ------------------------ Element Recorder
    ops.recorder("Element", "-file", f"{data_dir}/Fel1.out", "-time", "-ele", FirstColumn, "localForce")  # element forces in local coordinates
    ops.recorder("Element", "-file", f"{data_dir}/ForceEle1sec1.out", "-time", "-ele", FirstColumn, "section", "1", "force")  # section forces, axial and moment, node i
    ops.recorder("Element", "-file", f"{data_dir}/DefoEle1sec1.out", "-time", "-ele", FirstColumn, "section", "1", "deformation")  # section deformations, axial and curvature, node i
    ops.recorder("Element", "-file", f"{data_dir}/ForceEle1sec" + str(numIntgrPts) + ".out", "-time", "-ele", FirstColumn, "section", numIntgrPts, "force")  # section forces, axial and moment, node j
    ops.recorder("Element", "-file", f"{data_dir}/DefoEle1sec" + str(numIntgrPts) + ".out", "-time", "-ele", FirstColumn, "section", numIntgrPts, "deformation")  # section deformations, axial and curvature, node j
    yFiber = HCol / 2 - cover  # fiber location for stress-strain ops.recorder, local coords
    zFiber = BCol / 2 - cover  # fiber location for stress-strain ops.recorder, local coords

    ops.recorder("Element", "-file", f"{data_dir}/SSconcEle1sec1.out", "-time", "-ele", FirstColumn, "section", numIntgrPts, "fiber", yFiber, zFiber, Materials.IDconcCore, "stressStrain")  # steel fiber stress-strain, node i
    ops.recorder("Element", "-file", f"{data_dir}/SSreinfEle1sec1.out", "-time", "-ele", FirstColumn, "section", numIntgrPts, "fiber", yFiber, zFiber, Materials.IDSteel, "stressStrain")  # steel fiber stress-strain, node i

    if printProgression:
        print("RUNNING GRAVITY ANALYSIS")
    # define GRAVITY load applied to beams and columns -- eleLoad applies loads in local coordinate axis
    tsTagGravity = 101
    patternTagGravity = 101
    ops.timeSeries('Linear', tsTagGravity)
    ops.pattern('Plain', patternTagGravity, tsTagGravity)
    for frame in range(1, NFrame + 1):
        for level in range(1, NStory + 1):
            for pier in range(1, NBay + 2):
                elemID = N0col + level * Dlevel + frame * Dframe + pier
                # print('-ele', elemID, '-type', '-beamUniform', 0.0, 0.0, -QdlCol)  # COLUMNS Wy, Wz, Wx
                ops.eleLoad('-ele', elemID, '-type', '-beamUniform', 0.0, 0.0, -QdlCol)  # COLUMNS Wy, Wz, Wx

    for frame in range(1, NFrame + 1):
        for level in range(2, NStory + 2):
            for bay in range(1, NBay + 1):
                elemID = N0beam + level * Dlevel + frame * Dframe + bay
                ops.eleLoad('-ele', elemID, '-type', '-beamUniform', -QdlBeam, 0.0)  # BEAMS

    for frame in range(1, NFrame):
        for level in range(2, NStory + 2):
            for bay in range(1, NBay + 2):
                elemID = N0gird + level * Dlevel + frame * Dframe + bay
                ops.eleLoad('-ele', elemID, '-type', '-beamUniform', -QdlGird, 0.0)  # GIRDS

    # Gravity-analysis parameters -- load-controlled static analysis
    Tol = 1.0e-8  # convergence tolerance for test
    NstepGravity = 10  # apply gravity in 10 steps
    constraintsTypeGravity = "Plain"  # default;

    if 'RigidDiaphragm' in locals() or 'RigidDiaphragm' in globals():  # Check if a variable named RigidDiaphragm exists in either local or global varaibles
        if RigidDiaphragm:
            constraintsTypeGravity = "Transformation"  # large model: try Transformation or Lagrange

    ops.constraints(constraintsTypeGravity)  # how it handles boundary conditions
    ops.numberer("RCM")  # renumber dof's to minimize band-width (optimization), if you want to
    ops.system("BandGeneral")  # how to store and solve the system of equations in the analysis (large model: try UmfPack)
    ops.test('EnergyIncr', Tol, 6, 0)  # Convergence criteria at each iteration
    ops.algorithm("Newton")  # use Newton's solution algorithm: updates tangent stiffness at every iteration
    ops.integrator("LoadControl", 1 / NstepGravity)  # determine the next time step for an analysis
    ops.analysis("Static")  # define type of analysis static or transient
    ok = ops.analyze(NstepGravity)

    if ok == 0:
        print('GRAVITY ANALYSIS SUCCESSFUL')
    else:
        print('GRAVITY ANALYSIS FAILED')

    ops.loadConst('-time', 0.0)  # hold gravity constant and restart time for further analysis

    print("MODEL BUILT SUCCESSFUL")

    # ---------------------------------------------------------------------------
    # Initialize the ele_shapes dictionary
    ele_shapes = {}
    # Generate ele_shapes dictionary with the same parameters for each key
    for elem_id in all_elements:
        ele_shapes[elem_id] = ('rect', [BGird, HGird])

    ax = osv.plot_model(0, 0, axis_off=0, az_el=(45.0, 45.0), local_axes=False, gauss_points=False)
    osv.plot_extruded_shapes_3d(ele_shapes, az_el=(45.0, 45.0), ax=False)
    # # ax = osv.plot_mode_shape(1, unDefoFlag=0, endDispFlag=0)
    # plt.xlabel('Length')
    # plt.ylabel('Height')
    # # Turn off the x-axis and y-axis tick labels
    # # ax.set_xticks([])  # Hide x-axis tick labels
    # # ax.set_yticks([])  # Hide y-axis tick labels
    # # ax.set_zticks([])  # Hide z-axis tick labels
    ax.view_init(elev=120, azim=-45, roll=46)  # Adjust the angles as needed
    plt.tight_layout()
    plt.show()


def run_modal(num_modes=3):
    # -------------------------------------------------------------------------------------------------------
    # ----------------------- Define damping  ---------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------
    # Eigenvalue analysis
    # Perform an eigenvalue analysis
    xDamp = 0.05  # 5% damping ratio
    num_modes = 3

    mode_i = 0  # mode 1
    mode_j = 2  # mode 3
    lambda_values = ops.eigen(num_modes)
    omegas = []  # List to store omega values
    frequencies = []
    periods = []
    for lam in lambda_values:
        omega = math.sqrt(lam)
        omegas.append(omega)  # Store omega value
        frequencies.append(omega / (2 * math.pi))  # Calculate frequency
        periods.append((2 * math.pi) / omega)  # Calculate period
    print(omegas)
    print("Mode  |     Omega       | Frequency (rad/s) | Period (s)")
    print("------|-----------------|-------------------|-----------")
    for i, mode in enumerate(range(1, num_modes + 1)):
        print(f"{mode:3d}   |    {omegas[i]:.4f}     |      {frequencies[i]:.4f}     |   {periods[i]:.4f}")
        print("------|-----------------|-------------------|-----------")

    MpropSwitch = 1
    KcurrSwitch = 0
    KinitSwitch = 0
    KcommSwitch = 1
    omega_i = omegas[mode_i]
    omega_j = omegas[mode_j]
    print('omega_i', omega_i)
    print('omega_j', omega_j)
    # ----------------------------------------
    alphaM = MpropSwitch * xDamp * (2 * omegas[mode_i] * omegas[mode_j]) / (omegas[mode_i] + omegas[mode_j])  # M-prop. damping; D = alphaM*M
    betaKcurr = KcurrSwitch * 2 * xDamp / (omegas[mode_i] + omegas[mode_j])  # K-proportional damping;      +beatKcurr*KCurrent
    betaKinit = KinitSwitch * 2 * xDamp / (omegas[mode_i] + omegas[mode_j])  # initial-stiffness proportional damping      +beatKinit*Kini
    betaKcomm = KcommSwitch * 2 * xDamp / (omegas[mode_i] + omegas[mode_j])  # last-committed K;   +betaKcomm*KlastCommitt

    #       (alpha_m, beta_k, beta_k_init, beta_k_comm) # RAYLEIGH  D = αM∗M + βK∗Kcurr + βKinit∗Kinit + βKcomm∗Kcommit
    ops.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)


def run_pushover(num_modes=3):
    # characteristics of pushover analysis
    Dmax = 0.1 * LBuilding  # maximum displacement of pushover. push to a % drift.
    Dincr = 0.0000001 * LBuilding  # displacement increment. you want this to be small, but not too small to slow analysis
    Dincr = 0.1 * mm

    # -- STATIC PUSHOVER/CYCLIC ANALYSIS
    # create load pattern for lateral pushover load coefficient when using linear load pattern
    # need to apply lateral load only to the master nodes of the rigid diaphragm at each floor
    tsTagPushover = 201
    patternTagPushover = 201
    ops.timeSeries('Linear', tsTagPushover)
    ops.pattern('Plain', patternTagPushover, tsTagPushover)
    for NodePush, FPush in zip(iNodePush, iFPush):
        ops.load(NodePush, FPush, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Define DISPLAY -------------------------------------------------------------
    # the deformed shape is defined in the build file
    ops.recorder("plot", dataDir + "/DFree.out", "Displ-X", 1200, 10, 300, 300, "-columns", "2", "1")  # a window to plot the nodal displacements versus time
    ops.recorder("plot", dataDir + "/DFree.out", "Displ-Z", 1200, 310, 300, 300, "-columns", "4", "1")  # a window to plot the nodal displacements versus time

    # ---------------------------------    perform Static Pushover Analysis
    fmt1 = "%s Pushover analysis: CtrlNode %.3i, dof %.1i, Disp=%.4f %s"  # format for screen/file output of DONE/PROBLEM analysis

    # ----------------------------------------------first analyze command------------------------
    Nsteps = int(Dmax / Dincr)  # number of pushover analysis steps
    ok = ops.analyze(Nsteps)  # this will return zero if no convergence problems were encountered
    # ----------------------------------------------if convergence failure-------------------------
    if ok != 0:
        # if analysis fails, we try some other stuff, performance is slower inside this loop
        Dstep = 0.0
        ok = 0
        while Dstep <= 1.0 and ok == 0:
            controlDisp = ops.nodeDisp(IDctrlNode, IDctrlDOF)
            Dstep = controlDisp / Dmax
            ok = ops.analyze(1)  # this will return zero if no convergence problems were encountered
            if ok != 0:
                # reduce step size if still fails to converge
                Nk = 4  # reduce step size
                DincrReduced = Dincr / Nk
                ops.integrator("DisplacementControl", IDctrlNode, IDctrlDOF, DincrReduced)
                for ik in range(1, Nk + 1):
                    ok = ops.analyze(1)  # this will return zero if no convergence problems were encountered
                    if ok != 0:
                        # if analysis fails, we try some other stuff
                        # performance is slower inside this loop
                        print("Trying Newton with Initial Tangent ..")
                        ops.test("NormDispIncr", Tol, 2000, 0)
                        ops.algorithm("Newton", "-initial")
                        ok = ops.analyze(1)
                        ops.test(testTypeStatic, TolStatic, maxNumIterStatic, 0)
                        ops.algorithm(algorithmTypeStatic)
                    if ok != 0:
                        print("Trying Broyden ..")
                        ops.algorithm("Broyden", "8")
                        ok = ops.analyze(1)
                        ops.algorithm(algorithmTypeStatic)
                    if ok != 0:
                        print("Trying NewtonWithLineSearch ..")
                        ops.algorithm("NewtonLineSearch", "0.8")
                        ok = ops.analyze(1)
                        ops.algorithm(algorithmTypeStatic)
                    if ok != 0:
                        # stop if still fails to converge
                        print(fmt1 % ("PROBLEM", IDctrlNode, IDctrlDOF, ops.nodeDisp(IDctrlNode, IDctrlDOF)))
                        return -1
                ops.integrator("DisplacementControl", IDctrlNode, IDctrlDOF, Dincr)  # bring back to original increment

    # -----------------------------------------------------------------------------------------------------
    if ok != 0:
        print(fmt1 % ("PROBLEM", IDctrlNode, IDctrlDOF, ops.nodeDisp(IDctrlNode, IDctrlDOF)))
    else:
        print(fmt1 % ("DONE", IDctrlNode, IDctrlDOF, ops.nodeDisp(IDctrlNode, IDctrlDOF)))


def run_THA(GMfolder, GMfile, GMdirection, GMfact, plotResults=False, printProgression=False, recordData=False):
    # Ground motion parameters
    dt, nPts = ReadRecord.ReadRecord(f'{GMfolder}/{GMfile}.AT2', f'{GMfolder}/{GMfile}.dat')
    print(dt, nPts)
    IDtimeTag = 100
    IDloadTag = 100

    print('\n<<<< Running One Direction (Vertical) Ground Motion Analysis >>>>')
    # ----------------------- "Vertical" -----------------------
    ops.timeSeries('Path', IDtimeTag, '-dt', dt, '-filePath', f'{GMfolder}/{GMfile}.dat', '-factor', GMfact)
    ops.pattern('UniformExcitation', IDloadTag, GMdirection, '-accel', IDtimeTag)

    # Create the transient analysis------------------------------------
    ops.setTime(0.0)
    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('Plain')
    ops.system('BandGeneral')
    ops.test('EnergyIncr', 1e-6, 200, 0)
    ops.algorithm('ModifiedNewton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    print(nPts)
    ok = ops.analyze(nPts - 5000, dt)
    Tol = 1e-8  # convergence tolerance for test
    te = {1: 'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance', 5: 'RelativeNormDispIncr',
          6: 'NormUnbalance'}
    algo = {1: 'KrylovNewton', 2: 'SecantNewton', 4: 'RaphsonNewton', 5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden',
            8: 'NewtonLineSearch'}
    for k in te:
        for l in algo:
            if ok != 0:
                if l < 4:
                    ops.algorithm(algo[l], '-initial')

                else:
                    ops.algorithm(algo[l], '-initial')

                ops.test(te[k], Tol, 200)
                ok = ops.analyze(nPts, dt)

                print(te[k], algo[l], ok)
                if ok == 0:
                    break
            else:
                continue

    osv.anim_defo(1, 0, 1)
    # osv.plot_extruded_shapes_3d(ele_shapes, az_el=(45.0, 45.0), ax=False)
    # ax = osv.plot_mode_shape(1, unDefoFlag=0, endDispFlag=0)
    plt.xlabel('Length')
    plt.ylabel('Height')
    # Turn off the x-axis and y-axis tick labels
    # ax.set_xticks([])  # Hide x-axis tick labels
    # ax.set_yticks([])  # Hide y-axis tick labels
    # ax.set_zticks([])  # Hide z-axis tick labels
    # ax.view_init(elev=125, azim=-45, roll=45)  # Adjust the angles as needed
    plt.show()


NStory = 4  # number of stories above ground level
NBay = 6  # number of bays in X
NBayZ = 3  # number of bays in Z
LCol = 3657.6 * mm  # column height Y axis)
LBeam = 6096 * mm  # beam length X axis
LGird = 6096 * mm  # girder length Z axis
HCol = 711.2 * mm  # square-Column width
BCol = HCol
HBeam = 609.6 * mm  # Beam depth -- perpendicular to bending axis
BBeam = 457.2 * mm  # Beam width -- parallel to bending axis
HGird = 609.6 * mm  # Girder depth -- perpendicular to bending axis
BGird = 457.2 * mm  # Girder width -- parallel to bending axis
cover = 63.5 * mm
numBarsTopCol = 8  # number of reinforcement bars on top layer
numBarsBotCol = 8  # number of reinforcement bars on bottom layer
numBarsIntCol = 6  # number of reinforcement bars on intermediate layers
barAreaTopCol = 25.4 * 25.4 * mm2  #
barAreaBotCol = 25.4 * 25.4 * mm2  # reinforcement bar area (m2)
barAreaIntCol = 25.4 * 25.4 * mm2  #
numBarsTopBeam = 6  # number of reinforcement bars on top layer
numBarsBotBeam = 6  # number of reinforcement bars on bottom layer
numBarsIntBeam = 2  # number of reinforcement bars on intermediate layers
barAreaTopBeam = 25.4 * 25.4 * mm2  #
barAreaBotBeam = 25.4 * 25.4 * mm2  # reinforcement bar area (m2)
barAreaIntBeam = 25.4 * 25.4 * mm2  #
numBarsTopGird = 6  # number of reinforcement bars on top layer
numBarsBotGird = 6  # number of reinforcement bars on bottom layer
numBarsIntGird = 2  # number of reinforcement bars on the intermediate layers
barAreaTopGird = 25.4 * 25.4 * mm2  #
barAreaBotGird = 25.4 * 25.4 * mm2  # reinforcement bar area (m2)
barAreaIntGird = 25.4 * 25.4 * mm2  #
nfCoreY = 20  # number of fibers in the core patch in the y direction
nfCoreZ = 20  # number of fibers in the core patch in the z direction
nfCoverY = 20  # number of fibers in the cover patches with long sides in the y direction
nfCoverZ = 20  # number of fibers in the cover patches with long sides in the z direction
Tslab = 152.4 * mm  # slab thickness

GMdirection = 1
GMfolder = 'GM'
GMfile = 'SN1594_CHICHI_TTN051-E'
GMdirection = 1
GMfact = 1

building_model(NStory, NBay, NBayZ, LCol, LBeam, LGird, HCol, BCol,
               HBeam, BBeam, HGird, BGird, cover, numBarsTopCol, numBarsBotCol,
               numBarsIntCol, barAreaTopCol, barAreaBotCol, barAreaIntCol,
               numBarsTopBeam, numBarsBotBeam, numBarsIntBeam, barAreaTopBeam, barAreaBotBeam, barAreaIntBeam,
               numBarsTopGird, numBarsBotGird, numBarsIntGird, barAreaTopGird, barAreaBotGird, barAreaIntGird,
               nfCoreY, nfCoreZ, nfCoverY, nfCoverZ, Tslab, printProgression=True)

run_modal(3)

# Uniform Earthquake ground motion (uniform acceleration input at all support nodes)
GMfolder = 'GM'
GMdirection = 1  # ground-motion direction
GMfile = "SN1594_CHICHI_TTN051-E"  # ground-motion filenames
GMfact = 1.5  # ground-motion scaling factor

run_THA(GMfolder, GMfile, GMdirection, GMfact, plotResults=False, printProgression=False, recordData=False)
