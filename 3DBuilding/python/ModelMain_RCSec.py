"""
Original Authors of Tcl Scripts  : 
    Silvia Mazzoni & Frank McKenna, 2006
Tcl Script Translation to Python :
    elastropy.com, 2023
Purpose : 
    Example 7 - 3D RC Frame with inelastic fiber section.
    Conversion of Tcl Script to OpenSeesPy Script
"""

# ===========================================================================
# Import Libraries
# ===========================================================================
import openseespy.opensees as os
import opsvis as osv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as npy
import pandas as pd
import os as os1
import time
import math
import shutil
import math as mt
from ReadSMDFile import ReadSMDFile

plt.rcParams['figure.dpi'] = 200


# ===========================================================================
# Main Code
# ===========================================================================

os.wipe()  # Cleans the existing database of nodes, materials, elements, recorders etc
os.model('basic', '-ndm', 3,'-ndf', 6)    # This is a 3D model with 3 translational and 3 rotational dofs per node (total 6 dofs)


from LibUnits import *               # Define units
from LibMaterialsRC import *         # Define RC materials 
from BuildRCrectSection import *     # Procedure for defining RC fiber section

dataDir = "DataOut"  
if os1.path.exists(dataDir):
    shutil.rmtree(dataDir) # This deletes the existing Data directory
os1.mkdir(dataDir)         # We use this directory to store the output data

GMdir = "../GM"

# ---------------------------------------------------------------------------
# Define Geometry
# ---------------------------------------------------------------------------

# define structure-geometry paramters
LCol  = 12*ft		# column height (parallel to Y axis)
LBeam = 20*ft		# beam length (parallel to X axis)
LGird = 20*ft		# girder length (parallel to Z axis)

# frame configuration
NStory = 3			# number of stories above ground level
NBay   = 1			# number of bays in X direction
NBayZ  = 1			# number of bays in Z direction

print('Number of Stories in Y: ', NStory, '\nNumber of bays in X: ', NBay, '\nNumber of bays in Z: ', NBayZ)


# define NODAL COORDINATES
# calculate locations of beam/column intersections:
X1 = 0.
X2 = X1 + LBeam
Y1 = 0.
Y2 = Y1 + LCol
Y3 = Y2 + LCol
Y4 = Y3 + LCol
Z1 = 0.0
Z2 = Z1 + LGird


os.node(111, X1, Y1, Z1)	# frame 1
os.node(112, X2, Y1, Z1)
os.node(121, X1, Y2, Z1)
os.node(122, X2, Y2, Z1)
os.node(131, X1, Y3, Z1)
os.node(132, X2, Y3, Z1)
os.node(141, X1, Y4, Z1)
os.node(142, X2, Y4, Z1)
os.node(211, X1, Y1, Z2)	# frame 2
os.node(212, X2, Y1, Z2)
os.node(221, X1, Y2, Z2)
os.node(222, X2, Y2, Z2)
os.node(231, X1, Y3, Z2)
os.node(232, X2, Y3, Z2)
os.node(241, X1, Y4, Z2)
os.node(242, X2, Y4, Z2)

# define Rigid Floor Diaphragm
Xa = (X2+X1)/2		        # mid-span coordinate for rigid diaphragm
Za = (Z2+Z1)/2

# rigid-diaphragm nodes in center of each diaphram
RigidDiaphragm = "ON"		    # this communicates to the analysis parameters that I will be using rigid diaphragms
os.node(1121, Xa, Y2, Za)		# master nodes for rigid diaphragm -- story 2, bay 1, frame 1-2
os.node(1131, Xa, Y3, Za)		# master nodes for rigid diaphragm -- story 3, bay 1, frame 1-2
os.node(1141, Xa, Y4, Za)		# master nodes for rigid diaphragm -- story 4, bay 1, frame 1-2

# Constraints for rigid diaphragm master nodes
os.fix(1121, 0,  1,  0,  1, 0,  1) # UX, UY=0, UZ, RX=0, RY, RZ=0
os.fix(1131, 0,  1,  0,  1, 0,  1) # UX, UY=0, UZ, RX=0, RY, RZ=0
os.fix(1141, 0,  1,  0,  1, 0,  1) # UX, UY=0, UZ, RX=0, RY, RZ=0

# define Rigid Diaphram, dof 2 is normal to floor
perpDirn = 2
os.rigidDiaphragm(perpDirn, 1121, *[121, 122, 221, 222]) # level 2
os.rigidDiaphragm(perpDirn, 1131, *[131, 132, 231, 232]) # level 3 
os.rigidDiaphragm(perpDirn, 1141, *[141, 142, 241, 242]) # level 4 


# determine support nodes where ground motions are input, 
# This setting is only for multiple-support excitation (earthquake analysis)
iSupportNode = [111, 112, 211, 212]

# BOUNDARY CONDITIONS
# fixY constrains the specified dofs of all nodes at a particluar level along Y-Direction
# this level is defined by the Y coordinate
os.fixY(0.0, *[1, 1, 1, 0, 1, 0])		# pin all Y=0.0 nodes


# calculated MODEL PARAMETERS, particular to this model
# Set up parameters that are particular to the model for displacement control 
IDctrlNode = 141		# node where displacement is applied for displacement control
IDctrlDOF  = 1			# degree of freedom of displacement applied for displacement control
LBuilding  = Y4		    # total building height


# Define SECTIONS 
SectionType = "FiberSection" 	

# define section tags:
ColSecTag   = 1
BeamSecTag  = 2
GirdSecTag  = 3
ColSecTagFiber  = 4
BeamSecTagFiber = 5
GirdSecTagFiber = 6
SecTagTorsion   = 70


# Section Properties:
HCol  = 28*inch		# square-Column width
BCol  = HCol
HBeam = 24*inch		# Beam depth -- perpendicualr to bending axis
BBeam = 18*inch		# Beam width -- parallel to bending axis
HGird = 24*inch		# Girder depth -- perpendicular to bending axis
BGird = 18*inch		# Girder width -- parallel to bending axis


# FIBER SECTION properties 
# Column section geometry:
cover = 2.5*inch	# rectangular-RC-Column cover
numBarsTopCol = 8		# number of longitudinal-reinforcement bars on top layer
numBarsBotCol = 8		# number of longitudinal-reinforcement bars on bottom layer
numBarsIntCol = 6		# TOTAL number of reinforcing bars on the intermediate layers
barAreaTopCol = 1.*inch*inch	# longitudinal-reinforcement bar area
barAreaBotCol = 1.*inch*inch	# longitudinal-reinforcement bar area
barAreaIntCol = 1.*inch*inch	# longitudinal-reinforcement bar area


numBarsTopBeam = 6		# number of longitudinal-reinforcement bars on top layer
numBarsBotBeam = 6		# number of longitudinal-reinforcement bars on bottom layer
numBarsIntBeam = 2		# TOTAL number of reinforcing bars on the intermediate layers
barAreaTopBeam = 1.*inch*inch	# longitudinal-reinforcement bar area
barAreaBotBeam = 1.*inch*inch	# longitudinal-reinforcement bar area
barAreaIntBeam = 1.*inch*inch	# longitudinal-reinforcement bar area

numBarsTopGird = 6		# number of longitudinal-reinforcement bars on top layer
numBarsBotGird = 6		# number of longitudinal-reinforcement bars on bottom layer
numBarsIntGird = 2		# TOTAL number of reinforcing bars on the intermediate layers
barAreaTopGird = 1.*inch*inch	# longitudinal-reinforcement bar area
barAreaBotGird = 1.*inch*inch	# longitudinal-reinforcement bar area
barAreaIntGird = 1.*inch*inch	# longitudinal-reinforcement bar area

nfCoreY  = 20		# number of fibers in the core patch in the y direction
nfCoreZ  = 20		# number of fibers in the core patch in the z direction
nfCoverY = 20		# number of fibers in the cover patches with long sides in the y direction
nfCoverZ = 20		# number of fibers in the cover patches with long sides in the z direction


# rectangular section with one layer of steel evenly distributed around the perimeter and a confined core.


BuildRCrectSection(ColSecTagFiber, HCol, BCol, cover, cover, IDconcCore,  
                   IDconcCover, IDSteel, numBarsTopCol, barAreaTopCol, 
                   numBarsBotCol, barAreaBotCol, numBarsIntCol, barAreaIntCol,  
                   nfCoreY, nfCoreZ, nfCoverY, nfCoverZ)

BuildRCrectSection(BeamSecTagFiber, HBeam, BBeam, cover, cover, IDconcCore,  
                   IDconcCover, IDSteel, numBarsTopBeam, barAreaTopBeam, 
                   numBarsBotBeam, barAreaBotBeam, numBarsIntBeam, barAreaIntBeam, 
                   nfCoreY, nfCoreZ, nfCoverY, nfCoverZ)  
                
BuildRCrectSection(GirdSecTagFiber, HGird, BGird, cover, cover, IDconcCore,  
                   IDconcCover, IDSteel, numBarsTopGird, barAreaTopGird, 
                   numBarsBotGird, barAreaBotGird, numBarsIntGird, barAreaIntGird,  
                   nfCoreY, nfCoreZ, nfCoverY, nfCoverZ)
	

# assign torsional Stiffness for 3D Model

os.uniaxialMaterial('Elastic', SecTagTorsion, Ubig)

os.section('Aggregator', ColSecTag,  *[SecTagTorsion, 'T'], '-section', ColSecTagFiber)
os.section('Aggregator', BeamSecTag, *[SecTagTorsion, 'T'], '-section', BeamSecTagFiber)
os.section('Aggregator', GirdSecTag, *[SecTagTorsion, 'T'], '-section', GirdSecTagFiber)


GammaConcrete = 150*pcf   			# Reinforced-Concrete weight density (weight per volume)
print(pcf)
print('GammaConcrete kip/in3', GammaConcrete)
pcfvalue = 1*pcf   			# Reinforced-Concrete weight density (weight per volume)
print('pcfvalue', pcfvalue)
QdlCol = GammaConcrete*HCol*BCol	# self weight of Column, weight per length
print('QdlCol', QdlCol)
QBeam  = GammaConcrete*HBeam*BBeam	# self weight of Beam, weight per length
QGird  = GammaConcrete*HGird*BGird	# self weight of Gird, weight per length
print('QGird in kip/in', QGird)


# define ELEMENTS 
# set up geometric transformations of element
#   separate columns and beams, in case of P-Delta analysis for columns
#   in 3D model, assign vector vecxz
IDColTransf  = 1 # all columns
IDBeamTransf = 2 # all beams
IDGirdTransf = 3 # all girders
ColTransfType = "Linear" 			# options, "Linear" or "PDelta" or "Corotational" 


os.geomTransf(ColTransfType, IDColTransf,  *[0, 0, 1])  	# only columns can have PDelta effects (gravity effects)
os.geomTransf('Linear', IDBeamTransf, *[0, 0, 1]) 
os.geomTransf('Linear', IDGirdTransf, *[1, 0, 0]) 

# Define Beam-Column Elements
np = 5	# number of Gauss integration points for nonlinear curvature distribution



# Frame 1
# columns
os.element('nonlinearBeamColumn', 1111, *[111, 121], np, ColSecTag, IDColTransf)		# level 1-2
os.element('nonlinearBeamColumn', 1112, *[112, 122], np, ColSecTag, IDColTransf)
os.element('nonlinearBeamColumn', 1121, *[121, 131], np, ColSecTag, IDColTransf)		# level 2-3
os.element('nonlinearBeamColumn', 1122, *[122, 132], np, ColSecTag, IDColTransf)
os.element('nonlinearBeamColumn', 1131, *[131, 141], np, ColSecTag, IDColTransf)		# level 3-4
os.element('nonlinearBeamColumn', 1132, *[132, 142], np, ColSecTag, IDColTransf)
# beams
os.element('nonlinearBeamColumn', 1221, *[121, 122], np, BeamSecTag, IDBeamTransf)	    # level 2
os.element('nonlinearBeamColumn', 1231, *[131, 132], np, BeamSecTag, IDBeamTransf)	    # level 3
os.element('nonlinearBeamColumn', 1241, *[141, 142], np, BeamSecTag, IDBeamTransf)	    # level 4


# Frame 2
# columns
os.element('nonlinearBeamColumn', 2111, *[211, 221], np, ColSecTag, IDColTransf)		# level 1-2
os.element('nonlinearBeamColumn', 2112, *[212, 222], np, ColSecTag, IDColTransf)
os.element('nonlinearBeamColumn', 2121, *[221, 231], np, ColSecTag, IDColTransf)		# level 2-3
os.element('nonlinearBeamColumn', 2122, *[222, 232], np, ColSecTag, IDColTransf)
os.element('nonlinearBeamColumn', 2131, *[231, 241], np, ColSecTag, IDColTransf)		# level 3-4
os.element('nonlinearBeamColumn', 2132, *[232, 242], np, ColSecTag, IDColTransf)
# beams
os.element('nonlinearBeamColumn', 2221, *[221, 222], np, BeamSecTag, IDBeamTransf)	# level 2
os.element('nonlinearBeamColumn', 2231, *[231, 232], np, BeamSecTag, IDBeamTransf)	# level 3
os.element('nonlinearBeamColumn', 2241, *[241, 242], np, BeamSecTag, IDBeamTransf)	# level 4


# girders connecting frames
# Frame 1-2
os.element('nonlinearBeamColumn', 1321, *[121, 221], np, GirdSecTag, IDGirdTransf)	# level 2
os.element('nonlinearBeamColumn', 1322, *[122, 222], np, GirdSecTag, IDGirdTransf)
os.element('nonlinearBeamColumn', 1331, *[131, 231], np, GirdSecTag, IDGirdTransf)	# level 3
os.element('nonlinearBeamColumn', 1332, *[132, 232], np, GirdSecTag, IDGirdTransf)
os.element('nonlinearBeamColumn', 1341, *[141, 241], np, GirdSecTag, IDGirdTransf)	# level 4
os.element('nonlinearBeamColumn', 1342, *[142, 242], np, GirdSecTag, IDGirdTransf)




# ---------------------------------------------------------------------------
# Define GRAVITY LOADS, weight and masses
# ---------------------------------------------------------------------------

# calculate dead load of frame, assume this to be an internal frame (do LL in a similar manner)
# calculate distributed weight along the beam length
Tslab = 6*inch			# 6-inch slab
Lslab = LGird/2 			# slab extends a distance of LGird/2 in/out of plane
DLfactor = 1.0				# scale dead load up a little
Qslab = GammaConcrete*Tslab*Lslab*DLfactor 
QdlBeam = Qslab + QBeam 	# dead load distributed along beam (one-way slab)
QdlGird = QGird 			# dead load distributed along girder
WeightCol = QdlCol*LCol  		# total Column weight
WeightBeam = QdlBeam*LBeam 	# total Beam weight
WeightGird = QdlGird*LGird 	# total Beam weight
print('LCol', LCol)
print('LBeam', LBeam)
print('LGird', LGird)
print('WeightCol', WeightCol)
print('WeightBeam', WeightBeam)
print('WeightGird', WeightGird)

# assign masses to the nodes that the columns are connected to 
# each connection takes the mass of 1/2 of each element framing into it (mass=weight/g)
Mmid  = (WeightCol/2 + WeightCol/2 +WeightBeam/2+WeightGird/2)/g
Mtop  = (WeightCol/2 + WeightBeam/2+WeightGird/2)/g


# frame 1
os.mass(121, *[Mmid, 0, Mmid, 0., 0., 0.])		# level 2
os.mass(122, *[Mmid, 0, Mmid, 0., 0., 0.])
os.mass(131, *[Mmid, 0, Mmid, 0., 0., 0.])		# level 3
os.mass(132, *[Mmid, 0, Mmid, 0., 0., 0.])
os.mass(141, *[Mtop, 0, Mtop, 0., 0., 0.])		# level 4
os.mass(142, *[Mtop, 0, Mtop, 0., 0., 0.])

# frame 2
os.mass(221, *[Mmid, 0, Mmid, 0., 0., 0.])		# level 2
os.mass(222, *[Mmid, 0, Mmid, 0., 0., 0.])
os.mass(231, *[Mmid, 0, Mmid, 0., 0., 0.])		# level 3
os.mass(232, *[Mmid, 0, Mmid, 0., 0., 0.])
os.mass(241, *[Mtop, 0, Mtop, 0., 0., 0.])		# level 4
os.mass(242, *[Mtop, 0, Mtop, 0., 0., 0.])

FloorWeight2 = 4*WeightCol + 2*WeightGird + 2*WeightBeam
FloorWeight3 = 4*WeightCol + 2*WeightGird + 2*WeightBeam
FloorWeight4 = 2*WeightCol + 2*WeightGird + 2*WeightBeam

print('FloorWeight2', FloorWeight2)
print('FloorWeight3', FloorWeight3)
print('FloorWeight4', FloorWeight4)

WeightTotal  = FloorWeight2+FloorWeight3+FloorWeight4			# total building weight
MassTotal    = WeightTotal/g							        # total building mass
print('WeightTotal', WeightTotal)
print('MassTotal', MassTotal)

# ---------------------------------------------------------------------------
# LATERAL-LOAD distribution for static pushover analysis
# ---------------------------------------------------------------------------

# calculate distribution of lateral load based on mass/weight distributions along building height
# Fj = WjHj/sum(WiHi)  * Weight   at each floor j
sumWiHi = FloorWeight2*Y2+FloorWeight3*Y3+FloorWeight4*Y4 		# sum of storey weight times height, for lateral-load distribution
WiHi2   = FloorWeight2*Y2 		# storey weight times height, for lateral-load distribution
WiHi3   = FloorWeight3*Y3 		# storey weight times height, for lateral-load distribution
WiHi4   = FloorWeight4*Y4 		# storey weight times height, for lateral-load distribution
F2      = WiHi2/sumWiHi*WeightTotal	# lateral load at level
F3      = WiHi3/sumWiHi*WeightTotal	# lateral load at level
F4      = WiHi4/sumWiHi*WeightTotal	# lateral load at level


# ---------------------------------------------------------------------------
# Define RECORDERS
# ---------------------------------------------------------------------------

os.recorder('Node', '-file', f"{dataDir}/DFree.out", '-time', '-node', *[141],                '-dof', *[1, 2, 3], 'disp');			# displacements of free node
os.recorder('Node', '-file', f"{dataDir}/DBase.out", '-time', '-node', *[111, 112, 211, 212], '-dof', *[1, 2, 3], 'disp')		# displacements of support nodes
os.recorder('Node', '-file', f"{dataDir}/RBase.out", '-time', '-node', *[111, 112, 211, 212], '-dof', *[1, 2, 3], 'reaction')		# support reaction

os.recorder('Element', '-file', f"{dataDir}/Fel1.out",             '-time', '-ele', *[1111],  'localForce')				# element forces in local coordinates
os.recorder('Element', '-xml',  f"{dataDir}/PlasticRotation1.out", '-time', '-ele', *[1111],  'plasticRotation')				# element forces in local coordinates
os.recorder('Element', '-file', f"{dataDir}/ForceEle1sec1.out",    '-time', '-ele', *[1111],  'section', 1,  'force')			# section forces, axial and moment, node i
os.recorder('Element', '-file', f"{dataDir}/DefoEle1sec1.out",     '-time', '-ele', *[11111], 'section', 1,  'deformation')		# section deformations, axial and curvature, node i
os.recorder('Element', '-file', f"{dataDir}/ForceEle1secnp.out",  '-time', '-ele',  *[111],   'section', np, 'force')			# section forces, axial and moment, node j
os.recorder('Element', '-file', f"{dataDir}/DefoEle1secnp.out",   '-time', '-ele',  *[1111],  'section', np, 'deformation')		# section deformations, axial and curvature, node j


yFiber = HCol/2-cover								# fiber location for stress-strain recorder, local coords
zFiber = BCol/2-cover								# fiber location for stress-strain recorder, local coords

os.recorder('Element', '-file', f"{dataDir}/SSconcEle1sec1.out",  '-time', '-ele', *[1111], 'section', np, 'fiber', yFiber, zFiber, IDconcCore, 'stressStrain')	# steel fiber stress-strain, node i
os.recorder('Element', '-file', f"{dataDir}/SSreinfEle1sec1.out", '-time', '-ele', *[1111], 'section', np, 'fiber', yFiber, zFiber, IDSteel,    'stressStrain')	# steel fiber stress-strain, node i



#%%
# ---------------------------------------------------------------------------
# Gravity Analysis
# ---------------------------------------------------------------------------
# GRAVITY LOADS # define gravity load applied to beams and columns -- 	eleLoad applies loads in local coordinate axis

tsTagGravity = 101
patternTagGravity  = 101              
os.timeSeries("Linear", tsTagGravity)   # Linear time series (syntax - timeSeries(tsType, tsTag, *tsArgs))
os.pattern("Plain", patternTagGravity, tsTagGravity)    # Plain pattern (syntax - pattern(patternType, patternTag, *patternArgs))


# Frame 1
# columns
os.eleLoad('-ele', *[1111], '-type', '-beamUniform', 0., 0., -QdlCol)      # level 1-2
os.eleLoad('-ele', *[1112], '-type', '-beamUniform', 0., 0., -QdlCol)
os.eleLoad('-ele', *[1121], '-type', '-beamUniform', 0., 0., -QdlCol)      # level 2-3
os.eleLoad('-ele', *[1122], '-type', '-beamUniform', 0., 0., -QdlCol)
os.eleLoad('-ele', *[1131], '-type', '-beamUniform', 0., 0., -QdlCol)      # level 3-4
os.eleLoad('-ele', *[1132], '-type', '-beamUniform', 0., 0., -QdlCol)


# beams
os.eleLoad('-ele', *[1221], '-type', '-beamUniform', -QdlBeam, 0.)    # level 2
os.eleLoad('-ele', *[1231], '-type', '-beamUniform', -QdlBeam, 0.)    # level 3
os.eleLoad('-ele', *[1241], '-type', '-beamUniform', -QdlBeam, 0.)    # level 4

# Frame 2
# columns
os.eleLoad('-ele', *[2111], '-type', '-beamUniform', 0., 0., -QdlCol)      # level 1-2
os.eleLoad('-ele', *[2112], '-type', '-beamUniform', 0., 0., -QdlCol)
os.eleLoad('-ele', *[2121], '-type', '-beamUniform', 0., 0., -QdlCol)      # level 2-3
os.eleLoad('-ele', *[2122], '-type', '-beamUniform', 0., 0., -QdlCol)
os.eleLoad('-ele', *[2131], '-type', '-beamUniform', 0., 0., -QdlCol)      # level 3-4
os.eleLoad('-ele', *[2132], '-type', '-beamUniform', 0., 0., -QdlCol)

# beams
os.eleLoad('-ele', *[2221], '-type', '-beamUniform', -QdlBeam, 0.)    # level 2
os.eleLoad('-ele', *[2231], '-type', '-beamUniform', -QdlBeam, 0.)    # level 3
os.eleLoad('-ele', *[2241], '-type', '-beamUniform', -QdlBeam, 0.)    # level 4

# girders connecting frames
# Frame 1-2
os.eleLoad('-ele', *[1321], '-type', '-beamUniform', -QdlGird, 0.)		# level 2
os.eleLoad('-ele', *[1322], '-type', '-beamUniform', -QdlGird, 0.)
os.eleLoad('-ele', *[1331], '-type', '-beamUniform', -QdlGird, 0.)		# level 3
os.eleLoad('-ele', *[1332], '-type', '-beamUniform', -QdlGird, 0.)
os.eleLoad('-ele', *[1341], '-type', '-beamUniform', -QdlGird, 0.)		# level 4
os.eleLoad('-ele', *[1342], '-type', '-beamUniform', -QdlGird, 0.)


# Gravity-analysis parameters -- load-controlled static analysis
Tol = 1.0e-8			# convergence tolerance for test
constraintsTypeGravity = "Plain"  # default

if 'RigidDiaphragm' in locals() or 'RigidDiaphragm' in globals():  # Check if a variable named RigidDiaphragm exxists in either local or global varaibles
    if RigidDiaphragm == "ON":
        constraintsTypeGravity = "Lagrange"  # large model: try Transformation

os.constraints(constraintsTypeGravity)      # Contrain handler, this handles how the comstraints are enforced at a dof
os.numberer("RCM")                # Scheme to define the numbering for degree of freedoms. This command also sets mapping between equation numbers and degree of freedoms
os.system('BandGeneral')          # defines the system of equations
os.test('EnergyIncr', Tol, 6)     # Convergence criteria at each iteration
                                  # 6 is the maximum number of iterations to check before returning failure status
os.algorithm('Newton')            #  Newton: this updates tangent stiffness at every iteration


NstepGravity = 10
DGravity = 1./NstepGravity
os.integrator('LoadControl', DGravity)        #  0.1 is the load increment -this applies gravity in 10 steps

os.analysis('Static')     #  define type of analysis static or transient
gravityAnalysisStatus  = os.analyze(NstepGravity)     

if (gravityAnalysisStatus == 0):
    print('Gravity Analysis Successfull')
else:
    print('Gravity Analysis Failed')


os.loadConst('-time', 0.0)
Tol = 1.0e-6			# reduce tolerance after gravity loads
print('Model Built Succesfully')
os.wipeAnalysis()


#%%
# ===========================================================================
# Earthquake Analysis Code
# ===========================================================================

# Uniform Earthquake ground motion (uniform acceleration input at all support nodes)
GMdirection = 1				# ground-motion direction
GMfile = "SN1594_CHICHI_TTN051-E" 		# ground-motion filenames
GMfact = 1.5				# ground-motion scaling factor

# set up ground-motion-analysis parameters
DtAnalysis	    = 0.01*sec	    # time-step Dt for lateral analysis
TmaxAnalysis	= 39.0*sec # 10.0*sec	# maximum duration of ground-motion analysis -- should be 50*$sec



# dynamic-analysis parameters

# Set up Analysis Parameters 
# CONSTRAINTS handler -- Determines how the constraint equations are enforced in the analysis (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/617.htm)
#          Plain Constraints -- Removes constrained degrees of freedom from the system of equations 
#          Lagrange Multipliers -- Uses the method of Lagrange multipliers to enforce constraints 
#          Penalty Method -- Uses penalty numbers to enforce constraints 
#          Transformation Method -- Performs a condensation of constrained degrees of freedom 
constraintsTypeDynamic = "Transformation"
os.constraints(constraintsTypeDynamic)

# DOF NUMBERER (number the degrees of freedom in the domain): (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/366.htm)
#   determines the mapping between equation numbers and degrees-of-freedom
#          Plain -- Uses the numbering provided by the user 
#          RCM -- Renumbers the DOF to minimize the matrix band-width using the Reverse Cuthill-McKee algorithm 
numbererTypeDynamic = "RCM"
os.numberer(numbererTypeDynamic) 

# SYSTEM (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/371.htm)
#   Linear Equation Solvers (how to store and solve the system of equations in the analysis)
#   -- provide the solution of the linear system of equations Ku = P. Each solver is tailored to a specific matrix topology. 
#          ProfileSPD -- Direct profile solver for symmetric positive definite matrices 
#          BandGeneral -- Direct solver for banded unsymmetric matrices 
#          BandSPD -- Direct solver for banded symmetric positive definite matrices 
#          SparseGeneral -- Direct solver for unsymmetric sparse matrices (-piv option)
#          SparseSPD -- Direct solver for symmetric sparse matrices 
#          UmfPack -- Direct UmfPack solver for unsymmetric matrices 
systemTypeDynamic = "BandGeneral"	# try UmfPack for large problems
os.system(systemTypeDynamic) 

# TEST: # convergence test to 
# Convergence TEST (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/360.htm)
#   -- Accept the current state of the domain as being on the converged solution path 
#   -- determine if convergence has been achieved at the end of an iteration step
#          NormUnbalance -- Specifies a tolerance on the norm of the unbalanced load at the current iteration 
#          NormDispIncr -- Specifies a tolerance on the norm of the displacement increments at the current iteration 
#          EnergyIncr-- Specifies a tolerance on the inner product of the unbalanced load and displacement increments at the current iteration 
#          RelativeNormUnbalance --
#          RelativeNormDispIncr --
#          RelativeEnergyIncr --
TolDynamic = 1.e-8                        # Convergence Test: tolerance
maxNumIterDynamic = 10                # Convergence Test: maximum number of iterations that will be performed before "failure to converge" is returned
printFlagDynamic = 0                # Convergence Test: flag used to print information on convergence (optional)        # 1: print information on each step; 
testTypeDynamic = "EnergyIncr"	# Convergence-test type
os.test(testTypeDynamic, TolDynamic, maxNumIterDynamic, printFlagDynamic)
# for improved-convergence procedure:
maxNumIterConvergeDynamic = 2000	
printFlagConvergeDynamic = 0	

# Solution ALGORITHM: -- Iterate from the last time step to the current (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/682.htm)
#          Linear -- Uses the solution at the first iteration and continues 
#          Newton -- Uses the tangent at the current iteration to iterate to convergence 
#          ModifiedNewton -- Uses the tangent at the first iteration to iterate to convergence 
#          NewtonLineSearch -- 
#          KrylovNewton -- 
#          BFGS -- 
#          Broyden -- 
algorithmTypeDynamic = "ModifiedNewton" 
os.algorithm(algorithmTypeDynamic)        

# Static INTEGRATOR: -- determine the next time step for an analysis  (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/689.htm)
#          LoadControl -- Specifies the incremental load factor to be applied to the loads in the domain 
#          DisplacementControl -- Specifies the incremental displacement at a specified DOF in the domain 
#          Minimum Unbalanced Displacement Norm -- Specifies the incremental load factor such that the residual displacement norm in minimized 
#          Arc Length -- Specifies the incremental arc-length of the load-displacement path 
# Transient INTEGRATOR: -- determine the next time step for an analysis including inertial effects 
#          Newmark -- The two parameter time-stepping method developed by Newmark 
#          HHT -- The three parameter Hilbert-Hughes-Taylor time-stepping method 
#          Central Difference -- Approximates velocity and acceleration by centered finite differences of displacement 
NewmarkGamma = 0.5	# Newmark-integrator gamma parameter (also HHT)
NewmarkBeta = 0.25	# Newmark-integrator beta parameter
integratorTypeDynamic = "Newmark"
os.integrator(integratorTypeDynamic, NewmarkGamma, NewmarkBeta)

# ANALYSIS  -- defines what type of analysis is to be performed (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/324.htm)
#          Static Analysis -- solves the KU=R problem, without the mass or damping matrices. 
#          Transient Analysis -- solves the time-dependent analysis. The time step in this type of analysis is constant. The time step in the output is also constant. 
#          variableTransient Analysis -- performs the same analysis type as the Transient Analysis object. The time step, however, is variable. This method is used when 
#                 there are convergence problems with the Transient Analysis object at a peak or when the time step is too small. The time step in the output is also variable.
analysisTypeDynamic = "Transient"
os.analysis(analysisTypeDynamic) 



# Define & apply damping
# RAYLEIGH damping parameters, Where to put M/K-prop damping, switches (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/1099.htm)
#          D=$alphaM*M + $betaKcurr*Kcurrent + $betaKcomm*KlastCommit + $beatKinit*$Kinitial
xDamp = 0.02					# damping ratio
MpropSwitch = 1.0
KcurrSwitch = 0.0
KcommSwitch = 1.0
KinitSwitch = 0.0
nEigenI = 1		# mode 1
nEigenJ = 3		# mode 3
lambdaN = os.eigen(nEigenJ)			# eigenvalue analysis for nEigenJ modes
print(lambdaN)
lambdaI = lambdaN[nEigenI - 1] 		# eigenvalue mode i
lambdaJ = lambdaN[nEigenJ - 1] 	    # eigenvalue mode j
omegaI = pow(lambdaI,0.5)
omegaJ = pow(lambdaJ,0.5)
alphaM = MpropSwitch*xDamp*(2*omegaI*omegaJ)/(omegaI+omegaJ)	# M-prop. damping; D = alphaM*M
betaKcurr = KcurrSwitch*2.*xDamp/(omegaI+omegaJ)         		# current-K;      +beatKcurr*KCurrent
betaKcomm = KcommSwitch*2.*xDamp/(omegaI+omegaJ)   		# last-committed K;   +betaKcomm*KlastCommitt
betaKinit = KinitSwitch*2.*xDamp/(omegaI+omegaJ)         			# initial-K;     +beatKinit*Kini
os.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm) 				# RAYLEIGH damping

xDamp = 0.05  # 5% damping ratio
num_modes = 3

mode_i = 0  # mode 1
mode_j = 2  # mode 3
lambda_values = os.eigen(num_modes)
omegas = []  # List to store omega values
frequencies = []
periods = []
for lam in lambda_values:
    omega = math.sqrt(lam)
    omegas.append(omega)  # Store omega value
    frequencies.append(omega / (2 * math.pi))  # Calculate frequency
    periods.append((2 * math.pi) / omega)  # Calculate period
print(omegas)
print("Mode  | Frequency (rad/s) | Period (s)")
print("-------|-----------------|------------")
for i, mode in enumerate(range(1, num_modes + 1)):
    print(f"{mode:3d}  | {frequencies[i]:.4f} | {periods[i]:.4f}")
#  perform Dynamic Ground-Motion Analysis
# the following commands are unique to the Uniform Earthquake excitation
IDloadTag = 400	 # for uniformSupport excitation
# Uniform EXCITATION: acceleration input
inFile = f"{GMdir}/{GMfile}.at2"
outFile = f"{GMdir}/{GMfile}.g3"	# set variable holding new filename (PEER files have .at2/dt2 extension)

dt = [0.0]   # Initialisation, this will be updated in ReadSMDFile
ReadSMDFile(inFile, outFile, dt)		# call procedure to convert the ground-motion file
GMfatt = g*GMfact		# data in input file is in g Unifts -- ACCELERATION TH

tsTagEq = 2
os.timeSeries('Path', tsTagEq, '-dt', dt[0], '-filePath', outFile, '-factor', GMfatt) 
os.pattern('UniformExcitation', IDloadTag, GMdirection, '-accel', tsTagEq)


Nsteps = int(TmaxAnalysis / DtAnalysis)
ok = os.analyze(Nsteps, DtAnalysis)

if ok != 0:  # analysis was not successful.
    # --------------------------------------------------------------------------------------------------
    # change some analysis parameters to achieve convergence
    # performance is slower inside this loop
    #    Time-controlled analysis
    ok = 0
    controlTime = os.getTime()
    
    while controlTime < TmaxAnalysis and ok == 0:
        controlTime = os.getTime()
        ok = os.analyze(1, DtAnalysis)
        
        if ok != 0:
            print("Trying Newton with Initial Tangent ..")
            os.test('NormDispIncr', Tol, 1000, 0)
            os.algorithm('Newton', '-initial')
            ok = os.analyze(1, DtAnalysis)
            os.test(testTypeDynamic, TolDynamic, maxNumIterDynamic, 0)
            os.algorithm(algorithmTypeDynamic)
        
        if ok != 0:
            print("Trying Broyden ..")
            os.algorithm('Broyden', 8)
            ok = os.analyze(1, DtAnalysis)
            os.algorithm(algorithmTypeDynamic)
        
        if ok != 0:
            print("Trying NewtonWithLineSearch ..")
            os.algorithm('NewtonLineSearch', 0.8)
            ok = os.analyze(1, DtAnalysis)
            os.algorithm(algorithmTypeDynamic)

print(f"Ground Motion Done. End Time: {os.getTime()}")






#%% Model Visualisations

ax = osv.plot_model(node_labels=0, element_labels=0,gauss_points=False, local_axes=False, axis_off=0)
plt.xlabel('Length')
plt.ylabel('Height')
# plt.title('Undeformed Model')

# Turn off the x-axis and y-axis tick labels
# ax.set_xticks([])  # Hide x-axis tick labels
# ax.set_yticks([])  # Hide y-axis tick labels
# ax.set_zticks([])  # Hide z-axis tick labels

# Turn off grid lines
# ax.grid(False)

# Invert the y-axis to make it vertical
# Adjust the view perspective to make y vertical and z horizontal
ax.view_init(elev=125, azim=-45, roll = 45, vertical_axis='z')  # Adjust the angles as needed


plt.show()
ax.cla()  


