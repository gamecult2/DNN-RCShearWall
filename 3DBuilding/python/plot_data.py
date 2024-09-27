"""
Original Authors of Tcl Scripts  : 
    Silvia Mazzoni & Frank McKenna, 2006
Tcl Script Translation to Python :
    elastropy.com, 2023
Purpose : 
    Example 7 - Data Plots
"""

# ===========================================================================
# Import Libraries
# ===========================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as npy

plt.close('all')
plt.rcParams['figure.dpi'] = 200

# ===========================================================================
# Main Code
# ===========================================================================

# Only Python Result Plot

freeDisp = npy.loadtxt('DataOut\DFree.out')
baseReaction = npy.loadtxt('DataOut\RBase.out')

freeDisp_tcl = npy.loadtxt(r'..\tcl\Data\DFree.out')
baseReaction_tcl = npy.loadtxt(r'..\tcl\Data\RBase.out')

dof_id = 1

plt.figure(1)
plt.plot(freeDisp[10:, dof_id], -baseReaction[10:, dof_id], 'b-', label='Python')
# plt.plot(freeDisp_tcl[10:, 1], -baseReaction_tcl[10:, 1], 'r-', label='Tcl')

plt.xlabel('Displacement (mm)')
plt.ylabel('Force (kN)')
plt.title('Force Displacement Response')
plt.legend()
plt.grid(True)
plt.show()

# Both Python and Tcl Result Plot
plt.figure(2)
plt.plot(freeDisp[10:, dof_id], -baseReaction[10:, dof_id], 'b-', label='Python')
plt.plot(freeDisp_tcl[10:, dof_id], -baseReaction_tcl[10:, dof_id], 'r-', label='Tcl')

plt.xlabel('Displacement (mm)')
plt.ylabel('Force (kN)')
plt.title('Force Displacement Response')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(freeDisp[:, 0], freeDisp[:, dof_id], 'b-', label='Python')
plt.plot(freeDisp_tcl[:, 0], freeDisp_tcl[:, dof_id], 'r-', label='Tcl')

plt.xlabel('Time (Sec)')
plt.ylabel('Displacement (mm)')
plt.title('Displacement Response')
plt.legend()
plt.grid(True)
plt.show()





