# ===========================================================================
# Import Libraries
# ===========================================================================
import openseespy.opensees as ops
import opsvis as osv
import matplotlib.pyplot as plt


def BuildRCrectSection(id, HSec, BSec, coverH, coverB, coreID, coverID, steelID,
                       numBarsTop, barAreaTop, numBarsBot, barAreaBot,
                       numBarsIntTot, barAreaInt, nfCoreY, nfCoreZ, nfCoverY, nfCoverZ):

    """
    Builds a fiber rectangular RC section with one steel layer top, 1 bot, 1 skin, confined core.

    Args:
      id: Tag for the generated section.
      HSec: Depth of section along local-y axis.
      BSec: Width of section along local-z axis.
      coverH: Distance from section boundary to neutral axis of reinforcement (top and bottom).
      coverB: Distance from section boundary to side of reinforcement.
      coreID: Material tag for the core patch.
      coverID: Material tag for the cover patches.
      steelID: Material tag for the reinforcing steel.
      numBarsTop: Number of reinforcing bars in the top layer.
      barAreaTop: Cross-sectional area of each bar in top layer.
      numBarsBot: Number of reinforcing bars in the bottom layer.
      barAreaBot: Cross-sectional area of each bar in bottom layer.
      numBarsIntTot: Total number of reinforcing bars on intermediate layers (symmetric about z-axis, 2 bars per layer).
      barAreaInt: Cross-sectional area of each bar in intermediate layer.
      nfCoreY: Number of fibers in the core patch in the y direction.
      nfCoreZ: Number of fibers in the core patch in the z direction.
      nfCoverY: Number of fibers in the cover patches with long sides in the y direction.
      nfCoverZ: Number of fibers in the cover patches with long sides in the z direction.
    """
    G = 1 * 1e12
    J = 1.0  # Torsional section stiffness factor
    GJ = G * J

    # Calculate inner core dimensions
    coverY = HSec / 2.0             # The distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
    coverZ = BSec / 2.0             # The distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
    coreY = coverY - coverH         # The distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
    coreZ = coverZ - coverB         # The distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
    numBarsInt = int(numBarsIntTot / 2)  # number of intermediate bars per side

    # Define the fiber section
    ops.section('Fiber', id, '-GJ', GJ)

    # Define core patch
    ops.patch('quad', coreID, nfCoreZ, nfCoreY, *[-coreY, coreZ], *[-coreY, -coreZ], *[coreY, -coreZ], *[coreY, coreZ])
    # print('quad', coreID, nfCoreZ, nfCoreY, -coreY, coreZ, -coreY, -coreZ, coreY, -coreZ, coreY, coreZ)

    # Define cover patches
    ops.patch('quad', coverID, 2, nfCoverY, *[-coverY, coverZ], *[-coreY, coreZ], *[coreY, coreZ], *[coverY, coverZ])
    ops.patch('quad', coverID, 2, nfCoverY, *[-coreY, -coreZ], *[-coverY, -coverZ], *[coverY, -coverZ], *[coreY, -coreZ])
    ops.patch('quad', coverID, nfCoverZ, 2, *[-coverY, coverZ], *[-coverY, -coverZ], *[-coreY, -coreZ], *[-coreY, coreZ])
    ops.patch('quad', coverID, nfCoverZ, 2, *[coreY, coreZ], *[coreY, -coreZ], *[coverY, -coverZ], *[coverY, coverZ])

    # Define steel layers
    # print('straight', steelID, numBarsInt, barAreaInt, -coreY, coreZ, coreY, coreZ)
    ops.layer('straight', steelID, numBarsInt, barAreaInt, *[-coreY, coreZ], *[coreY, coreZ])        # Intermediate skin +z
    ops.layer('straight', steelID, numBarsInt, barAreaInt, *[-coreY, -coreZ], *[coreY, -coreZ])      # Intermediate skin -z
    ops.layer('straight', steelID, numBarsTop, barAreaTop, *[coreY, coreZ], *[coreY, -coreZ])        # Top layer reinforcement
    ops.layer('straight', steelID, numBarsBot, barAreaBot, *[-coreY, coreZ], *[-coreY, -coreZ])      # Bottom layer reinforcement

    fibSec = [['section', 'Fiber', id+10, '-GJ', 1e-10],
              # Define the core patch
              ['patch', 'quadr', coreID, nfCoreZ, nfCoreY, *[-coreY, coreZ], *[-coreY, -coreZ], *[coreY, -coreZ], *[coreY, coreZ]],

              # Define the four cover patches
              ['patch', 'quadr', coverID, 2, nfCoverY, *[-coverY, coverZ], *[-coreY, coreZ], *[coreY, coreZ], *[coverY, coverZ]],
              ['patch', 'quadr', coverID, 2, nfCoverY, *[-coreY, -coreZ], *[-coverY, -coverZ], *[coverY, -coverZ], *[coreY, -coreZ]],
              ['patch', 'quadr', coverID, nfCoverZ, 2, *[-coverY, coverZ], *[-coverY, -coverZ], *[-coreY, -coreZ], *[-coreY, coreZ]],
              ['patch', 'quadr', coverID, nfCoverZ, 2, *[coreY, coreZ], *[coreY, -coreZ], *[coverY, -coverZ], *[coverY, coverZ]],

              # define reinforcing layers
              ['layer', 'straight', steelID, numBarsInt, barAreaInt, *[-coreY, coreZ], *[coreY, coreZ]],  # intermediate skin reinf. +z
              ['layer', 'straight', steelID, numBarsInt, barAreaInt, *[-coreY, -coreZ], *[coreY, -coreZ]],  # intermediate skin reinf. -z
              ['layer', 'straight', steelID, numBarsTop, barAreaTop, *[coreY, coreZ], *[coreY, -coreZ]],  # top layer reinfocement
              ['layer', 'straight', steelID, numBarsBot, barAreaBot, *[-coreY, coreZ], *[-coreY, -coreZ]]]  # bottom layer reinforcement

    osv.fib_sec_list_to_cmds(fibSec)  # This command converts the opsvis list to openseespy section object

    # osv.plot_fiber_section(fibSec)
    # plt.axis('equal')




