import sys

sys.path.append('Result_files')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BackUp import Module as MO

# Unite
m = 1.0  # Meters
KN = 1.0  # KiloNewtons
sec = 1.0  # Seconds

mm = 0.001 * m  # Milimeters
cm = 0.01 * m  # Centimeters
ton = KN * (sec ** 2) / m  # mass unit (derived)
g = 9.81 * (m / sec ** 2)  # gravitational constant (derived)
MPa = 1e3 * (KN / m ** 2)  # Mega Pascal
GPa = 1e6 * (KN / m ** 2)  # Giga Pascal

# Geometric Dimensions
L_X = 6.0 * m  # Span in X-direction
L_Y = 7.0 * m  # Span in Y-direction
L_Z = 3.5 * m  # Story height

# Material Definition
f_c_1 = -25 * MPa  # f'c in compression for unconfined concrete
f_c_2 = -28 * MPa  # f'c in compression for confined concrete
eps_c = -0.002  # strain at maximum stress in compression
eps_u = -0.02  # strain at ultimate stress in compression
f_y = 420.0 * MPa  # fy for reinforcing steel
E_s = 210.0 * GPa  # E for reinforcing steel

# Section Definition
rebar = 0.25 * np.pi * (25 * mm) ** 2

# uniaxial Kent-Scott-Park concrete material with degraded linear unloading/reloading
mat_KSP_unconf = {'ID': 'Concrete02',
                  'matTag': 1,
                  'fpc': f_c_1,
                  'epsc0': eps_c,
                  'fpcu': 0.2 * f_c_1,
                  'epsU': eps_u,
                  'lamda': 0.1,
                  'ft': -0.1 * f_c_1,
                  'Ets': (-0.1 * f_c_1) / 0.002}
# uniaxial Kent-Scott-Park concrete material with degraded linear unloading/reloading
mat_KSP_conf = {'ID': 'Concrete02',
                'matTag': 2,
                'fpc': f_c_2,
                'epsc0': eps_c,
                'fpcu': 0.2 * f_c_2,
                'epsU': eps_u,
                'lamda': 0.1,
                'ft': -0.1 * f_c_2,
                'Ets': (-0.1 * f_c_2) / 0.002}

# uniaxial Giuffre-Menegotto-Pinto steel with isotropic strain hardening
mat_GMP = {'ID': 'Steel02',
           'matTag': 3,
           'Fy': f_y,
           'E0': E_s,
           'b': 0.005,
           'R0': 20.0,
           'cR1': 0.925,
           'cR2': 0.15}

sections = {'Beam': {'B': 300 * mm,
                     'H': 600 * mm,
                     'cover': 40 * mm,
                     'n_bars_top': 3,
                     'n_bars_bot': 3,
                     'n_bars_int_tot': 4,
                     'bar_area_top': rebar,
                     'bar_area_bot': rebar,
                     'bar_area_int': rebar},

            'Column': {'B': 300 * mm,
                       'H': 400 * mm,
                       'cover': 40 * mm,
                       'n_bars_top': 3,
                       'n_bars_bot': 3,
                       'n_bars_int_tot': 4,
                       'bar_area_top': rebar,
                       'bar_area_bot': rebar,
                       'bar_area_int': rebar}
            }

# Loading
C_L = 80.0 * (KN)  # Concentrated load
m_1 = 200.0 * ton  # lumped mass 1

# ---------------------------------------------------------------------------------------------
# -------------------------------- Run Analyzes -----------------------------------------------
# ---------------------------------------------------------------------------------------------

# # Gravity analysis
# MO.build_model(L_X, L_Y, L_Z, mat_KSP_unconf, mat_KSP_conf, mat_GMP, sections, C_L, m_1)
# MO.run_gravity()

# Modal analysis
MO.build_model(L_X, L_Y, L_Z, mat_KSP_unconf, mat_KSP_conf, mat_GMP, sections, C_L, m_1)
MO.run_modal()

# # Pushover analysis in X directions
# MO.build_model(L_X, L_Y, L_Z, mat_KSP_unconf, mat_KSP_conf, mat_GMP, sections, C_L, m_1)
# MO.run_gravity()
# MO.reset_analysis()
# MO.run_pushover(m_1, direction='X')
#
# # Pushover analysis in Y directions
# MO.build_model(L_X, L_Y, L_Z, mat_KSP_unconf, mat_KSP_conf, mat_GMP, sections, C_L, m_1)
# MO.run_gravity()
# MO.reset_analysis()
# MO.run_pushover(m_1, direction='Y')
#
# Time history analysis in X directions
MO.build_model(L_X, L_Y, L_Z, mat_KSP_unconf, mat_KSP_conf, mat_GMP, sections, C_L, m_1)
MO.run_gravity()
MO.reset_analysis()
MO.run_time_history(direction='X')

# Time history analysis in Y directions
MO.build_model(L_X, L_Y, L_Z, mat_KSP_unconf, mat_KSP_conf, mat_GMP, sections, C_L, m_1)
MO.run_gravity()
MO.reset_analysis()
MO.run_time_history(direction='Y')
# ---------------------------------------------------------------------------------------------
# --------------------------------Visualization------------------------------------------------
# ---------------------------------------------------------------------------------------------

# # -------------------------------- Pushover Curve ---------------------------------------------
# df_R_X = pd.read_table('Result_files/Pushover_Horizontal_ReactionsX.out', sep=" ", header=None,
#                        names=["Pseudo-Time", "R1_X", "R2_X", "R3_X", "R4_X"])
# df_R_Y = pd.read_table('Result_files/Pushover_Horizontal_ReactionsY.out', sep=" ", header=None,
#                        names=["Pseudo-Time", "R1_Y", "R2_Y", "R3_Y", "R4_Y"])
#
# df_R_X['sum_R'] = df_R_X.values[:, 1:5].sum(axis=1)
# df_R_Y['sum_R'] = df_R_Y.values[:, 1:5].sum(axis=1)
#
# df_D_X = pd.read_table('Result_files/Pushover_Story_DisplacementX.out', sep=" ", header=None,
#                        names=["Pseudo-Time", "D1_X", "D2_X", "D3_X", "D4_X"])
# df_D_Y = pd.read_table('Result_files/Pushover_Story_DisplacementY.out', sep=" ", header=None,
#                        names=["Pseudo-Time", "D1_Y", "D2_Y", "D3_Y", "D4_Y"])
#
# df_D_X['avg_D'] = df_D_X.values[:, 1:5].mean(axis=1)
# df_D_Y['avg_D'] = df_D_Y.values[:, 1:5].mean(axis=1)
#
# plt.figure(figsize=(10, 5))
#
# plt.plot(df_D_X['avg_D'], -df_R_X['sum_R'], color='#C0392B', linewidth=1.5)
# plt.plot(df_D_Y['avg_D'], -df_R_Y['sum_R'], color='#27AE60', linewidth=1.5)
#
# plt.ylabel('Base Shear (KN)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
# plt.xlabel('Average of Roof Displacement (m)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
# plt.grid(which='both')
# plt.title('Pushover Curve', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
# plt.yticks(fontname='Cambria', fontsize=14)
# plt.xticks(fontname='Cambria', fontsize=14)
# plt.legend(['X-Direction', 'Y-Direction'], prop={'family': 'Cambria', 'size': 14})
# plt.show()
# # ---------------------------------------------------------------------------------------------

# ------------------------------------ Ground Motion history ----------------------------------
G_M = np.loadtxt('Result_files/acc_1.txt')
times = np.arange(0, 0.02 * len(G_M), 0.02)
plt.figure(figsize=(12, 4))
plt.plot(times, G_M, color='#6495ED', linewidth=1.2)
plt.ylabel('Acceleration (m/s2)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.xlabel('Time (sec)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.grid(which='both')
plt.title('Time history of Ground Motion record', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
plt.yticks(fontname='Cambria', fontsize=14)
plt.show()
# ---------------------------------------------------------------------------------------------

# ------------------------------ Time history of displacement X Y -----------------------------
story_disp_X = np.loadtxt('Result_files/TimeHistory_Story_DisplacementX1.1.out')
story_disp_Y = np.loadtxt('Result_files/TimeHistory_Story_DisplacementY1.1.out')

plt.figure(figsize=(12, 5))
plt.plot(story_disp_X[:, 0], story_disp_X[:, 1], color='#DE3163', linewidth=1.2)
plt.plot(story_disp_Y[:, 0], story_disp_Y[:, 2], color='#FFBF00', linewidth=1.2)
plt.ylabel('Horizontal Displacement (m)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.xlabel('Time (sec)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.grid(which='both')
plt.title('Time history of horizontal dispacement', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
plt.yticks(fontname='Cambria', fontsize=14);
plt.xticks(fontname='Cambria', fontsize=14);
plt.legend(['X-Direction', 'Y-Direction'], prop={'family': 'Cambria', 'size': 14})
plt.show()
# ---------------------------------------------------------------------------------------------


# ------------------------------ Time history of acceleration X Y -----------------------------
story_accel_X = np.loadtxt('Result_files/TimeHistory_Story_AccelerationX1.1.out')
story_accel_Y = np.loadtxt('Result_files/TimeHistory_Story_AccelerationY1.1.out')

plt.figure(figsize=(12, 5))
plt.plot(story_accel_X[:, 0], story_accel_X[:, 1], color='#DE3163', linewidth=1.2)
plt.plot(story_accel_Y[:, 0], story_accel_Y[:, 2], color='#FFBF00', linewidth=1.2)
plt.ylabel('Horizontal Acceleration (m/s2)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.xlabel('Time (sec)', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.grid(which='both')
plt.title('Time history of horizontal acceleration', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
plt.yticks(fontname='Cambria', fontsize=14)
plt.xticks(fontname='Cambria', fontsize=14)
plt.legend(['X-Direction', 'Y-Direction'], prop={'family': 'Cambria', 'size': 14})
plt.show()
# ---------------------------------------------------------------------------------------------
