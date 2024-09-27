from Units import *
import openseespy.opensees as ops

# CONCRETE ---------------------------------------------------------------
fc = -1 * 23.4 * MPa  # CONCRETE Compressive Strength, ksi   (+Tension, -Compression)
Ec = 31.5 * MPa  # Concrete Elastic Modulus
nu = 0.2
Gc = Ec / 2 / (1 + nu)  # Torsional stiffness Modulus

# confined concrete
Kfc = 1.3  # ratio of confined to unconfined concrete strength
Kres = 0.2  # ratio of residual/ultimate to maximum stress
fc1C = Kfc * fc  # CONFINED concrete (mander model), maximum stress
eps1C = 2 * fc1C / Ec  # strain at maximum stress
fc2C = Kres * fc1C  # ultimate stress
eps2C = 20 * eps1C  # strain at ultimate stress
lam = 0.1  # ratio between unloading slope at eps2 and initial slope Ec

# unconfined concrete
fc1U = fc  # UNCONFINED concrete (todeschini parabolic model), maximum stress
eps1U = -0.003  # strain at maximum strength of unconfined concrete
fc2U = Kres * fc1U  # ultimate stress
eps2U = -0.01  # strain at ultimate stress

# tensile-strength properties
ftC = -0.14 * fc1C  # tensile strength +tension
ftU = -0.14 * fc1U  # tensile strength +tension
Ets = ftU / 0.002  # tension softening stiffness

# Concrete IDs
IDconcCore = 1
IDconcCover = 2
ops.uniaxialMaterial('Concrete02', IDconcCore, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)  # CORE CONCRETE  (confined)
ops.uniaxialMaterial('Concrete02', IDconcCover, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)  # COVER CONCRETE  (unconfined)
# print('Concrete02', IDconcCore, fc1C, eps1C, fc2C, eps2C, lam, ftC, Ets)  # CORE CONCRETE  (confined)
# print('Concrete02', IDconcCover, fc1U, eps1U, fc2U, eps2U, lam, ftU, Ets)  # COVER CONCRETE  (unconfined)


# # Beam
# Pinching4Beam = 100
# ops.uniaxialMaterial('Pinching4', Pinching4Beam, ePf1, ePd1, ePf2, ePd2, ePf3, ePd3, ePf4, ePd4, <eNf1, eNd1, eNf2, eNd2, eNf3, eNd3, eNf4, eNd4>, rDispP, rForceP, uForceP, rDispN, rForceN, uForceN>, gK1, gK2, gK3, gK4, gKLim, gD1, gD2, gD3, gD4, gDLim, gF1, gF2, gF3, gF4, gFLim, gE, dmgType)
# # Column
# Pinching4Column = 200
# ops.uniaxialMaterial('Pinching4', Pinching4Column, ePf1, ePd1, ePf2, ePd2, ePf3, ePd3, ePf4, ePd4, <eNf1, eNd1, eNf2, eNd2, eNf3, eNd3, eNf4, eNd4>, rDispP, rForceP, uForceP, rDispN, rForceN, uForceN>, gK1, gK2, gK3, gK4, gKLim, gD1, gD2, gD3, gD4, gDLim, gF1, gF2, gF3, gF4, gFLim, gE, dmgType)
#


# STEEL
Fy = 335 * MPa  # STEEL yield stress
Es = 200000 * MPa  # modulus of steel
Bs = 0.01  # strain-hardening ratio
R0 = 18  # control the transition from elastic to plastic branches
cR1 = 0.925  # control the transition from elastic to plastic branches
cR2 = 0.15  # control the transition from elastic to plastic branches

# Steel IDs
IDSteel = 3
ops.uniaxialMaterial('Steel02', IDSteel, Fy, Es, Bs, R0, cR1, cR2)  # steel Y web
# print('Steel02', IDSteel, Fy, Es, Bs, R0, cR1, cR2)