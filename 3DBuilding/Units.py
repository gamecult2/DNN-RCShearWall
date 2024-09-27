from Units import *

# ------------------------------------------------------------------------
# Define units - All results will be in { mm, N, MPa, Sec and Tonne }
# ------------------------------------------------------------------------
mm = 1.0  # 1 millimeter (length)
N = 1.0  # 1 Newton (force)
sec = 1.0  # 1 second (time)
t = 1.0  # 1 tonne (mass)

# Length
m = 1000.0 * mm  # 1 meter
cm = 10.0 * mm  # 1 centimeter
in_ = 25.4 * mm  # 1 inch (conversion factor)

# Area
m2 = m * m  # Square meter
cm2 = cm * cm  # Square centimeter
mm2 = mm * mm  # Square millimeter
in2 = in_ * in_  # Square inch (conversion factor)

# Volume
m3 = m * m * m  # Cubic meter
cm3 = cm * cm * cm  # Cubic centimeter
mm3 = mm * mm * mm  # Cubic millimeter
in3 = in_ * in_ * in_  # Cubic inch (conversion factor)

# Force
kN = 1000.0 * N  # 1 kilo-Newton
MN = 1000.0 * kN  # 1 Mega-Newton
kN_m = kN * m  # Kilo-Newton meter (moment)
MN_m = MN * m  # Mega-Newton meter (moment)

# Stress/Pressure
MPa = N / mm2  # MegaPascal (Pressure)
kPa = 0.001 * MPa  # KiloPascal (Pressure)
GPa = 1000 * MPa  # GigaPascal (Pressure)
psi = 6.895 * kPa  # Pounds per square inch (conversion factor)

# Density
t_m3 = t / m3  # Tonne per cubic meter
kg_m3 = 1.0 * t_m3 / 1000  # Converting tonne per cubic meter to kg per cubic meter
g_cm3 = kg_m3 / 1000  # grams per cubic centimeter
t_mm3 = t / mm3  # Tonne per cubic millimeter

# Mass
kg = 1.0 * N * sec * sec / m  # Kilogram (mass)
tonne = kg * 1000  # 1 tonne is 1000 kilograms

# Other
pi = 3.14159265359  # Pi
rad = pi / 180  # Radian (angular measure)
deg = 180 / pi  # Degree (angular measure)
Hz = 1 / sec  # Hertz (frequency)

# Gravitational acceleration (m/s^2), converted to mm/s^2 for consistency with units
g = 9.81 * m / (sec * sec) # Acceleration due to gravity
print("Gravity", g)