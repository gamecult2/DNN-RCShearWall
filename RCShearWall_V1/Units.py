# ------------------------------------------------------------------------
# Define units - All results will be in { mm, N, MPa and Sec }
# ------------------------------------------------------------------------
mm = 1.0  # 1 millimeter
N = 1.0  # 1 Newton
sec = 1.0  # 1 second

m = 1000.0 * mm  # 1 meter is 1000 millimeters
cm = 10.0 * mm  # 1 centimeter is 10 millimeters
kN = 1000.0 * N  # 1 kilo-Newton is 1000 Newtons
m2 = m * m  # Square meter
cm2 = cm * cm  # Square centimeter
mm2 = mm * mm  # Square millimeter
MPa = N / mm2  # MegaPascal (Pressure)
kPa = 0.001 * MPa  # KiloPascal (Pressure)
GPa = 1000 * MPa  # GigaPascal (Pressure)


def rebarArea(rebarDiameter):
    a = 3.1416 * (rebarDiameter / 2) ** 2  # compute area
    return a