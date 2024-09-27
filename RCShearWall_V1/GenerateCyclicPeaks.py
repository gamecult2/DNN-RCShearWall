import os

def generate_peaks(Dmax, DincrStatic=0.01, CycleType="Full", Fact=1):
    """
    Generates incremental displacements for Dmax.

    Parameters:
    - Dmax: Peak displacement (can be + or negative).
    - DincrStatic: Displacement increment (optional, default=0.01, independently of units).
    - CycleType: Full (0->+peak), Half (0->+peak->0), Full (0->+peak->0->-peak->0) (optional, def=Full).
    - Fact: Scaling factor (optional, default=1).

    Returns:
    - iDstep: List of displacement increments.
    """

    if not os.path.exists("data"):
        os.makedirs("data")

    outFileID = open("data/tmpDsteps.py", "w")

    Disp = 0.0

    outFileID.write("iDstep = [")
    outFileID.write(str(Disp))
    outFileID.write(", ")
    outFileID.write(str(Disp))  # Open vector definition and some 0

    Dmax *= Fact  # Scale value

    if Dmax < 0:  # Avoid divide by zero
        dx = -DincrStatic
    else:
        dx = DincrStatic

    NstepsPeak = int(abs(Dmax) / DincrStatic)

    for i in range(1, NstepsPeak + 1):  # Zero to one
        Disp += dx
        outFileID.write(", ")
        outFileID.write(str(Disp))  # Write to file

    if CycleType != "Push":
        for i in range(1, NstepsPeak + 1):  # One to zero
            Disp -= dx
            outFileID.write(", ")
            outFileID.write(str(Disp))  # Write to file

        if CycleType != "Half":
            for i in range(1, NstepsPeak + 1):  # Zero to minus one
                Disp -= dx
                outFileID.write(", ")
                outFileID.write(str(Disp))  # Write to file

            for i in range(1, NstepsPeak + 1):  # Minus one to zero
                Disp += dx
                outFileID.write(", ")
                outFileID.write(str(Disp))  # Write to file

    outFileID.write("]")  # Close vector definition
    outFileID.close()

    from data.tmpDsteps import iDstep  # Source Python file to define entire vector
    return iDstep

