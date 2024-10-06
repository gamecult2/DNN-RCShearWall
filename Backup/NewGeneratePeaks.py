import os
# from SFI_MVLEM_2CLH18 import H
# -------------------------------------------------
# NewGeneratePeaks Dmax DincrStatic CycleType Fact
# Created by: Mushi Chang
# Date: 10/2022
# -------------------------------------------------
# Input variables
#   Dmax : peak displacement
#   DincrStatic : displacement increment (optional, default = 0.01, independently of units)
#   CycleType : Full, Half, Push
#   Fact : scaling factor (optional. default=1)
#   iDstepFileName : file name where dispalcement history is stored temporarily, until next step
# Output variable
#   iDstep : vector of displacement variable

def NewGeneratePeaks (Dmax, DincrStatic = 0.01, CycleType = 'Full', H = 1):
    tmpDsteps = []
    Disp = 0
    tmpDsteps.append(float(Disp))
    tmpDsteps.append(float(Disp))
    Dmax = Dmax * H              # scale drift ratio by story height for displacement cycles
    if Dmax < 0:
        dx = -DincrStatic
    else:
        dx = DincrStatic
    NStepsPeak = int(abs(Dmax / DincrStatic))
    for i in range(1, NStepsPeak):
        Disp = Disp + dx
        tmpDsteps.append(float(Disp))
    if CycleType != 'Push':
        for i in range(1, NStepsPeak):
            Disp = Disp - dx
            tmpDsteps.append(float(Disp))
        if CycleType != 'Half':
            for i in range(1, NStepsPeak):
                Disp = Disp - dx
                tmpDsteps.append(float(Disp))
            for i in range(1, NStepsPeak):
                Disp = Disp + dx
                tmpDsteps.append(float(Disp))
    return tmpDsteps
# iDsteps = NewGeneratePeaks(0.02, 0.015, 'Full', Fact)
# print(iDsteps)
