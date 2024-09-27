import numpy as np
import time
from Units import *
import matplotlib.pyplot as plt
import Building_Model as buildmodel
import openseespy.opensees as ops


def plotting(x_data, y_data, x_label, y_label, title, save_fig=True, plotValidation=True):
    plt.rcParams.update({'font.size': 10, "font.family": ["Times New Roman", "Cambria"]})
    # Plot Force vs. Displacement
    # plt.figure(figsize=(4.0, 4.2), dpi=100)
    plt.figure(figsize=(4 * 1.1, 3 * 1.25))
    # plt.figure(figsize=(7 / 3, 6 / 3), dpi=100)
    # Read test output data to plot
    if plotValidation:
        Test = np.loadtxt(f"DataValidation/{title}.txt", delimiter="\t", unpack="False")
        plt.plot(Test[0, :], Test[1, :], color="black", linewidth=1.0, linestyle="--", label='Experimental Test')
    plt.plot(x_data, y_data, color='red', linewidth=1.2, label='Numerical Test')
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'size': 10}
    plt.xlabel(x_label, fontdict=font_settings)
    plt.ylabel(y_label, fontdict=font_settings)
    plt.yticks(fontname='Cambria', fontsize=10)
    plt.xticks(fontname='Cambria', fontsize=10)
    plt.title(f"Specimen : {title}", fontdict={'fontname': 'Times New Roman', 'fontstyle': 'normal', 'size': 10})
    plt.tight_layout()
    plt.legend(fontsize='small')
    if save_fig:
        plt.savefig('DataValidation/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------ Selected Experiment for Validation Model ------------------------------
def test():
    # https://doi.org/10.1016/j.engstruct.2009.02.018
    global name
    name = 'TestBuilding'
    return tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadcoef, DisplacementStep


# ------- Select Model for Validation -----------------------------------------------------------------------------------------------
validation_model = test()
tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff, DisplacementStep = validation_model

#  ---------------- RUN  ANALYSIS ---------------------------------------------------------------
buildmodel.BuildingModel(tw, hw, lw, lbe, fc, fyb, fyw, rouYb, rouYw, loadCoeff)
[x, y] = buildmodel.run_THA(DisplacementStep,,
buildmodel.reset_analysis()
# plotting(x, y, 'Displacement (mm)', 'Base Shear (kN)', f'{name}', save_fig=False, plotValidation=True)
