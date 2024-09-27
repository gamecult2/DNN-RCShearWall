import math
import time

import matplotlib.pyplot as plt
from keras import backend as K
from keras.saving.save import load_model
from utils import *
from RCWall_DataProcessing import *


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


data_folder = Path("RCWall_Data/Dataset_full")
# ---------------------- Load the saved NN model ----------------------------------------
# loaded_model = load_model("DNN_Models/DNN_Bi-LSTM(CYCLIC)", custom_objects={'r_square': r_square})
# info = 'Bi-LSTM'

loaded_model = load_model("DNN_Models/DNN_LSTM-AE(CYCLIC)300k", custom_objects={'r_square': r_square})
info = 'LSTM-AE'

# loaded_model = load_model("DNN_Models/NN_CNN(CYCLIC)")
# info = 'CNN'

# ---------------------- Load the scaler for normalization -------------------------------
param_scaler = (data_folder /'Scaler/param_scaler.joblib')
print(param_scaler)
disp_cyclic_scaler = (data_folder /'Scaler/disp_cyclic_scaler.joblib')
shear_cyclic_scaler = (data_folder /'Scaler/shear_cyclic_scaler.joblib')

# disp_pushover_scaler = 'RCWall_Data/Scaler/disp_pushover_scaler.joblib'
# shear_pushover_scaler = 'RCWall_Data/Scaler/shear_pushover_scaler.joblib'


# **********************************************************************
# EXPERIMENTAL DATASET
# **********************************************************************
# ------- Load New data for prediction -------------------------------------------------------------------------------------
def reftest():
    name = 'Thomsen_and_Wallace_RW2'
    sign = '(4)'
    parameters_input = np.array((102.0, 3810.0, 1220.0, 190.0, 41.75, 434.0, 448.0, 0.0294, 0.003, 0.092)).reshape(1, -1)
    max_displacement = 86
    return parameters_input, max_displacement, name, sign


def ref4():
    name = 'Thomsen_RW1'
    sign = '(4)'
    parameters_input = np.array((102.0, 3810.0, 1220.0, 190.0, 42.75, 434.0, 448.0, 0.0294, 0.003, 0.094)).reshape(1, -1)
    max_displacement = 70
    return parameters_input, max_displacement, name, sign


def ref5():
    name = 'Tran_S63'
    sign = '(5)'
    parameters_input = np.array((152.0, 2600.0, 1220.0, 180, 50, 455, 505.0, 0.0715, 0.0061, 0.1)).reshape(1, -1)
    max_displacement = 76
    return parameters_input, max_displacement, name, sign

def ref55():
    name = 'Tran_S63'
    sign = '(55)'
    parameters_input = np.array((152.0, 1829.0, 1220.0, 180, 56, 477, 477, 0.073, 0.0061, 0.085)).reshape(1, -1)
    max_displacement = 76
    return parameters_input, max_displacement, name, sign


def ref6():
    name = 'Alarcon_W1'
    sign = '(6)'
    parameters_input = np.array((100.0, 1600.0, 700.0, 70.0, 27.0, 469.0, 445.0, 0.0449, 0.0054, 0.17)).reshape(1, -1)
    max_displacement = 39
    return parameters_input, max_displacement, name, sign


def ref7():
    name = 'Aejaz_SW1'
    sign = '(7)'
    parameters_input = np.array((100.0, 3600.0, 1250.0, 125.0, 35.0, 540.0, 540.0, 0.0308, 0.002, 0.09)).reshape(1, -1)
    max_displacement = 105
    return parameters_input, max_displacement, name, sign

def ref8():
    name = 'Faraone_AR2-8'
    sign = '(5)'
    parameters_input = np.array((304.0, 4800.0, 2400.0, 410.0, 41.36, 503.94, 503.94, 0.0187, 0.0085, 0.090)).reshape(1, -1)
    max_displacement = 120
    return parameters_input, max_displacement, name, sign


sequence_length = 499
num_cycles = 8
repetition_cycles = 2
initial_displacement = 15
num_points = math.ceil(sequence_length / (num_cycles * repetition_cycles))  # Ensure at least 500 points in total

for ref_func in [reftest, ref4, ref5, ref55, ref6, ref7, ref8]:
    parameters_input, max_displacement, name, sign = ref_func()
    displacement_input = generate_increasing_cyclic_loading_with_repetition(num_cycles=num_cycles, max_displacement=max_displacement, num_points=num_points, repetition_cycles=repetition_cycles).reshape(1, -1)[:, 1:sequence_length+1]
    # displacement_input = generate_increasing_cyclic_loading(num_cycles=num_cycles, initial_displacement=initial_displacement, max_displacement=max_displacement, num_points=num_points, repetition_cycles=repetition_cycles).reshape(1, -1)[:, :500]
    # displacement_input = generate_increasing_cyclic_loading_with_exponential_growth(num_cycles=num_cycles, initial_displacement=initial_displacement, max_displacement=max_displacement, num_points=num_points, repetition_cycles=repetition_cycles).reshape(1, -1)[:, 1:sequence_length]

    # ------- Normalize New data ------------------------------------------
    parameters_input = normalize(parameters_input, scaler_filename=param_scaler, sequence=False, fit=False)
    displacement_input = normalize(displacement_input, scaler_filename=disp_cyclic_scaler, sequence=True, fit=False)

    # ------- Predict New data --------------------------------------------
    predicted_shear = loaded_model.predict([parameters_input, displacement_input])

    # ------- Denormalize New data ------------------------------------------
    parameter_values = denormalize(parameters_input, scaler_filename=param_scaler, sequence=False)
    DisplacementStep = denormalize(displacement_input, scaler_filename=disp_cyclic_scaler, sequence=True)
    predicted_shear = denormalize(predicted_shear, scaler_filename=shear_cyclic_scaler, sequence=True)
    predicted_shear -= 45

    plotting(DisplacementStep[-1, 5:499], predicted_shear[-1, 5:499], 'Displacement (mm)', 'Base Shear (kN)', name, sign, save_fig=False, plotValidation=True)
    print("Max predicted_shear", np.max(predicted_shear))

# '''
# **********************************************************************
# TESTING DATASET
# **********************************************************************
# Define the number of sample to be used
batch_size = 300000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 499
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, pushover=False)
InParams, InDisp, OutShear = returned_data

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutShear, test_size=0.20, random_state=42)

# ---------------------- Plotting the results ---------------------------------------------
test_index = 400
new_input_parameters = X_param_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_disp_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]

# Now, you can use the loaded model to make predictions on new data
predicted_shear = loaded_model.predict([new_input_parameters, new_input_displacement])

# ------- Denormalize New data ------------------------------------------
new_input_parameters = denormalize(new_input_parameters, scaler_filename=param_scaler, sequence=False)
new_input_displacement = denormalize(new_input_displacement, scaler_filename=disp_cyclic_scaler, sequence=True)
real_shear = denormalize(real_shear, scaler_filename=shear_cyclic_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear, scaler_filename=shear_cyclic_scaler, sequence=True)

test_index = (15, 4, 56)
name = (15, 4, 56)
name = ('(1)', '(2)', '(3)')
# test_index = list(range(1, 100))
# name = list(range(1, 100))

for k, i in enumerate(test_index):
    print("Max predicted_shear", np.max(predicted_shear[i]))
    print("Max real_shear", np.max(real_shear[i]))
    print("Max predicted_shear", np.max(predicted_shear[i]))
    print("Max real_shear", np.max(real_shear[i]))
    plt.figure(figsize=(4 * 1.1, 3 * 1.25))
    plt.subplots_adjust(top=0.918, bottom=0.139, left=0.194, right=0.979, hspace=0.2, wspace=0.185)
    plt.plot(new_input_displacement[i], predicted_shear[i], color='blue', linewidth=0.8, label=f'DNN prediction')
    plt.plot(new_input_displacement[i], real_shear[i], color="black", linewidth=0.8, linestyle="--", label=f'Reference {name[k]}')
    font_settings = {'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12}
    plt.xlabel('Displacement (mm)', fontdict=font_settings)
    plt.ylabel('Base Shear (kN)', fontdict=font_settings, labelpad=-1)
    plt.title(f'Specimen {name[k]}', fontdict={'fontname': 'Times New Roman', 'fontstyle': 'normal', 'size': 14})
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    plt.yticks(fontname='Cambria', fontsize=12)
    plt.xticks(fontname='Cambria', fontsize=12)
    plt.tight_layout()
    plt.legend(prop={'family': 'Times New Roman', 'size': 9})
    # plt.savefig(f'DataValidation/DNNModelValidation/CyclicPredicted{name[k]}{info}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
#'''
