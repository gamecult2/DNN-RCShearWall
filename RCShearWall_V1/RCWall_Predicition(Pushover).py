import math
import matplotlib.pyplot as plt
from keras import backend as K
from keras.saving.save import load_model
from GenerateCyclicLoading import *
from RCWall_DataProcessing import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import interpolate
from scipy.stats import pearsonr


def calculate_metrics(target, output):
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    # r2 = r_square(target, output)
    R, p = pearsonr(target, output)
    return mse, mae, R


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def plotting(x_data, y_data, x_label, y_label, title, sign, save_fig=False, plotValidation=False):
    plt.rcParams.update({'font.size': 14, "font.family": ["Times New Roman", "Cambria"]})

    # Plot Force vs. Displacement
    plt.figure(figsize=(4 * 1.1, 3 * 1.25))
    plt.subplots_adjust(top=0.918, bottom=0.139, left=0.194, right=0.979, hspace=0.2, wspace=0.185)

    # Read test output data to plot
    plt.plot(x_data[4:], y_data[4:], color='red', linewidth=1.2, label=f'DNN prediction')
    if plotValidation:
        Test = np.loadtxt(f"DataValidation/{title}.txt", delimiter="\t", unpack="False")
        plt.plot(Test[0, :], Test[1, :], color="black", linewidth=0.8, linestyle="--", label=f'Reference {sign}')
        print(len(Test[1]))
        f = interpolate.interp1d(np.arange(len(Test[1])), Test[1], kind="linear")
        new_x = np.linspace(0, len(Test[1]) - 1, len(y_data[4:]))
        reshaped_Test = f(new_x)
        print(len(reshaped_Test))
        # print("Max real_shear", np.max(Test[1]))
        # print("Max predicted_shear", np.max(predicted_shear))
        mse, mae, r = calculate_metrics(reshaped_Test, y_data[4:])
        ax = plt.gca()  # Get current axes
        plt.text(0.99, 0.03, f"MSE: {mse:.3f}, MAE: {mae:.3f}, R: {r:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

    plt.grid(linestyle='dotted')
    font_settings = {'fontname': 'Times New Roman', 'fontstyle': 'italic', 'size': 12}

    plt.xlabel('Displacement (mm)', fontdict=font_settings)
    plt.ylabel('Base Shear (kN)', fontdict=font_settings, labelpad=-1)
    plt.axhline(0, color='black', linewidth=0.4)
    plt.axvline(0, color='black', linewidth=0.4)
    plt.grid(linestyle='dotted')
    plt.yticks(fontname='Cambria', fontsize=12)
    plt.xticks(fontname='Cambria', fontsize=12)
    # Adjust yticks to only show positive values
    plt.ylim(bottom=-0.08 * max(y_data))
    plt.xlim(left=-0.08 * max(x_data))
    plt.tight_layout()
    plt.title(f"Specimen {sign} - {name} ", fontdict={'fontname': 'Times New Roman', 'fontstyle': 'normal', 'size': 14})
    plt.tight_layout()
    plt.legend(prop={'family': 'Times New Roman', 'size': 10}, loc='center right')
    if save_fig:
        plt.savefig('DataValidation/DNNModelValidation/' + title + '.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------- Load the saved NN model ----------------------------------------
loaded_model = load_model("DNN_Models/DNN_Bi-LSTM(PUSHOVER)65k", custom_objects={'r_square': r_square})
info = 'Bi-LSTM'

# loaded_model = load_model("DNN_Models/DNN_LSTM-AE(PUSHOVER)", custom_objects={'r_square': r_square})
# info = 'LSTM-AE'

# loaded_model = load_model("DNN_Models/NN_CNN(PUSHOVER)")
# info = 'CNN'

# ---------------------- Load the scaler for normalization -------------------------------
param_scaler = 'RCWall_Data/Scaler/param_scaler.joblib'

disp_cyclic_scaler = 'RCWall_Data/Scaler/disp_cyclic_scaler.joblib'
shear_cyclic_scaler = 'RCWall_Data/Scaler/shear_cyclic_scaler.joblib'

disp_pushover_scaler = 'RCWall_Data/Scaler/disp_pushover_scaler.joblib'
shear_pushover_scaler = 'RCWall_Data/Scaler/shear_pushover_scaler.joblib'


# **********************************************************************
# EXPERIMENTAL DATASET
# **********************************************************************
# ------- Load New data for prediction -------------------------------------------------------------------------------------
def reftest():
    name = 'Thomsen_and_Wallace_RW2'
    sign = 'A'
    parameters_input = np.array((102, 3810, 1220, 190, 42, 434, 448, 0.0300, 0.0029, 0.0920)).reshape(1, -1)
    max_displacement = 86
    return parameters_input, max_displacement, name, sign

def ref5():
    name = 'Tran_S63'
    sign = '(5)'
    parameters_input = np.array((152.0, 2500.0, 1220.0, 208.5, 48.6, 452.5, 505.0, 0.0712, 0.0061, 0.08588235294117647)).reshape(1, -1)
    max_displacement = 76
    return parameters_input, max_displacement, name, sign

# def refD():
#     name = 'Lefas_SW22'
#     sign = '(D)'
#     parameters_input = np.array((65, 650, 1350, 140, 51, 470, 470, 0.0330, 0.0025, 0.1000)).reshape(1, -1)
#     max_displacement = 16
#     return parameters_input, max_displacement, name, sign

def refD():
    name = 'Lu_SW1-1'
    sign = '(D)'
    parameters_input = np.array((125.0, 2000.0, 1000.0, 200.0, 21, 379.0, 392.0, 0.0188, 0.0038, 0.12)).reshape(1, -1)
    max_displacement = 20
    return parameters_input, max_displacement, name, sign

def refE():
    name = 'Zhang_SW12'
    sign = '(E)'
    parameters_input = np.array((120, 1000, 2000, 200, 20, 352, 379, 0.0167, 0.0100, 0.2000)).reshape(1, -1)
    max_displacement = 23
    return parameters_input, max_displacement, name, sign


def refF():
    name = 'Mosoarca_SW1'
    sign = '(F)'
    parameters_input = np.array((82.0, 2600.0, 1250.0, 100.0, 50.0, 386.0, 386.0, 0.009, 0.01, 0.014)).reshape(1, -1)
    max_displacement = 27
    return parameters_input, max_displacement, name, sign

def refG():
    name = 'Rao_1'
    sign = '(E)'
    parameters_input = np.array((200.0, 3000.0, 1560.0, 200.0, 25.0, 400.0, 400.0, 0.0075, 0.0027, 0.02)).reshape(1, -1)
    max_displacement = 75
    return parameters_input, max_displacement, name, sign


def refH():
    name = 'Bismarck'
    sign = '(H)'
    parameters_input = np.array((203.0, 3000.0, 3048.0, 380.0, 20.8, 462.7, 462.9, 0.0067, 0.0067, 0.001)).reshape(1, -1)
    max_displacement = 50
    return parameters_input, max_displacement, name, sign


def refE():
    name = 'Dazio_WSH2'
    sign = '(E)'
    parameters_input = np.array((150.0, 4260.0, 2000.0, 200.0, 42.5, 583.1, 484.9, 0.0132, 0.003, 0.067)).reshape(1, -1)
    max_displacement = 55
    return parameters_input, max_displacement, name, sign


for ref_func in [refD]:
    parameters_input, max_displacement, name, sign = ref_func()
    displacement_input = np.linspace(0, max_displacement, 500).reshape(1, -1)[:, :499]

    # ------- Normalize New data ------------------------------------------
    parameters_input = normalize(parameters_input, scaler_filename=param_scaler, sequence=False, fit=False)
    displacement_input = normalize(displacement_input, scaler_filename=disp_pushover_scaler, sequence=True, fit=False)

    # ------- Predict New data --------------------------------------------
    predicted_shear = loaded_model.predict([parameters_input, displacement_input])

    # ------- Denormalize New data ------------------------------------------
    parameter_values = denormalize(parameters_input, scaler_filename=param_scaler, sequence=False)
    DisplacementStep = denormalize(displacement_input, scaler_filename=disp_pushover_scaler, sequence=True)
    predicted_shear = denormalize(predicted_shear, scaler_filename=shear_pushover_scaler, sequence=True)


    plotting(DisplacementStep[-1, 0:499], predicted_shear[-1, 0:499], 'Displacement (mm)', 'Base Shear (kN)', name, sign, save_fig=False, plotValidation=True)


# '''
# **********************************************************************
# TESTING DATASET
# **********************************************************************
# Define the number of sample to be used
batch_size = 200000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 499
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, pushover=True)
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
new_input_displacement = denormalize(new_input_displacement, scaler_filename=disp_pushover_scaler, sequence=True)
real_shear = denormalize(real_shear, scaler_filename=shear_pushover_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear, scaler_filename=shear_pushover_scaler, sequence=True)

test_index = (38, 45, 316)
name = ('(A)', '(B)', '(C)')

# test_index = list(range(300, 400))
# name = list(range(300, 400))

for k, i in enumerate(test_index):

    # print("Parameters", new_input_parameters[i])
    # print("Max Displacement", np.max(new_input_displacement[i]))
    print("Max predicted_shear", np.max(predicted_shear[i]))
    print("Max real_shear", np.max(real_shear[i]))
    print("Max predicted_shear", np.max(predicted_shear[i]))
    print("Max real_shear", np.max(real_shear[i]))
    plt.figure(figsize=(4 * 1.1, 3 * 1.25))
    plt.subplots_adjust(top=0.918, bottom=0.139, left=0.194, right=0.979, hspace=0.2, wspace=0.185)
    plt.plot(new_input_displacement[i], predicted_shear[i], color='red', linewidth=1.0, label=f'DNN prediction')
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
    plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    # plt.savefig(f'DataValidation/DNNModelValidation/PushoverPredicted{name[k]}{info}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()

#'''
