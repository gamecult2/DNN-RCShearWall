import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from RCWall_Cyclic_Parameters import *

# Allocate space for Bidirectional(LSTM)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Activate the GPU
tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))


def normalization(data):
    actual_min = np.min(data)
    actual_max = np.max(data)
    normalized_data = (2 * (data - actual_min) / (actual_max - actual_min)) - 1
    return normalized_data, actual_min, actual_max


def restoration(normalized_data, actual_min, actual_max):
    restored_data = ((normalized_data + 1) / 2) * (actual_max - actual_min) + actual_min
    return restored_data


def read_data(batch_size=100, sequence_length=500, normalize_data=True, save_normalized_data=False, smoothed_data=False):
    # ---------------------- Read Data  -------------------------------
    if smoothed_data:
        folder = "RCWall_Data/Smoothed/"
        print('----------------------------------------')
        print("Smoothed Data Loaded")
    else:
        folder = "RCWall_Data/"
        print('----------------------------------------')
        print("Normal Data Loaded")

    # Input files (Structural Parameters + Cyclic Loading)
    InParams = np.genfromtxt(f"RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=batch_size)
    InDisp = np.genfromtxt(f"RCWall_Data/InputDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # Output files (Hysteresis Curve)
    OutCycShear = np.genfromtxt(f"{folder}OutputCyclicShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # OutCycDisp = np.genfromtxt(f"{folder}OutputCyclicDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # # Output files (Pushover Curve)
    # OutPushShear = np.genfromtxt(f"{folder}OutputPushoverShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # OutPushDisp = np.genfromtxt(f"{folder}OutputPushoverDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))

    if normalize_data:
        print('----------------------------------------')
        print("Normalized Data Loaded")

        # ---------------------- Data Normalization  ----------------------
        # Input Normalization (Structural Parameters + Cyclic Loading)
        param_scaler = MinMaxScaler(feature_range=(0, 1))
        NormInParams = param_scaler.fit_transform(InParams)

        # disp_scaler = StandardScaler()
        # NormInDisp = disp_scaler.fit_transform(InDisp)

        # Output Normalization (Hysteresis Curve)
        # cyc_shear_scaler = StandardScaler()
        # NormOutCycShear = cyc_shear_scaler.fit_transform(OutCycShear)
        # cyc_disp_scaler = StandardScaler()
        # NormOutCycDisp = cyc_disp_scaler.fit_transform(OutCycDisp)

        # Output Normalization (Hysteresis Curve)
        # push_shear_scaler = StandardScaler()
        # NormOutPushShear = push_shear_scaler.fit_transform(OutPushShear)
        # push_disp_scaler = StandardScaler()
        # NormOutPushDisp = push_disp_scaler.fit_transform(OutPushDisp)

        # # ---------------------- Data Normalization  ----------------------
        # Normalize Input Displacement
        NormInDisp, InDisp_min, InDisp_max = normalization(InDisp)

        # Normalize Output Cyclic Shear
        NormOutCycShear, OutCycShear_min, OutCycShear_max = normalization(OutCycShear)
        # NormOutCycDisp, OutCycDisp_min, OutCycDisp_max = normalization(OutCycDisp)

        if save_normalized_data:
            # ---------------------- Save Normalized Data --------------------
            # Save normalized Input data to CSV files
            np.savetxt("RCWall_Data/Normalized/InputParameters.csv", NormInParams, delimiter=',')
            np.savetxt("RCWall_Data/Normalized/InputDisplacement.csv", NormInDisp, delimiter=',')
            # Save normalized Output data to CSV files
            np.savetxt("RCWall_Data/Normalized/OutputCyclicShear.csv", NormOutCycShear, delimiter=',')
            # np.savetxt("RCWall_Data/Normalized/OutputCyclicDisplacement.csv", NormOutCycDisp, delimiter=',')
            # Save normalized Output data to CSV files
            # np.savetxt("RCWall_Data/Normalized/OutputPushoverShear.csv", NormOutPushShear, delimiter=',')
            # np.savetxt("RCWall_Data/Normalized/OutputPushoverDisplacement.csv", NormOutPushDisp, delimiter=',')

        print('----------------------------------------')
        print('InputParameters Shape = ', InParams.shape)
        print('InputDisplacement Shape = ', InDisp.shape)

        # return (NormInParams, NormInDisp, NormOutCycShear, NormOutCycDisp, NormOutPushShear, NormOutPushDisp, InDisp_min, InDisp_max, OutCycShear_min, OutCycShear_max, OutCycDisp_min, OutCycDisp_max), \
        #        (param_scaler, disp_scaler, cyc_shear_scaler, cyc_disp_scaler, push_shear_scaler, push_disp_scaler)
        return (NormInParams, NormInDisp, NormOutCycShear, InDisp_min, InDisp_max, OutCycShear_min, OutCycShear_max), \
            (param_scaler)
    else:
        print('----------------------------------------')
        print("Raw Data Loaded")
        return InParams, InDisp, OutCycShear  # , OutCycDisp, OutPushShear, OutPushDisp


def read_data_validation(batch_size=100, sequence_length=500, normalize_data=True, save_normalized_data=False):
    # ---------------------- Read Data  -------------------------------
    # Input files (Structural Parameters + Cyclic Loading)
    InParams = np.genfromtxt(f"RCWall_Data/InputParameters_values.csv", delimiter=',', max_rows=batch_size)
    InDisp = np.genfromtxt(f"RCWall_Data/InputDisplacement_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # Output files (Hysteresis Curve)
    OutCycShear = np.genfromtxt(f"RCWall_Data/Validation/OutputCyclicShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))
    # # Output files (Pushover Curve)
    # OutPushShear = np.genfromtxt(f"{folder}OutputPushoverShear_values.csv", delimiter=',', max_rows=batch_size, usecols=range(sequence_length))

    # ---------------------- Data Normalization  ----------------------
    # Input Normalization (Structural Parameters + Cyclic Loading)
    param_scaler = MinMaxScaler(feature_range=(0, 1))
    NormInParams = param_scaler.transform(InParams)

    # Normalize Input Displacement
    NormInDisp, InDisp_min, InDisp_max = normalization(InDisp)

    # Normalize Output Cyclic Shear
    NormOutCycShear, OutCycShear_min, OutCycShear_max = normalization(OutCycShear)
    # NormOutCycDisp, OutCycDisp_min, OutCycDisp_max = normalization(OutCycDisp)
    print('----------------------------------------')
    print("Normalized Data Loaded")

    return NormInParams, NormInDisp, NormOutCycShear


