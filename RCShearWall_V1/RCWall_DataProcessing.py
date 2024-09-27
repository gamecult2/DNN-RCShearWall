import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
# from RCWall_Cyclic_Parameters import *
import joblib
from pathlib import Path  # For path handling
import pyarrow as pa
import pyarrow.parquet as pq


def normalize(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, fit=False, save_scaler_path=None):
    # Check NOT fit (First Normalization) Then must load a scaler or scaler_filename
    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    if scaler is None:
        if scaler_filename:
            # Load the scaler if a filename is provided
            if not os.path.exists(scaler_filename):
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
            scaler = joblib.load(scaler_filename)
        else:
            # Create a new scaler if neither scaler nor scaler filename is provided
            scaler = MinMaxScaler(feature_range=range)

    if sequence:
        data_reshaped = data.reshape(-1, 1)

        if fit:
            data_scaled = scaler.fit_transform(data_reshaped)
            print("Min value of the scaler:", scaler.data_min_)
            print("Max value of the scaler:", scaler.data_max_)
        else:
            data_scaled = scaler.transform(data_reshaped)

        data_scaled = data_scaled.reshape(data.shape)

    else:
        if fit:
            data_scaled = scaler.fit_transform(data)
            print("Min value of the scaler:", scaler.data_min_)
            print("Max value of the scaler:", scaler.data_max_)
        else:
            data_scaled = scaler.transform(data)

    # Save the scaler if a path is provided
    if save_scaler_path and fit:
        joblib.dump(scaler, save_scaler_path)

    if scaler_filename:
        return data_scaled
    else:
        return data_scaled, scaler


def denormalize(data_scaled, scaler=None, scaler_filename=None, sequence=False):
    if scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for denormalization.")

    if sequence:
        data_reshaped = data_scaled.reshape(-1, 1)

        if scaler_filename:
            scaler = joblib.load(scaler_filename)

        data_restored_1d = scaler.inverse_transform(data_reshaped)
        data_restored = data_restored_1d.reshape(data_scaled.shape)

    else:
        if scaler_filename:
            scaler = joblib.load(scaler_filename)

        data_restored = scaler.inverse_transform(data_scaled)

    return data_restored


def load_data(data_size=100, sequence_length=500, normalize_data=True, save_normalized_data=False, pushover=False):
    # ---------------------- Read Data  -------------------------------
    data_folder = Path("RCWall_Data/Dataset_full")  # Base data folder
    file_suffix = "Pushover" if pushover else "Cyclic"

    # Read input and output data from Parquet files
    # 310022
    InParams = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size].to_numpy(dtype=float)
    InDisp = pd.read_parquet(data_folder / f"Input{file_suffix}Displacement.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=float)
    OutShear = pd.read_parquet(data_folder / f"Output{file_suffix}Shear.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=float)
    print("Shape of Parameters:", InParams.shape)
    print("Shape of Displacement:", InDisp.shape)
    print("Shape of Shear Load:", OutShear.shape)

    if normalize_data:
        NormInParams, param_scaler = normalize(InParams, sequence=False, range=(0, 1), fit=True, save_scaler_path=data_folder / "Scaler/param_scaler.joblib")
        NormInDisp, disp_scaler = normalize(InDisp, sequence=True, range=(-1, 1), fit=True, save_scaler_path=data_folder / f"Scaler/disp_{file_suffix.lower()}_scaler.joblib")
        NormOutShear, shear_scaler = normalize(OutShear, sequence=True, range=(-1, 1), fit=True, save_scaler_path=data_folder / f"Scaler/shear_{file_suffix.lower()}_scaler.joblib")

        if save_normalized_data:
            np.savetxt(data_folder / f"Normalized/InputParameters.csv", NormInParams, delimiter=',')
            np.savetxt(data_folder / f"Normalized/Input{file_suffix}Displacement.csv", NormInDisp, delimiter=',')
            np.savetxt(data_folder / f"Normalized/Output{file_suffix}Shear.csv", NormOutShear, delimiter=',')

        return (NormInParams, NormInDisp, NormOutShear), (param_scaler, disp_scaler, shear_scaler)
    else:
        return (InParams, InDisp, OutShear), (InParams, InDisp, OutShear)


def split_and_convert(data, test_size=0.2, val_size=0.2, random_state=42, device='cpu'):
    """Splits data into train, validation, and test sets, then converts to PyTorch tensors.

      Args:
        data: A tuple of numpy arrays (X_param, X_disp, Y_shear).
        test_size: The proportion of data to use for testing.
        val_size: The proportion of data to use for validation.
        random_state: The random state for the train_test_split.
        device: The device to move tensors to (e.g., 'cpu', 'cuda').

      Returns:
        A tuple of PyTorch tensors: (X_param_train, X_disp_train, Y_shear_train,
                                    X_param_val, X_disp_val, Y_shear_val,
                                    X_param_test, X_disp_test, Y_shear_test)
      """

    X_param, X_disp, Y_shear = data

    # Convert to PyTorch tensors
    X_param = torch.tensor(X_param, dtype=torch.float32, device=device)
    X_disp = torch.tensor(X_disp, dtype=torch.float32, device=device)
    Y_shear = torch.tensor(Y_shear, dtype=torch.float32, device=device)

    # Split into train+val and test
    X_param_temp, X_param_test, X_disp_temp, X_disp_test, Y_shear_temp, Y_shear_test = train_test_split(
        X_param, X_disp, Y_shear, test_size=test_size, random_state=random_state)

    # Split train+val into train and val
    X_param_train, X_param_val, X_disp_train, X_disp_val, Y_shear_train, Y_shear_val = train_test_split(
        X_param_temp, X_disp_temp, Y_shear_temp, test_size=val_size / (1 - test_size), random_state=random_state)

    print("Shape of training :", X_param_train.shape)
    print("Shape of validation :", X_param_val.shape)
    print("Shape of testing :", X_param_test.shape)

    return (
        X_param_train, X_disp_train, Y_shear_train,
        X_param_val, X_disp_val, Y_shear_val,
        X_param_test, X_disp_test, Y_shear_test
    )
