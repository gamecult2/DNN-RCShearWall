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

    data = np.asarray(data, dtype=np.float32) # Ensure input is numpy array

    # Load or create scaler
    if scaler is None:
        if scaler_filename:
            if not os.path.exists(scaler_filename):
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
            scaler = joblib.load(scaler_filename)
        else:
            scaler = MinMaxScaler(feature_range=range)

    # Normalize data
    if sequence:
        data_reshaped = data.reshape(-1, 1)

        if fit:
            data_scaled = scaler.fit_transform(data_reshaped)

        else:
            data_scaled = scaler.transform(data_reshaped)

        data_scaled = data_scaled.reshape(data.shape)

    else:
        if fit:
            data_scaled = scaler.fit_transform(data)
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

    data_scaled = np.asarray(data_scaled, dtype=np.float32) # Ensure input is numpy array

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


def load_data(data_size=100, sequence_length=500, normalize_data=True, analysis='CYCLIC', verbose=True):
    # ---------------------- Read Data  -------------------------------
    data_folder = Path("RCWall_Data/New_Data")  # Base data folder
    file_suffix = "Pushover" if analysis == 'PUSHOVER' else "Cyclic"

    # Read input and output data from Parquet files    # 310022
    InParams = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size].to_numpy(dtype=float)
    InDisp = pd.read_parquet(data_folder / f"Input{file_suffix}Displacement.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=float)
    OutShear = pd.read_parquet(data_folder / f"Output{file_suffix}Shear.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=float)
    if verbose:
        print(f"\nDataset shape:")
        print("  Parameters    :", InParams.shape)
        print("  Displacement  :", InDisp.shape)
        print("  Lateral Load  :", OutShear.shape)

    if normalize_data:
        NormInParams, param_scaler = normalize(InParams, sequence=False, range=(0, 1), fit=True, save_scaler_path=data_folder / "Scaler/param_scaler.joblib")
        NormInDisp, disp_scaler = normalize(InDisp, sequence=True, range=(-1, 1), fit=True, save_scaler_path=data_folder / f"Scaler/disp_{file_suffix.lower()}_scaler.joblib")
        NormOutShear, shear_scaler = normalize(OutShear, sequence=True, range=(-1, 1), fit=True, save_scaler_path=data_folder / f"Scaler/shear_{file_suffix.lower()}_scaler.joblib")
        if verbose:
            print("\nDataset Max and Mean values:")
            print("  Parameters:")
            print(f"    Max  : {np.round(np.max(InParams, axis=0), 0)}")  # Round to 2 decimal places
            print(f"    Min  : {np.round(np.min(InParams, axis=0), 2)}")
            print(f"  Displacement:")
            print(f"    Max  : {np.round(np.max(InDisp), 2)}")
            print(f"    Min  : {np.round(np.min(InDisp), 2)}")
            print(f"  Lateral Load:")
            print(f"    Max  : {np.round(np.max(OutShear), 2)}")
            print(f"    Min  : {np.round(np.min(OutShear), 2)}")
        return (NormInParams, NormInDisp, NormOutShear), (param_scaler, disp_scaler, shear_scaler)
    else:
        return (InParams, InDisp, OutShear), (InParams, InDisp, OutShear)


def split_and_convert(data, test_size=0.2, val_size=0.2, random_state=42, device='cuda', verbose=True):
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
    # Input validation
    if not len(data) == 3:
        raise ValueError(f"Expected 3 arrays in data tuple, got {len(data)}")

    X_param, X_disp, Y_shear = data

    # Check if dimensions match
    n_samples = X_param.shape[0]
    if not (X_disp.shape[0] == n_samples and Y_shear.shape[0] == n_samples):
        raise ValueError("All input arrays must have the same number of samples")

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
    if verbose:
        print(f"\nDataset splits:")
        print(f"  Training   : {X_param_train.shape[0]} -- ({100 * len(X_param_train) / n_samples:.1f}%)")
        print(f"  Validation : {X_param_val.shape[0]} -- ({100 * len(X_param_val) / n_samples:.1f}%)")
        print(f"  Testing    : {X_param_test.shape[0]} -- ({100 * len(X_param_test) / n_samples:.1f}%)")

    return (
        X_param_train, X_disp_train, Y_shear_train,
        X_param_val, X_disp_val, Y_shear_val,
        X_param_test, X_disp_test, Y_shear_test
    )
