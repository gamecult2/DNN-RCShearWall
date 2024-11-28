import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
# from RCWall_Cyclic_Parameters import *
import joblib
from pathlib import Path  # For path handling


# import pyarrow as pa
# import pyarrow.parquet as pq


def log_transform(data, epsilon=1e-10):
    sign = np.sign(data)
    log_data = sign * np.log1p(np.abs(data) + epsilon)
    return log_data


def inverse_log_transform(data, epsilon=1e-10):
    sign = np.sign(data)
    return sign * (np.expm1(np.abs(data)) - epsilon)


def normalize2(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, fit=False, save_scaler_path=None, scaling_strategy='minmax', handle_small_values=True, small_value_threshold=1e-3):
    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    data = np.asarray(data, dtype=np.float32)

    # Create or load scaler based on strategy
    if scaler is None:
        if scaler_filename and os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
        else:
            if scaling_strategy == 'minmax':
                scaler = MinMaxScaler(feature_range=range)
            elif scaling_strategy == 'robust':
                scaler = RobustScaler()
            elif scaling_strategy == 'log_minmax':
                scaler = MinMaxScaler(feature_range=range)
            elif scaling_strategy == 'symmetric_log':
                scaler = MinMaxScaler(feature_range=range)
            else:
                raise ValueError(f"Unknown scaling strategy: {scaling_strategy}")

    # Reshape if sequence
    if sequence:
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data

    # Apply transformations based on strategy
    if scaling_strategy == 'log_minmax':
        data_transformed = log_transform(data_reshaped)
    elif scaling_strategy == 'symmetric_log':
        data_transformed = np.sign(data_reshaped) * np.log1p(np.abs(data_reshaped))
    else:
        data_transformed = data_reshaped

    # Special handling for small values if enabled
    if handle_small_values:
        small_mask = np.abs(data_transformed) < small_value_threshold
        if np.any(small_mask):
            # Preserve the sign of small values while scaling them up
            data_transformed[small_mask] = (
                    np.sign(data_transformed[small_mask]) *
                    small_value_threshold *
                    np.abs(data_transformed[small_mask]) / small_value_threshold
            )

    # Apply scaling
    if fit:
        data_scaled = scaler.fit_transform(data_transformed)
    else:
        data_scaled = scaler.transform(data_transformed)

    # Reshape back if sequence
    if sequence:
        data_scaled = data_scaled.reshape(original_shape)

    # Save scaler if path provided
    if save_scaler_path and fit:
        joblib.dump(scaler, save_scaler_path)

    if scaler_filename:
        return data_scaled
    else:
        return data_scaled, scaler


def denormalize3(data, scaler, scaling_strategy='minmax', handle_small_values=True, small_value_threshold=1e-3):
    """
    Denormalize the data using the provided scaler
    """
    data_denorm = scaler.inverse_transform(data)

    if scaling_strategy == 'log_minmax':
        data_denorm = inverse_log_transform(data_denorm)
    elif scaling_strategy == 'symmetric_log':
        data_denorm = np.sign(data_denorm) * (np.exp(np.abs(data_denorm)) - 1)

    if handle_small_values:
        small_mask = np.abs(data_denorm) < small_value_threshold
        if np.any(small_mask):
            # Restore original scale for small values
            data_denorm[small_mask] = data_denorm[small_mask] * small_value_threshold

    return data_denorm


def denormalize2(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, scaling_strategy='minmax', handle_small_values=True, small_value_threshold=1e-3):
    if scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for denormalization.")

    data = np.asarray(data, dtype=np.float32)

    # Load scaler if not provided
    if scaler is None:
        if scaler_filename and os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
        else:
            raise ValueError("Unable to load scaler from the provided filename.")

    # Reshape if sequence
    if sequence:
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data

    # Inverse transform the scaled data
    data_inverse = scaler.inverse_transform(data_reshaped)

    # Undo transformations based on scaling strategy
    if scaling_strategy == 'log_minmax':
        # Use the specific inverse log transform
        data_restored = inverse_log_transform(data_inverse)
    elif scaling_strategy == 'symmetric_log':
        # Undo symmetric log transform
        data_restored = np.sign(data_inverse) * (np.exp(np.abs(data_inverse)) - 1)
    else:
        data_restored = data_inverse

    # Special handling for small values if enabled
    if handle_small_values:
        small_mask = np.abs(data_restored) < small_value_threshold
        if np.any(small_mask):
            # Scale back small values while preserving their sign and relative scale
            data_restored[small_mask] = (
                    np.sign(data_restored[small_mask]) *
                    small_value_threshold *
                    np.abs(data_restored[small_mask]) / small_value_threshold
            )

    # Reshape back if sequence
    if sequence:
        data_restored = data_restored.reshape(original_shape)

    return data_restored


def normalize(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, fit=False, save_scaler_path=None):
    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    data = np.asarray(data, dtype=np.float32)  # Ensure input is numpy array

    # Load or create scaler
    if scaler is None:
        if scaler_filename:
            if not os.path.exists(scaler_filename):
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
            scaler = joblib.load(scaler_filename)
        else:
            scaler = MinMaxScaler(feature_range=range)
            # scaler = RobustScaler()

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

    data_scaled = np.asarray(data_scaled, dtype=np.float32)  # Ensure input is numpy array

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


def load_data(data_size=100, sequence_length=500, input_parameters=17, data_folder="RCWall_Data/ProcessedData/FullData", normalize_data=True, verbose=True):
    # ---------------------- Read Data  -------------------------------
    # Define data and scaler folders
    data_folder = Path(data_folder)
    scaler_folder = data_folder / "Scaler"
    scaler_folder.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

    # Read input and output data from Parquet files
    InParams = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size, :input_parameters].to_numpy(dtype=float)
    InDisp = pd.read_parquet(data_folder / "InputDisplacement.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=float)
    OutShear = pd.read_parquet(data_folder / "OutputShear.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=float)
    if verbose:
        print(f"\nDataset shape:")
        print("  Parameters    :", InParams.shape)
        print("  Displacement  :", InDisp.shape)
        print("  Lateral Load  :", OutShear.shape)

    if normalize_data:
        # NormInParams, param_scaler = normalize(InParams, sequence=True, range=(0, 1), fit=True, save_scaler_path=data_folder / "Scaler/param_scaler.joblib")
        # NormInDisp, disp_scaler = normalize(InDisp, sequence=True, range=(-1, 1), fit=True, save_scaler_path=data_folder / f"Scaler/disp_scaler.joblib")
        # NormOutShear, shear_scaler = normalize(OutShear, sequence=True, range=(-1, 1), fit=True, save_scaler_path=data_folder / f"Scaler/shear_scaler.joblib")

        NormInParams, param_scaler = normalize2(InParams, sequence=False, range=(0, 1), scaling_strategy='robust', fit=True, save_scaler_path=data_folder / "Scaler/param_scaler.joblib")
        NormInDisp, disp_scaler = normalize2(InDisp, sequence=True, range=(-1, 1), scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-5, fit=True, save_scaler_path=data_folder / "Scaler/disp_scaler.joblib")
        NormOutShear, shear_scaler = normalize2(OutShear, sequence=True, range=(-1, 1), scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-5, fit=True, save_scaler_path=data_folder / "Scaler/shear_scaler.joblib")

        save_normalized_data = False
        if save_normalized_data:
            pd.DataFrame(NormInParams).to_csv(data_folder / f"Normalized/InputParameters.csv", index=False)
            pd.DataFrame(NormInDisp).to_csv(data_folder / f"Normalized/InputDisplacement.csv", index=False)
            pd.DataFrame(NormOutShear).to_csv(data_folder / f"Normalized/OutputShear.csv", index=False)

        if verbose:
            print("\nDataset Max and Mean values:")
            print("  Parameters:")
            print("    Max  :", ", ".join(f"{val:.2f}" for val in np.max(InParams, axis=0)))
            print("    Min  :", ", ".join(f"{val:.2f}" for val in np.min(InParams, axis=0)))
            print(f"  Displacement:")
            print(f"    Max  : {np.round(np.max(InDisp), 2)}")
            print(f"    Min  : {np.round(np.min(InDisp), 2)}")
            print(f"  Lateral Load:")
            print(f"    Max  : {np.round(np.max(OutShear), 2)}")
            print(f"    Min  : {np.round(np.min(OutShear), 2)}")

        return (NormInParams, NormInDisp, NormOutShear), (param_scaler, disp_scaler, shear_scaler)
    else:
        return (InParams, InDisp, OutShear), (InParams, InDisp, OutShear)


'''
def split_and_convert(data, test_size=0.2, val_size=0.2, random_state=42, device='cuda', verbose=True):
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
'''


def split_and_convert(data, test_size=0.2, val_size=0.2, random_state=42, device='cuda', verbose=True):
    # Ensure all arrays have the same number of samples
    n_samples = len(data[0])
    for array in data:
        if len(array) != n_samples:
            raise ValueError("All input arrays must have the same number of samples")

    # Convert all arrays to PyTorch tensors
    tensors = [torch.tensor(array, dtype=torch.float32, device=device) for array in data]

    # Split into Test and Temporary
    temp_splits = train_test_split(*tensors, test_size=test_size, random_state=random_state)
    test_splits = temp_splits[1::2]  # Extract Test dataset
    temp_splits = temp_splits[0::2]  # Extract Temporary dataset

    # Split Temporary into Training and Validation
    train_val_splits = train_test_split(*temp_splits, test_size=val_size / (1 - test_size), random_state=random_state)
    val_splits = train_val_splits[1::2]  # Extract Validation set
    train_splits = train_val_splits[0::2]  # Extract Training set

    if verbose:
        print(f"\nDataset splits:")
        print(f"  Training   : {len(train_splits[0])} -- ({100 * len(train_splits[0]) / n_samples:.1f}%)")
        print(f"  Validation : {len(val_splits[0])} -- ({100 * len(val_splits[0]) / n_samples:.1f}%)")
        print(f"  Testing    : {len(test_splits[0])} -- ({100 * len(test_splits[0]) / n_samples:.1f}%)")

        print(f"\nDataset splits shape:")
        for i, train_split in enumerate(train_splits):
            print(f"  Training {i + 1} shape: {train_split.shape}")

    return *train_splits, *val_splits, *test_splits
