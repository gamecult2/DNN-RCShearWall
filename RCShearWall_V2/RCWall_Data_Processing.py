import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, Normalizer, PowerTransformer
import os
# from RCWall_Cyclic_Parameters import *
import joblib
from pathlib import Path  # For path handling


# =================================================================================================================================================================
def normalize(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, scaling_strategy='minmax', fit=False, save_scaler_path=None):
    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    # Ensure input is a NumPy array
    data = np.asarray(data, dtype=np.float32)

    # Load or create scaler
    if scaler is None:
        if scaler_filename:
            if os.path.exists(scaler_filename):
                scaler = joblib.load(scaler_filename)
            else:
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
        else:
            # Define scaler mapping
            scaler_mapping = {
                'minmax': MinMaxScaler(feature_range=range),
                'robust': RobustScaler(),
                'standard': StandardScaler(),
                'maxabs': MaxAbsScaler(),
                'quantile': QuantileTransformer(output_distribution='normal'),
                'power': PowerTransformer(method='yeo-johnson')
            }
            if scaling_strategy in scaler_mapping:
                scaler = scaler_mapping[scaling_strategy]
            else:
                raise ValueError(f"Unknown scaling strategy: {scaling_strategy}. "
                                 f"Available strategies: {list(scaler_mapping.keys())}")

    # Handle sequence data
    if sequence:
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data

    # Scale data
    if fit:
        data_scaled = scaler.fit_transform(data_reshaped)
    else:
        data_scaled = scaler.transform(data_reshaped)

    # Reshape back if sequence
    if sequence:
        data_scaled = data_scaled.reshape(original_shape)

    # Save the scaler if specified
    if save_scaler_path and fit:
        joblib.dump(scaler, save_scaler_path)

    # Return results
    if scaler_filename:
        return data_scaled
    else:
        return data_scaled, scaler


def denormalize(data, scaler=None, scaler_filename=None, sequence=False):
    if scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for denormalization.")

    data = np.asarray(data, dtype=np.float32)

    if scaler is None:
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
        else:
            raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")

    if sequence:
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data

    try:
        data_restored = scaler.inverse_transform(data_reshaped)
    except Exception as e:
        raise ValueError(f"Error during inverse transformation: {str(e)}")

    if sequence:
        data_restored = data_restored.reshape(original_shape)

    return data_restored


# =================================================================================================================================================================

def load_data(data_size=100, sequence_length=500, input_parameters=17, loading_type="full", data_folder="RCWall_Data/ProcessedData/FullData", normalize_data=True, verbose=True):
    # Define data and scaler folders
    data_folder = Path(data_folder)
    scaler_folder = data_folder / "Scaler"
    scaler_folder.mkdir(parents=True, exist_ok=True)

    # Read input and output data from Parquet files
    InParams = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size, :input_parameters].to_numpy(dtype=np.float32)
    InDisp = pd.read_parquet(data_folder / "InputDisplacement.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=np.float32)
    OutShear = pd.read_parquet(data_folder / "OutputShear.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=np.float32)
    # Read InputParameters first to create appropriate mask
    input_params_df = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size]

    # Create mask based on loading type
    if loading_type == "cyclic":
        mask = input_params_df.iloc[:, -1] == 0
    elif loading_type == "monotonic":
        mask = input_params_df.iloc[:, -1] == 1
    else:  # full
        mask = pd.Series([True] * len(input_params_df))

    # Load and filter data
    InParams = input_params_df[mask].iloc[:, :input_parameters].to_numpy(dtype=np.float32)
    InDisp = pd.read_parquet(data_folder / "InputDisplacement.parquet").iloc[:data_size][mask].iloc[:, :sequence_length].to_numpy(dtype=np.float32)
    OutShear = pd.read_parquet(data_folder / "OutputShear.parquet").iloc[:data_size][mask].iloc[:, :sequence_length].to_numpy(dtype=np.float32)

    if verbose:
        print(f"\nDataset shape and type:")
        print("  Parameters    : Shape =", InParams.shape, " , Type =", InParams.dtype)
        print("  Displacement  : Shape =", InDisp.shape, ", Type =", InDisp.dtype)
        print("  Lateral Load  : Shape =", OutShear.shape, ", Type =", OutShear.dtype)

    if normalize_data:
        NormInParams, param_scaler = normalize(InParams, sequence=False, range=(0, 1), scaling_strategy='robust', fit=True, save_scaler_path=data_folder / "Scaler/param_scaler.joblib")
        NormInDisp, disp_scaler = normalize(InDisp, sequence=True, range=(-1, 1), scaling_strategy='robust', fit=True, save_scaler_path=data_folder / "Scaler/disp_scaler.joblib")
        NormOutShear, shear_scaler = normalize(OutShear, sequence=True, range=(-1, 1), scaling_strategy='robust', fit=True, save_scaler_path=data_folder / "Scaler/shear_scaler.joblib")

        save_normalized_data = False
        if save_normalized_data:
            pd.DataFrame(NormInParams).to_csv(data_folder / f"Normalized/InputParameters.csv", index=False)
            pd.DataFrame(NormInDisp).to_csv(data_folder / f"Normalized/InputDisplacement.csv", index=False)
            pd.DataFrame(NormOutShear).to_csv(data_folder / f"Normalized/OutputShear.csv", index=False)

        if verbose:
            print(f"\nDataset Max and Mean values:")
            print("  Parameters:")
            print("    Max  :", ", ".join(f"{val:.2f}" for val in np.max(InParams, axis=0)))
            print("    Min  :", ", ".join(f"{val:.2f}" for val in np.min(InParams, axis=0)))
            print("  Normalized Parameters:")
            print("    Max  :", ", ".join(f"{val:.2f}" for val in np.max(NormInParams, axis=0)))
            print("    Min  :", ", ".join(f"{val:.2f}" for val in np.min(NormInParams, axis=0)))
            print(f"  Displacement:")
            print(f"    Max  : {np.round(np.max(InDisp), 3)}    |      Max  : {np.round(np.max(NormInDisp), 3)}    |     {(np.max(InDisp)).dtype}")
            print(f"    Min  : {np.round(np.min(InDisp), 3)}   |       Min  : {np.round(np.min(NormInDisp), 3)}    |     {(np.max(InDisp)).dtype}")
            print(f"  Lateral Load:")
            print(f"    Max  : {np.round(np.max(OutShear), 3)}      |      Max  : {np.round(np.max(NormOutShear), 3)}    |     {(np.max(InDisp)).dtype}")
            print(f"    Min  : {np.round(np.min(OutShear), 3)}     |      Min  : {np.round(np.min(NormOutShear), 3)}    |     {(np.max(InDisp)).dtype}")

        return (NormInParams, NormInDisp, NormOutShear), (param_scaler, disp_scaler, shear_scaler)
    else:
        return (InParams, InDisp, OutShear)


def load_data_crack(data_size=100, sequence_length=500, input_parameters=17, crack_length=168, data_folder="RCWall_Data/ProcessedData/FullData", normalize_data=True, verbose=True):
    # Define data and scaler folders
    data_folder = Path(data_folder)
    scaler_folder = data_folder / "Scaler"
    scaler_folder.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

    # Read input and output data from Parquet files
    # InParams = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size, :input_parameters].to_numpy(dtype=np.float32)
    # InDisp = pd.read_parquet(data_folder / "InputDisplacement.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=np.float32)
    # OutShear = pd.read_parquet(data_folder / "OutputShear.parquet").iloc[:data_size, :sequence_length].to_numpy(dtype=np.float32)
    # # New outputs for a1, c1, a2, c2
    # Outa1 = pd.read_parquet(data_folder / "a1.parquet").iloc[:data_size, :crack_length].replace(10, 0).to_numpy(dtype=np.float32)
    # Outc1 = pd.read_parquet(data_folder / "c1.parquet").iloc[:data_size, :crack_length].to_numpy(dtype=np.float32)
    # Outa2 = pd.read_parquet(data_folder / "a2.parquet").iloc[:data_size, :crack_length].replace(10, 0).to_numpy(dtype=np.float32)
    # Outc2 = pd.read_parquet(data_folder / "c2.parquet").iloc[:data_size, :crack_length].to_numpy(dtype=np.float32)

    # First read InputParameters and create a mask for rows where last column is 0
    InputParameters = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size]
    mask = InputParameters.iloc[:, -1] == 0  # Create boolean mask for last column == 0

    # Apply the mask when reading each file
    InParams = InputParameters[mask].iloc[:, :input_parameters].to_numpy(dtype=np.float32)
    InDisp = pd.read_parquet(data_folder / "InputDisplacement.parquet").iloc[:data_size][mask].iloc[:, :sequence_length].to_numpy(dtype=np.float32)
    OutShear = pd.read_parquet(data_folder / "OutputShear.parquet").iloc[:data_size][mask].iloc[:, :sequence_length].to_numpy(dtype=np.float32)

    # Apply the same mask to the crack-related outputs
    Outa1 = pd.read_parquet(data_folder / "a1.parquet").iloc[:data_size][mask].iloc[:, :crack_length].replace(10, 0).to_numpy(dtype=np.float32)
    Outc1 = pd.read_parquet(data_folder / "c1.parquet").iloc[:data_size][mask].iloc[:, :crack_length].to_numpy(dtype=np.float32)
    Outa2 = pd.read_parquet(data_folder / "a2.parquet").iloc[:data_size][mask].iloc[:, :crack_length].replace(10, 0).to_numpy(dtype=np.float32)
    Outc2 = pd.read_parquet(data_folder / "c2.parquet").iloc[:data_size][mask].iloc[:, :crack_length].to_numpy(dtype=np.float32)
    if verbose:
        print(f"\nDataset shape:")
        print("  Parameters   :", InParams.shape)
        print("  Displacement :", InDisp.shape)
        print("  Lateral Load :", OutShear.shape)
        print("  a1           :", Outa1.shape)
        print("  c1           :", Outc1.shape)
        print("  a2           :", Outa2.shape)
        print("  c2           :", Outc2.shape)

    if normalize_data:
        NormInParams, param_scaler = normalize(InParams,
                                                sequence=False,
                                                range=(0, 1),
                                                scaling_strategy='robust',
                                                fit=True,
                                                save_scaler_path=data_folder / "Scaler/param_scaler.joblib")

        NormInDisp, disp_scaler = normalize(InDisp,
                                             sequence=True,
                                             range=(-1, 1),
                                             scaling_strategy='robust',
                                             fit=True,
                                             save_scaler_path=data_folder / "Scaler/disp_scaler.joblib")

        NormOutShear, shear_scaler = normalize(OutShear,
                                                sequence=True,
                                                range=(-1, 1),
                                                scaling_strategy='robust',
                                                fit=True,
                                                save_scaler_path=data_folder / "Scaler/shear_scaler.joblib")

        NormOuta1, outa1_scaler = normalize(Outa1,
                                             sequence=True,
                                             range=(-1, 1),
                                             scaling_strategy='robust',
                                             fit=True,
                                             save_scaler_path=data_folder / "Scaler/outa1_scaler.joblib")

        NormOutc1, outc1_scaler = normalize(Outc1,
                                             sequence=True,
                                             range=(-1, 1),
                                             scaling_strategy='robust',
                                             fit=True,
                                             save_scaler_path=data_folder / "Scaler/outc1_scaler.joblib")

        NormOuta2, outa2_scaler = normalize(Outa2,
                                             sequence=True,
                                             range=(-1, 1),
                                             scaling_strategy='robust',
                                             fit=True,
                                             save_scaler_path=data_folder / "Scaler/outa2_scaler.joblib")

        NormOutc2, outc2_scaler = normalize(Outc2,
                                             sequence=True,
                                             range=(-1, 1),
                                             scaling_strategy='robust',
                                             fit=True,
                                             save_scaler_path=data_folder / "Scaler/outc2_scaler.joblib")

        # Optional: save normalized data
        save_normalized_data = False
        if save_normalized_data:
            pd.DataFrame(NormInParams).to_csv(data_folder / f"Normalized/InputParameters.csv", index=False)
            pd.DataFrame(NormInDisp).to_csv(data_folder / f"Normalized/InputDisplacement.csv", index=False)
            pd.DataFrame(NormOutShear).to_csv(data_folder / f"Normalized/OutputShear.csv", index=False)
            pd.DataFrame(NormOuta1).to_csv(data_folder / f"Normalized/Outa1.csv", index=False)
            pd.DataFrame(NormOutc1).to_csv(data_folder / f"Normalized/Outc1.csv", index=False)
            pd.DataFrame(NormOuta2).to_csv(data_folder / f"Normalized/Outa2.csv", index=False)
            pd.DataFrame(NormOutc2).to_csv(data_folder / f"Normalized/Outc2.csv", index=False)

        # Verbose output of max and min values
        if verbose:
            print("\nDataset Max and Mean values:")
            print("  Parameters:")
            print("    Max  :", ", ".join(f"{val:.2f}" for val in np.max(InParams, axis=0)))
            print("    Min  :", ", ".join(f"{val:.2f}" for val in np.min(InParams, axis=0)))
            print(f"  Displacement:")
            print(f"    Max  : {np.round(np.max(InDisp), 3)}      |      Max  : {np.round(np.max(NormInDisp), 2)}")
            print(f"    Min  : {np.round(np.min(InDisp), 3)}     |       Min  : {np.round(np.min(NormInDisp), 2)}")
            print(f"  Lateral Load:")
            print(f"    Max  : {np.round(np.max(OutShear), 3)}      |      Max  : {np.round(np.max(NormOutShear), 2)}")
            print(f"    Min  : {np.round(np.min(OutShear), 3)}     |      Min  : {np.round(np.min(NormOutShear), 2)}")
            print(f"  Angle 1:")
            print(f"    Max  : {np.round(np.max(Outa1), 5)}      |      Max  : {np.round(np.max(NormOuta1), 5)}")
            print(f"    Min  : {np.round(np.min(Outa1), 5)}     |      Min  : {np.round(np.min(NormOuta1), 5)}")
            print(f"  Crack 1:")
            print(f"    Max  : {np.round(np.max(Outc1), 5)}      |      Max  : {np.round(np.max(NormOutc1), 5)}")
            print(f"    Min  : {np.round(np.min(Outc1), 5)}     |      Min  : {np.round(np.min(NormOutc1), 5)}")
            print(f"  Angle 2:")
            print(f"    Max  : {np.round(np.max(Outa2), 5)}      |      Max  : {np.round(np.max(NormOuta2), 5)}")
            print(f"    Min  : {np.round(np.min(Outa2), 5)}     |      Min  : {np.round(np.min(NormOuta2), 5)}")
            print(f"  Crack 2:")
            print(f"    Max  : {np.round(np.max(Outc2), 5)}      |      Max  : {np.round(np.max(NormOutc2), 5)}")
            print(f"    Min  : {np.round(np.min(Outc2), 5)}     |      Min  : {np.round(np.min(NormOutc2), 5)}")

        # Return normalized data and scalers
        return (NormInParams, NormInDisp, NormOutShear, NormOuta1, NormOutc1, NormOuta2, NormOutc2), \
            (param_scaler, disp_scaler, shear_scaler, outa1_scaler, outc1_scaler, outa2_scaler, outc2_scaler)
    else:
        # Return raw data if normalization is not requested
        return (InParams, InDisp, OutShear, Outa1, Outc1, Outa2, Outc2)


def split_and_convert(data, test_size=0.2, val_size=0.2, random_state=42, device=None, verbose=True):
    # Ensure all arrays have the same number of samples
    n_samples = len(data[0])
    for array in data:
        if len(array) != n_samples:
            raise ValueError("All input arrays must have the same number of samples")

    # Check for device availability and set default to CPU if 'cuda' is not available
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    return train_splits, val_splits, test_splits


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
