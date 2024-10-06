import os
import joblib
from sklearn.preprocessing import MinMaxScaler


def normalization(data, scaler=None, scaler_filename=None, range=(-1, 1), sequence=False, fit=False, save_scaler_path=None):
    """Normalizes data using a MinMaxScaler, handling loading, fitting, and saving.

    Args:
        data: The data to be normalized.
        scaler: A pre-existing MinMaxScaler object (optional).
        scaler_filename: A filename to load a saved scaler from (optional).
        range: The desired feature range for the scaler (default: (-1, 1)).
        sequence: Whether to reshape the data for sequence processing (default: False).
        fit: Whether to fit the scaler to the data (default: False).
        save_scaler_path: A path to save the fitted scaler to (optional).

    Returns:
        The normalized data. Also returns the scaler object if it was created or loaded.

    Raises:
        ValueError: If neither a scaler nor a scaler filename is provided when fit=False.
        FileNotFoundError: If the specified scaler_filename does not exist.
    """

    if not fit and scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for normalization when fit=False.")

    scaler = _load_or_create_scaler(scaler, scaler_filename, range)

    data_scaled = _transform_data(data, scaler, sequence, fit)  # Pass fit explicitly

    if fit and save_scaler_path:
        joblib.dump(scaler, save_scaler_path)

    return data_scaled, scaler if scaler_filename is None else data_scaled


def _load_or_create_scaler(scaler, scaler_filename, range):
    """Loads a scaler from a file or creates a new one."""
    if scaler is None:
        if scaler_filename:
            try:
                scaler = joblib.load(scaler_filename)
            except FileNotFoundError:
                raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
        else:
            scaler = MinMaxScaler(feature_range=range)
    return scaler


def _transform_data(data, scaler, sequence, fit=False):  # Explicitly define fit as an argument
    """Transforms the data using the scaler, handling reshaping for sequences."""
    if sequence:
        data_reshaped = data.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data_reshaped) if fit else scaler.transform(data_reshaped)
        return data_scaled.reshape(data.shape)
    else:
        return scaler.fit_transform(data) if fit else scaler.transform(data)

def denormalization(data_scaled, scaler=None, scaler_filename=None, sequence=False):
    """Denormalizes data using a MinMaxScaler, handling loading and reshaping.

    Args:
        data_scaled: The scaled data to be denormalized.
        scaler: A pre-existing MinMaxScaler object (optional).
        scaler_filename: A filename to load a saved scaler from (optional).
        sequence: Whether to reshape the data for sequence processing (default: False).

    Returns:
        The denormalized data.

    Raises:
        ValueError: If neither a scaler nor a scaler filename is provided.
        FileNotFoundError: If the specified scaler_filename does not exist.
    """

    if scaler is None and scaler_filename is None:
        raise ValueError("Either a scaler or a scaler filename must be provided for denormalization.")

    scaler = _load_scaler(scaler, scaler_filename)

    return _inverse_transform_data(data_scaled, scaler, sequence)


def _load_scaler(scaler, scaler_filename):
    """Loads a scaler from a file or uses the provided one."""
    if scaler_filename:
        try:
            scaler = joblib.load(scaler_filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file '{scaler_filename}' not found.")
    return scaler


def _inverse_transform_data(data_scaled, scaler, sequence):
    """Inverse-transforms the data using the scaler, handling reshaping for sequences."""
    if sequence:
        data_reshaped = data_scaled.reshape(-1, 1)
        data_restored_1d = scaler.inverse_transform(data_reshaped)
        return data_restored_1d.reshape(data_scaled.shape)
    else:
        return scaler.inverse_transform(data_scaled)
