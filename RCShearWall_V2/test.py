import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from joblib import load  # Assuming joblib is used for scalers
from DNNModels import *
from functions import *
from RCWall_Data_Processing import *

# Determine the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__} --- Using device: {device}")


def main():
    # Model Hyperparameters
    PARAMETERS_FEATURES = 17  # Based on the input parameters length
    DISPLACEMENT_FEATURES = 1  # Assuming single-channel displacement
    SEQUENCE_LENGTH = 500  # As per your previous configuration
    MODEL_PATH = 'checkpoints/TimeSeriesTransformer_best_full.pt'

    # Input Parameters
    new_input_parameters = np.array([102, 102, 3650, 1220, 3.0, 190, 42, 434, 448, 448, 0.030, 0.0030, 0.0049, 0.0035, 0.080, 124440, 1])
    # new_input_parameters = np.array([378, 707, 10224, 2840, 3.6, 340, 40, 290, 610, 410, 0.0115, 0.0213, 0.0119, 0.0121, 0.003, 1297240, 1])

    # Generate loading
    num_cycles = 8
    max_displacement = 86
    repetition_cycles = 2
    num_points = math.ceil(SEQUENCE_LENGTH / (num_cycles * repetition_cycles))
    DisplacementStep = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]

    new_input_displacement = DisplacementStep
    print("\033[92m USED PARAMETERS -> (Characteristic):", new_input_parameters)

    # new_input_displacement = torch.tensor(DisplacementStep.reshape(1, -1, 1), dtype=torch.float32)
    new_input_displacement = DisplacementStep

    # Load the model
    model = TimeSeriesTransformer(
        parameters_features=PARAMETERS_FEATURES,
        displacement_features=DISPLACEMENT_FEATURES,
        sequence_length=SEQUENCE_LENGTH
    )

    # Load the state dictionary
    checkpoint = torch.load(MODEL_PATH)
    model.eval()

    # Scaler Paths
    param_scaler = 'RCWall_Data/Run_Full/FullData/Scaler/param_scaler.joblib'
    disp_scaler = 'RCWall_Data/Run_Full/FullData/Scaler/disp_scaler.joblib'
    shear_scaler = 'RCWall_Data/Run_Full/FullData/Scaler/shear_scaler.joblib'

    # ------- Normalize New data ------------------------------------------
    # Normalize new data
    new_input_parameters = normalize(new_input_parameters.reshape(1, -1), scaler_filename=param_scaler, sequence=False, fit=False)
    new_input_parameters = torch.tensor(new_input_parameters, dtype=torch.float32)

    # Normalize new input displacement using the 'normalizer' scaler
    new_input_displacement = normalize(new_input_displacement, scaler_filename=disp_scaler, sequence=True, fit=False)
    new_input_displacement = torch.tensor(new_input_displacement.reshape(1, -1), dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        predictions = model(new_input_parameters, new_input_displacement)

    # ------- Denormalize Data -------------------------------------------
    new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), scaler_filename=disp_scaler, sequence=True)
    predictions = denormalize(predictions.cpu().numpy(), scaler_filename=shear_scaler, sequence=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[0], label='Predicted Shear')
    plt.xlabel('Displacement')
    plt.ylabel('Predicted Shear')
    plt.title('Shear vs Displacement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(new_input_displacement[0], label='Predicted Shear')
    plt.xlabel('Displacement')
    plt.ylabel('Predicted Shear')
    plt.title('Shear vs Displacement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(new_input_displacement[0], predictions[0], label='Predicted Shear')
    plt.xlabel('Displacement')
    plt.ylabel('Predicted Shear')
    plt.title('Shear vs Displacement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
