from RCShearWall_V2.utils.functions import *
from RCWall_Data_Processing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__} --- Using device: {device}")


def main():
    SEQUENCE_LENGTH = 500  # As per your previous configuration

    # MODEL_PATH = 'checkpoints/LSTM_AE_Model_3_best_full.pt'
    MODEL_PATH = 'checkpoints/+++TOP+++DecoderOnlyTransformer_best_full.pt'

    # Input Parameters
    new_input_parameters = np.array([102, 102, 3650, 1220, 3.0, 190, 42, 434, 448, 448, 0.030, 0.0030, 0.0049, 0.0035, 0.080, 124440, 1])

    # Generate loading
    num_cycles = 8
    max_displacement = 86
    repetition_cycles = 2
    num_points = math.ceil(SEQUENCE_LENGTH / (num_cycles * repetition_cycles))
    new_input_displacement = generate_cyclic_loading_linear(num_cycles, max_displacement, num_points, repetition_cycles)[:SEQUENCE_LENGTH]
    # new_input_displacement = [(max(new_input_displacement) / len(new_input_displacement)) * i for i in range(int((max(new_input_displacement)) / (max(new_input_displacement) / len(new_input_displacement))))]

    print("\033[92m USED PARAMETERS -> (Characteristic):", new_input_parameters)

    # Load the model
    model = torch.load(MODEL_PATH)  # Use weights_only=True to address FutureWarning
    model.eval()

    # Scaler Paths
    param_scaler = 'RCWall_Data/Run_Final_Full/FullData/Scaler/param_scaler.joblib'
    disp_scaler = 'RCWall_Data/Run_Final_Full/FullData/Scaler/disp_scaler.joblib'
    shear_scaler = 'RCWall_Data/Run_Final_Full/FullData/Scaler/shear_scaler.joblib'

    # ------- Normalize New data ------------------------------------------
    new_input_parameters = normalize(new_input_parameters.reshape(1, -1), scaler_filename=param_scaler, sequence=False, fit=False)
    new_input_parameters = torch.tensor(new_input_parameters, dtype=torch.float32)

    new_input_displacement = normalize(new_input_displacement, scaler_filename=disp_scaler, sequence=True, fit=False)
    new_input_displacement = torch.tensor(new_input_displacement.reshape(1, -1), dtype=torch.float32)

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
    plt.plot(new_input_displacement[0], label='Displacement')
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
    Test = np.loadtxt(f"DataValidation/Thomsen_and_Wallace_RW2.txt", delimiter="\t", unpack="False")
    plt.plot(Test[0, :], Test[1, :], color="black", linewidth=1.0, linestyle="--", label='Experimental Test')
    plt.xlabel('Displacement')
    plt.ylabel('Predicted Shear')
    plt.title('Shear vs Displacement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
