import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torch.utils.data import DataLoader, TensorDataset
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from RCWall_Data_Processing import *
from utils.earlystopping import EarlyStopping
from Models_Response import *

# Determine the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__} --- Using device: {device}")
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.memory_stats()
    print("CUDA memory cleared.")
else:
    print("CUDA not available; no memory to clear.")

# Enable performance optimizations
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_sliding_windows(parameters, displacements, shear_outputs, window_size):
    batch_size, full_sequence_length = displacements.shape
    _, parameters_features = parameters.shape
    # Create sliding windows
    windowed_inputs = []
    windowed_outputs = []

    # Convert inputs to numpy arrays if they aren't already
    if isinstance(parameters, np.ndarray) is False:
        parameters = np.array(parameters, dtype=np.float32)
    if isinstance(displacements, np.ndarray) is False:
        displacements = np.array(displacements, dtype=np.float32)
    if isinstance(shear_outputs, np.ndarray) is False:
        shear_outputs = np.array(shear_outputs, dtype=np.float32)

    # Expand dimensions of parameters and displacements to match input dimensions
    parameters = np.repeat(parameters[:, np.newaxis, :], full_sequence_length, axis=1)  # [batch_size, full_sequence_length, parameters_features]
    displacements = displacements[..., np.newaxis]  # [batch_size, full_sequence_length, 1]

    # Combine displacements and parameters
    combined_input = np.concatenate([displacements, parameters], axis=-1)  # [batch_size, full_sequence_length, parameters_features + 1]

    # Loop through each batch and extract windows
    for b in tqdm(range(batch_size), desc="Processing batches"):
        for start in range(full_sequence_length - window_size + 1):
            window_input = combined_input[b, start:start + window_size, :]
            window_output = shear_outputs[b, start:start + window_size]

            windowed_inputs.append(window_input)
            windowed_outputs.append(window_output)

    # Convert the lists of windows to numpy arrays
    windowed_inputs_array = np.array(windowed_inputs, dtype=np.float32)
    windowed_outputs_array = np.array(windowed_outputs, dtype=np.float32)

    return windowed_inputs_array, windowed_outputs_array


# Define hyperparameters
DATA_FOLDER = "RCWall_Data/Run_Full/FullData"
DATA_SIZE = 500
SEQUENCE_LENGTH = 500
WINDOW_SIZE = 30  # Sliding window size
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
CRACK_LENGTH = 168
TEST_SIZE = 0.02
VAL_SIZE = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 5
PATIENCE = 5

# Load and preprocess data
data, scaler = load_data(DATA_SIZE, SEQUENCE_LENGTH, PARAMETERS_FEATURES, DATA_FOLDER, True, True)

# Create sliding windows for entire dataset
parameters = data[0]  # (batch, full_sequence_length, parameters_features)
# print(f"Parameters shape: {parameters.shape}")
displacements = data[1]  # (batch, full_sequence_length)
# print(f"Displacements shape: {displacements.shape}")
shear_outputs = data[2]  # (batch, full_sequence_length)
# print(f"Shear outputs shape: {shear_outputs.shape}")

# Create sliding windows
windowed_inputs, windowed_outputs = create_sliding_windows(
    parameters, displacements, shear_outputs, WINDOW_SIZE
)

sliding_data = (windowed_inputs, windowed_outputs)

# Split and convert data
(train_splits, val_splits, test_splits) = split_and_convert(sliding_data, TEST_SIZE, VAL_SIZE, 44, device, True)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(*train_splits), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_splits), BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(*test_splits), BATCH_SIZE, shuffle=False)

# Initialize model, loss, and optimizer
# model = LSTM_AE_Model_1(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_2(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = Transformer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTM_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMA2_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = AttentionLSTM_AEModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMAInspiredModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTMModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = TimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = ShearTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = EnhancedTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerShearModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = torch.compile(model)

model = SlidingWindowLSTM_AE(PARAMETERS_FEATURES + DISPLACEMENT_FEATURES, DISPLACEMENT_FEATURES, WINDOW_SIZE, SEQUENCE_LENGTH).to(device)

# Visualize the computation graph
torchinfo.summary(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, min_lr=1e-6)
criterion = nn.SmoothL1Loss().to(device)  # nn.MSELoss().to(device)
earlystop = EarlyStopping(PATIENCE, verbose=False, save_full_model=True, checkpoint_dir='../checkpoints', model_name=f"{type(model).__name__}")

# Initialize tracking variables
train_losses, val_losses, train_r2_scores, val_r2_scores = [], [], [], []
best_val_loss = float("inf")  # Track the best validation loss
best_epoch = 0  # Track the epoch number corresponding to the best validation loss

# ================ Training phase ================
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss, epoch_train_r2 = 0.0, 0.0
    batch_count = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
                             bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | Batch Loss: {postfix[0][batch_loss]:.4f} | Batch R²: {postfix[0][batch_r2]:.4f} | Avg R²: {postfix[0][avg_r2]:.4f}"),
                             postfix=[{"batch_loss": 0.0, "batch_r2": 0.0, "avg_r2": 0.0}], leave=False)

    for batch_input, batch_output in train_loader_tqdm:
        batch_input, batch_output = batch_input.to(device), batch_output.to(device)
        batch_count += 1  # Increment batch counter
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_input)

        loss = criterion(outputs, batch_output)
        r2 = r2_score(batch_output.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_r2 += r2

        # Update progress bar with real-time batch metrics
        train_loader_tqdm.postfix[0]["batch_loss"] = loss.item()
        train_loader_tqdm.postfix[0]["batch_r2"] = r2  # .item()
        train_loader_tqdm.postfix[0]["avg_r2"] = epoch_train_r2 / batch_count  # Running average
        train_loader_tqdm.update(1)

    # Calculate average training loss and R² for the epoch
    epoch_train_loss /= len(train_loader)
    epoch_train_r2 /= len(train_loader)
    train_losses.append(epoch_train_loss)
    train_r2_scores.append(epoch_train_r2)

    # ================ Validation phase ================
    model.eval()
    val_loss, val_r2 = 0.0, 0.0
    batch_count = 0

    val_loader_tqdm = tqdm(val_loader,
                           desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                           bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | Batch Loss: {postfix[0][batch_loss]:.4f} | Batch R²: {postfix[0][batch_r2]:.4f} | Avg R²: {postfix[0][avg_r2]:.4f}"),
                           postfix=[{"batch_loss": 0.0, "batch_r2": 0.0, "avg_r2": 0.0}], leave=False)

    with torch.no_grad():
        for batch_input, batch_output in val_loader_tqdm:
            batch_count += 1
            val_outputs = model(batch_input)
            batch_loss = criterion(val_outputs, batch_output)
            batch_r2 = r2_score(batch_output.detach().cpu().numpy(), val_outputs.detach().cpu().numpy())

            # Update validation metrics
            val_loss += batch_loss.item()
            val_r2 += batch_r2

            # Update progress bar with real-time batch metrics
            val_loader_tqdm.postfix[0]["batch_loss"] = batch_loss
            val_loader_tqdm.postfix[0]["batch_r2"] = batch_r2
            val_loader_tqdm.postfix[0]["avg_r2"] = val_r2 / batch_count  # Running average
            val_loader_tqdm.update(1)

    # Calculate average validation loss and R² for the epoch
    val_loss /= len(val_loader)
    val_r2 /= len(val_loader)
    val_losses.append(val_loss)
    val_r2_scores.append(val_r2)

    # Update learning rate
    lr = scheduler.get_last_lr()[0]  # Assuming a single learning rate group
    scheduler.step(val_loss)

    # Print epoch summary
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Learning Rate: {lr}, Train Loss: {epoch_train_loss:.4f}, Train R²: {epoch_train_r2:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}\n')

    # Early Stopping
    if earlystop(val_loss, model):
        print("Early stopping triggered")
        break

best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed
print(f"Best Epoch: {best_epoch}")

# Final test evaluation
model.eval()
test_loss = test_r2 = 0

with torch.no_grad():
    for batch_param, batch_disp, batch_shear in test_loader:
        test_outputs = model(batch_param, batch_disp)
        test_loss += criterion(batch_shear, test_outputs).item()
        test_r2 += r2_score(batch_shear.detach().cpu().numpy(), test_outputs.detach().cpu().numpy())

    test_loss /= len(test_loader)
    test_r2 /= len(test_loader)

print(f'Final Model Performance - Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}')


# Plotting
def plot_metric(train_data, val_data, best_epoch, ylabel, title):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_data) + 1)
    plt.plot(epochs, train_data, label=f"Training {ylabel}")
    plt.plot(epochs, val_data, label=f"Validation {ylabel}")
    if best_epoch:
        plt.scatter(best_epoch, val_data[best_epoch - 1], color='red', s=100, label="Best Model")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{title} Over Epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot loss & Plot R2 score
plot_metric(train_losses, val_losses, best_epoch, "MSE Loss", "Training and Validation Loss")
plot_metric(train_r2_scores, val_r2_scores, best_epoch, "R2 Score", "Training and Validation R2 Score")

# Select a specific test index (e.g., 2)
test_index = 3
param_scaler, disp_scaler, shear_scaler = scaler

# Loop over the loader and get the first batch
for data, displacement, shear in test_loader:
    new_input_parameters = data[:test_index, :]
    new_input_displacement = displacement[:test_index, :]
    real_shear = shear[:test_index, :]
    break

# (X_param_test, X_disp_test, Y_shear_test) = test_splits
# new_input_parameters = X_param_test[:test_index]
# new_input_displacement = X_disp_test[:test_index]
# real_shear = Y_shear_test[:test_index]


# Restore best weights
trained_model = torch.load(f"checkpoints/{type(model).__name__}_best_full.pt", weights_only=False)
trained_model.eval()

with torch.no_grad():
    predicted_shear = trained_model(new_input_parameters, new_input_displacement)

# Move tensors to CPU for plotting and denormalization
# new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
# real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
# predicted_shear = denormalize(predicted_shear.cpu().numpy(), shear_scaler, sequence=True)

new_input_displacement = denormalize2(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True, scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-3)
real_shear = denormalize2(real_shear.cpu().numpy(), shear_scaler, sequence=True, scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-3)
predicted_shear = denormalize2(predicted_shear.cpu().numpy(), shear_scaler, sequence=True, scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-3)

# plotting
for i in range(test_index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Time Series Plot
    ax1.plot(predicted_shear[i], label=f'Predicted Shear - {i + 1}')
    ax1.plot(real_shear[i], label=f'Real Shear - {i + 1}')
    ax1.set_xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax1.set_ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax1.set_title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    ax1.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
    ax1.legend()
    ax1.grid()

    # Hysteresis Loop Plot
    ax2.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    ax2.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
    ax2.set_xlabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax2.set_ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax2.set_title('Predicted Hysteresis', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    ax2.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()
