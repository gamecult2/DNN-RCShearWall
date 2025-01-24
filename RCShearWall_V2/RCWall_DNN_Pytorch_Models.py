import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging
import sys
from torchinfo import summary
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from RCWall_Data_Processing import *
from utils.earlystopping import EarlyStopping
from Models_Response import *

def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true, y_pred = y_true.float(), y_pred.float()
    ss_total = torch.sum((y_true - y_true.mean()) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__} --- Using device: {device}")

gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

# Define hyperparameters
DATA_FOLDER = ("RCWall_Data/Run_Final_Full/FullData")
DATA_SIZE = 500
SEQUENCE_LENGTH = 500
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
TEST_SIZE = 0.10
VAL_SIZE = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 3
PATIENCE = 5

# Load and preprocess data
data, normalizer = load_data(DATA_SIZE,SEQUENCE_LENGTH, PARAMETERS_FEATURES, DATA_FOLDER, True, True)

# Split and convert data
train_splits, val_splits, test_splits = split_and_convert(data, TEST_SIZE, VAL_SIZE, 44, device, True)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(*train_splits), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_splits), BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(*test_splits), BATCH_SIZE, shuffle=True)

# Initialize model, loss, and optimizer
# model = LLaMA2_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# ==============================================================================================================
# model = xLSTM_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = AttentionLSTM_AEModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# ==============================================================================================================
# model = LSTM_AE_Model_1(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3_Optimized(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3_slice(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# ==============================================================================================================
# model = Transformer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = Transformer_Model_2(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = TransformerAEModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# ==============================================================================================================
# model = ShearTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerShearModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# ==============================================================================================================
# model = TimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# ==============================================================================================================
model = DecoderOnlyTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)

# model = torch.compile(model)
torchinfo.summary(model)

# Initialize training component
scaler = GradScaler('cuda')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-4)
criterion = nn.SmoothL1Loss().to(device)  # nn.MSELoss().to(device)
earlystop = EarlyStopping(PATIENCE, checkpoint_dir='checkpoints', model_name=f"{type(model).__name__}", save_full_model=True, verbose=False)

# Initialize tracking variables
train_losses, val_losses = [], []
train_r2_scores, val_r2_scores = [], []
best_val_loss = float("inf")
best_epoch = 0

# ================ Training phase ================
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss, epoch_train_r2 = 0.0, 0.0
    batch_count = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
                             bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | LR: {postfix[0][lr]:.6f} | Batch Loss: {postfix[0][batch_loss]:.4f} | Batch R²: {postfix[0][batch_r2]:.4f} | Avg R²: {postfix[0][avg_r2]:.4f}"),
                             postfix=[{"lr": 0.0, "batch_loss": 0.0, "batch_r2": 0.0, "avg_r2": 0.0}], leave=True)

    for batch_param, batch_disp, batch_shear in train_loader_tqdm:
        batch_param = batch_param.to(device, non_blocking=False)
        batch_disp = batch_disp.to(device, non_blocking=False)
        batch_shear = batch_shear.to(device, non_blocking=False)
        batch_count += 1

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(batch_param, batch_disp)
            loss = criterion(outputs, batch_shear)
            r2 = r2_score_torch(batch_shear, outputs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        # scheduler.step()  # Uncomment if using per-batch scheduling

        # Update metrics using in-place operations
        epoch_train_loss += loss.item()
        epoch_train_r2 += r2.item()

        # Update progress bar with real-time batch metrics
        train_loader_tqdm.postfix[0].update({
            "lr" : scheduler.get_last_lr()[0],
            "batch_loss" : loss.item(),
            "batch_r2" : r2,
            "avg_r2" : epoch_train_r2 / batch_count  # Running average
        })

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

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_param, batch_disp, batch_shear in val_loader_tqdm:
            batch_param = batch_param.to(device, non_blocking=True)
            batch_disp = batch_disp.to(device, non_blocking=True)
            batch_shear = batch_shear.to(device, non_blocking=True)
            batch_count += 1

            val_outputs = model(batch_param, batch_disp)
            batch_loss = criterion(val_outputs, batch_shear)
            batch_r2 = r2_score_torch(batch_shear, val_outputs)

            # Update validation metrics
            val_loss += batch_loss.item()
            val_r2 += batch_r2.item()

            # Update progress bar with real-time batch metrics
            val_loader_tqdm.postfix[0].update({
                "batch_loss": batch_loss,
                "batch_r2": batch_r2,
                "avg_r2": val_r2 / batch_count   # Running average
            })

    # Calculate average validation loss and R² for the epoch
    val_loss /= len(val_loader)
    val_r2 /= len(val_loader)
    val_losses.append(val_loss)
    val_r2_scores.append(val_r2)

    # Update learning rate
    scheduler.step(val_loss)

    # Print epoch summary
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Train R²: {epoch_train_r2:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}\n')

    # Early Stopping
    if earlystop(val_loss, model):
        print("Early stopping triggered")
        break

    if device.type == "cuda":
        torch.cuda.empty_cache()

best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed
print(f"Best Epoch: {best_epoch}")

# Final test evaluation =======================
model.eval()
test_loss = test_r2 = 0

with torch.no_grad(), torch.amp.autocast('cuda'):
    for batch_param, batch_disp, batch_shear in test_loader:
        batch_param = batch_param.to(device, non_blocking=True)
        batch_disp = batch_disp.to(device, non_blocking=True)
        batch_shear = batch_shear.to(device, non_blocking=True)

        test_outputs = model(batch_param, batch_disp)
        test_loss += criterion(batch_shear, test_outputs).item()
        test_r2 += r2_score_torch(batch_shear, test_outputs)

    test_loss /= len(test_loader)
    test_r2 /= len(test_loader)

print(f'Final Model Performance - Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}')

# Plotting
def plot_metric(train_data, val_data, best_epoch, ylabel, title, model_name=None, save_figure=False):
    plt.figure(figsize=(6, 5))
    epochs = range(1, len(train_data) + 1)

    plt.plot(epochs, train_data, color='#3152a1', label=f"Training {ylabel}", linewidth=2)
    plt.plot(epochs, val_data, color='red', label=f"Validation {ylabel}", linewidth=2)
    if best_epoch:
        plt.scatter(best_epoch, val_data[best_epoch - 1], color='red', s=100, label="Best Model")
    plt.xlabel("Epochs", fontname='Times New Roman', fontsize=14)
    plt.ylabel(ylabel, fontname='Times New Roman', fontsize=14)
    plt.yticks(fontname='Times New Roman', fontsize=12)
    plt.xticks(fontname='Times New Roman', fontsize=12)
    plt.title(f"{title} Over Epochs", fontname='Times New Roman', fontsize=12)
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save figure if requested
    if save_figure and model_name:
        # Create a directory to save the figures if it doesn't exist
        os.makedirs("figures", exist_ok=True)
        # Save the figure as an SVG file
        filename = f"figures/{model_name}_{title.replace(' ', '_')}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')  # Save as SVG
    # plt.show()

# Plot loss & Plot R2 score
plot_metric(train_losses, val_losses, best_epoch, "Loss", "Training and Validation Loss", f"{type(model).__name__}" , save_figure=True)
plot_metric(train_r2_scores, val_r2_scores, best_epoch, "R2 Score", "Training and Validation R2 Score", f"{type(model).__name__}" , save_figure=True)


# Select a specific test index (e.g., 2)
test_index = 20
param_scaler, disp_scaler, shear_scaler = normalizer

# Loop over the loader and get the first batch
for data, displacement, shear in test_loader:
    new_input_parameters = data[:test_index, :]
    new_input_displacement = displacement[:test_index, :]
    real_shear = shear[:test_index, :]
    break

# Restore best weights
trained_model = torch.load(f"checkpoints/{type(model).__name__}_best_full.pt", weights_only=False)
trained_model.eval()

with torch.no_grad():
    predicted_shear = trained_model(new_input_parameters, new_input_displacement)

# Move tensors to CPU for plotting and denormalization
new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear.cpu().numpy(), shear_scaler, sequence=True)

'''
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
'''

# Define the training folder
training_folder = "training"

# Create the training folder if it doesn't exist
if not os.path.exists(training_folder):
    os.makedirs(training_folder)

# Save all information to a file in the training folder
file_name = f"{type(model).__name__}_{BATCH_SIZE}.txt"
file_path = os.path.join(training_folder, file_name)  # Full file path

with open(file_path, "w", encoding="utf-8") as f:  # Specify encoding as utf-8
    f.write("=== Model Training Summary ===\n\n")
    # Save hyperparameters
    f.write("=== Hyperparameters ===\n")
    f.write(f"DATA_FOLDER: {DATA_FOLDER}\n")
    f.write(f"DATA_SIZE: {DATA_SIZE}\n")
    f.write(f"SEQUENCE_LENGTH: {SEQUENCE_LENGTH}\n")
    f.write(f"DISPLACEMENT_FEATURES: {DISPLACEMENT_FEATURES}\n")
    f.write(f"PARAMETERS_FEATURES: {PARAMETERS_FEATURES}\n")
    f.write(f"TEST_SIZE: {TEST_SIZE}\n")
    f.write(f"VAL_SIZE: {VAL_SIZE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"PATIENCE: {PATIENCE}\n")
    f.write(f"Model Type: {type(model).__name__}\n\n")

    # Save training and validation metrics in a table
    f.write("=== Training and Validation Metrics ===\n")
    f.write("| Epoch | Train Loss | Train R2 | Val Loss | Val R2  |\n")
    f.write("|-------|------------|----------|----------|---------|\n")
    for epoch in range(len(train_losses)):
        f.write(f"| {epoch + 1:^5} | {train_losses[epoch]:^10.4f} | {train_r2_scores[epoch]:^8.4f} | {val_losses[epoch]:^8.4f} | {val_r2_scores[epoch]:^7.4f} |\n")
    f.write("\n")
    f.write(f"Best Epoch: {best_epoch}\n\n")

    # Save test metrics
    f.write("=== Test Metrics ===\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test R2: {test_r2:.4f}\n\n")

    # Save best validation metrics and equivalent training metrics
    best_epoch_index = best_epoch - 1  # Convert to 0-based index
    best_val_loss = val_losses[best_epoch_index]  # Best validation loss
    best_val_r2 = val_r2_scores[best_epoch_index]  # Corresponding validation R²
    train_loss_at_best_val = train_losses[best_epoch_index]  # Training loss at best validation epoch
    train_r2_at_best_val = train_r2_scores[best_epoch_index]  # Training R² at best validation epoch

    f.write("=== Best Validation Metrics ===\n")
    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    f.write(f"Best Validation R²: {best_val_r2:.4f}\n\n")

    f.write("=== Training Metrics at Best Validation Epoch ===\n")
    f.write(f"Training Loss: {train_loss_at_best_val:.4f}\n")
    f.write(f"Training R²: {train_r2_at_best_val:.4f}\n\n")

    # Save model summary and total parameters
    f.write("=== Model Summary ===\n")
    # model_summary = summary(model, verbose=0)  # Get model summary
    total_params = sum(p.numel() for p in model.parameters())  # Calculate total parameters
    f.write(f"Total Parameters: {total_params:,}\n")  # Format with commas for readability
    # f.write(str(model_summary))  # Save the model summary

print(f"Model training summary saved to '{file_name}'")