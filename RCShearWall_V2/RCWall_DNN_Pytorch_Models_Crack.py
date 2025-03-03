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

import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from RCWall_Data_Processing import *
from utils.earlystopping import EarlyStopping
from utils.utils import *
from Models_Crack import *


# Define MAE and MSE functions
def mae_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true, y_pred = y_true.float(), y_pred.float()
    return torch.mean(torch.abs(y_true - y_pred))


def mse_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true, y_pred = y_true.float(), y_pred.float()
    return torch.mean((y_true - y_pred) ** 2)


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
DATA_FOLDER = "RCWall_Data/Run_Full2/FullData"
DATA_SIZE = 500000
SEQUENCE_LENGTH = 500
DISPLACEMENT_FEATURES = 500
SHEAR_FEATURES = 500
PARAMETERS_FEATURES = 17
CRACK_LENGTH = 168
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 150
PATIENCE = 20

# Load and preprocess data
data, normalizer = load_data_crack(DATA_SIZE, SEQUENCE_LENGTH, PARAMETERS_FEATURES, CRACK_LENGTH, DATA_FOLDER, True, True)

# Split and convert data
train_splits, val_splits, test_splits = split_and_convert(data, TEST_SIZE, VAL_SIZE, 34, device, True)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(*train_splits), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_splits), BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(*test_splits), BATCH_SIZE, shuffle=False)

# Initialize model, loss, and optimizer
# model = CrackTimeSeriesTransformer2().to(device)
# model = CrackDetectionModel3(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SHEAR_FEATURES, CRACK_LENGTH, SEQUENCE_LENGTH).to(device)
# model = EnhancedTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerShearModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = SpatialAwareCrackTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SHEAR_FEATURES, SEQUENCE_LENGTH).to(device)
# model = CrackPatternCNN().to(device)
model = CrackDetectionModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SHEAR_FEATURES, CRACK_LENGTH).to(device)
# model = CrackPatternTransformer().to(device)
# model = torch.compile(model)

# Visualize the computation graph
torchinfo.summary(model)

# Initialize training component
scaler = GradScaler('cuda')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-5)
criterion = nn.SmoothL1Loss().to(device)  # nn.MSELoss().to(device)
earlystop = EarlyStopping(PATIENCE, checkpoint_dir='checkpoints', model_name=f"{type(model).__name__}", save_full_model=True, verbose=False)

# Initialize tracking variables
train_losses, val_losses = [], []
train_r2_scores, val_r2_scores = [], []
train_mae_scores, val_mae_scores = [], []
train_mse_scores, val_mse_scores = [], []
best_val_loss = float("inf")
best_epoch = 0

# ================ Training phase ================
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss, epoch_train_r2, epoch_train_mae, epoch_train_mse = 0.0, 0.0, 0.0, 0.0
    batch_count = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
                             bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | LR: {postfix[0][lr]:.6f} | Remaining: {remaining} | R² a1: {postfix[0][r2_a1]:.4f} | R² c1: {postfix[0][r2_c1]:.4f} | R² a2: {postfix[0][r2_a2]:.4f} | R² c2: {postfix[0][r2_c2]:.4f} | Avg R²: {postfix[0][avg_r2]:.4f}"),
                             postfix=[{"lr": 0.0, "r2_a1": 0.0, "r2_c1": 0.0, "r2_a2": 0.0, "r2_c2": 0.0, "avg_r2": 0.0}], leave=False)

    for batch_param, batch_disp, batch_shear, batch_a1, batch_c1, batch_a2, batch_c2 in train_loader_tqdm:
        batch_param = batch_param.to(device, non_blocking=True)
        batch_disp = batch_disp.to(device, non_blocking=True)
        batch_shear = batch_shear.to(device, non_blocking=True)
        batch_a1 = batch_a1.to(device, non_blocking=True)
        batch_c1 = batch_c1.to(device, non_blocking=True)
        batch_a2 = batch_a2.to(device, non_blocking=True)
        batch_c2 = batch_c2.to(device, non_blocking=True)
        batch_count += 1  # Increment batch counter

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            # Forward pass with all inputs
            a1_pred, c1_pred, a2_pred, c2_pred = model(batch_param, batch_disp, batch_shear)

            loss_a1 = criterion(a1_pred, batch_a1)
            loss_c1 = criterion(c1_pred, batch_c1)
            loss_a2 = criterion(a2_pred, batch_a2)
            loss_c2 = criterion(c2_pred, batch_c2)

            loss = loss_a1 + loss_c1 + loss_a2 + loss_c2

            r2_a1 = r2_score_torch(batch_a1, a1_pred)
            r2_c1 = r2_score_torch(batch_c1, c1_pred)
            r2_a2 = r2_score_torch(batch_a2, a2_pred)
            r2_c2 = r2_score_torch(batch_c2, c2_pred)

            r2 = (r2_a1 + r2_c1 + r2_a2 + r2_c2) / 4

            # Calculate MAE and MSE using the defined functions
            mae_a1 = mae_score_torch(batch_a1, a1_pred)
            mae_c1 = mae_score_torch(batch_c1, c1_pred)
            mae_a2 = mae_score_torch(batch_a2, a2_pred)
            mae_c2 = mae_score_torch(batch_c2, c2_pred)

            mse_a1 = mse_score_torch(batch_a1, a1_pred)
            mse_c1 = mse_score_torch(batch_c1, c1_pred)
            mse_a2 = mse_score_torch(batch_a2, a2_pred)
            mse_c2 = mse_score_torch(batch_c2, c2_pred)

            mae = (mae_a1 + mae_c1 + mae_a2 + mae_c2) / 4
            mse = (mse_a1 + mse_c1 + mse_a2 + mse_c2) / 4

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        # scheduler.step()

        # Update epoch metrics
        epoch_train_loss += loss.item()
        epoch_train_r2 += r2.item()
        epoch_train_mae += mae.item()
        epoch_train_mse += mse.item()

        # Update progress bar with real-time batch metrics
        train_loader_tqdm.postfix[0].update({
            "lr": scheduler.get_last_lr()[0],
            "r2_a1": r2_a1,
            "r2_c1": r2_c1,
            "r2_a2": r2_a2,
            "r2_c2": r2_c2,
            "avg_r2": epoch_train_r2 / batch_count  # Running average
        })

    # Calculate average training loss, R², MAE, and MSE for the epoch
    epoch_train_loss /= len(train_loader)
    epoch_train_r2 /= len(train_loader)
    epoch_train_mae /= len(train_loader)
    epoch_train_mse /= len(train_loader)
    train_losses.append(epoch_train_loss)
    train_r2_scores.append(epoch_train_r2)
    train_mae_scores.append(epoch_train_mae)
    train_mse_scores.append(epoch_train_mse)

    # ================ Validation phase ================
    model.eval()
    val_loss, val_r2, val_mae, val_mse = 0.0, 0.0, 0.0, 0.0
    batch_count = 0

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                           bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | Batch Loss: {postfix[0][batch_loss]:.4f} | Batch R²: {postfix[0][batch_r2]:.4f} | Avg R²: {postfix[0][avg_r2]:.4f}"),
                           postfix=[{"batch_loss": 0.0, "batch_r2": 0.0, "avg_r2": 0.0}], leave=False)

    with torch.no_grad():
        for batch_param, batch_disp, batch_shear, batch_a1, batch_c1, batch_a2, batch_c2 in val_loader_tqdm:
            batch_param = batch_param.to(device, non_blocking=True)
            batch_disp = batch_disp.to(device, non_blocking=True)
            batch_shear = batch_shear.to(device, non_blocking=True)
            batch_a1 = batch_a1.to(device, non_blocking=True)
            batch_c1 = batch_c1.to(device, non_blocking=True)
            batch_a2 = batch_a2.to(device, non_blocking=True)
            batch_c2 = batch_c2.to(device, non_blocking=True)
            batch_count += 1

            # Forward pass with all inputs
            a1_pred, c1_pred, a2_pred, c2_pred = model(batch_param, batch_disp, batch_shear)

            # Compute validation loss and metrics
            loss_a1 = criterion(a1_pred, batch_a1)
            loss_c1 = criterion(c1_pred, batch_c1)
            loss_a2 = criterion(a2_pred, batch_a2)
            loss_c2 = criterion(c2_pred, batch_c2)

            val_batch_loss = loss_a1 + loss_c1 + loss_a2 + loss_c2
            val_loss += val_batch_loss.item()

            # R² calculations
            r2_a1 = r2_score_torch(batch_a1, a1_pred).item()
            r2_c1 = r2_score_torch(batch_c1, c1_pred).item()
            r2_a2 = r2_score_torch(batch_a2, a2_pred).item()
            r2_c2 = r2_score_torch(batch_c2, c2_pred).item()
            current_batch_r2 = (r2_a1 + r2_c1 + r2_a2 + r2_c2) / 4
            val_r2 += current_batch_r2

            # MAE calculations
            mae_a1 = mae_score_torch(batch_a1, a1_pred).item()
            mae_c1 = mae_score_torch(batch_c1, c1_pred).item()
            mae_a2 = mae_score_torch(batch_a2, a2_pred).item()
            mae_c2 = mae_score_torch(batch_c2, c2_pred).item()
            current_batch_mae = (mae_a1 + mae_c1 + mae_a2 + mae_c2) / 4
            val_mae += current_batch_mae

            # MSE calculations
            mse_a1 = mse_score_torch(batch_a1, a1_pred).item()
            mse_c1 = mse_score_torch(batch_c1, c1_pred).item()
            mse_a2 = mse_score_torch(batch_a2, a2_pred).item()
            mse_c2 = mse_score_torch(batch_c2, c2_pred).item()
            current_batch_mse = (mse_a1 + mse_c1 + mse_a2 + mse_c2) / 4
            val_mse += current_batch_mse

            # Update progress bar with CURRENT BATCH metrics
            val_loader_tqdm.postfix[0].update({
                "batch_loss": val_batch_loss.item(),
                "batch_r2": current_batch_r2,
                "avg_r2": val_r2 / batch_count
            })

    # Calculate average validation loss, R², MAE, and MSE for the epoch
    val_loss /= len(val_loader)
    val_r2 /= len(val_loader)
    val_mae /= len(val_loader)
    val_mse /= len(val_loader)
    val_losses.append(val_loss)
    val_r2_scores.append(val_r2)
    val_mae_scores.append(val_mae)
    val_mse_scores.append(val_mse)

    # Print epoch summary
    print(f'Epoch [{epoch + 1}/{EPOCHS}], '
          f'Train Loss: {epoch_train_loss:.4f}, '
          f'Train MAE: {epoch_train_mae:.4f}, '
          f'Train MSE: {epoch_train_mse:.4f},'
          f'Train R²: {epoch_train_r2:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val MAE: {val_mae:.4f}, '
          f'Val MSE: {val_mse:.4f}, '
          f'Val R²: {val_r2:.4f}\n')

    # Update learning rate
    scheduler.step(val_loss)

    # Early Stopping
    if earlystop(val_loss, model):
        print("Early stopping triggered")
        break

best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed
print(f"Best Epoch: {best_epoch}")

# Final test evaluation
model.eval()
test_loss, test_r2, test_mae, test_mse = 0.0, 0.0, 0.0, 0.0

with torch.no_grad():
    for batch_param, batch_disp, batch_shear, batch_a1, batch_c1, batch_a2, batch_c2 in test_loader:
        batch_param = batch_param.to(device, non_blocking=True)
        batch_disp = batch_disp.to(device, non_blocking=True)
        batch_shear = batch_shear.to(device, non_blocking=True)
        batch_a1 = batch_a1.to(device, non_blocking=True)
        batch_c1 = batch_c1.to(device, non_blocking=True)
        batch_a2 = batch_a2.to(device, non_blocking=True)
        batch_c2 = batch_c2.to(device, non_blocking=True)

        # Forward pass with all inputs
        a1_pred, c1_pred, a2_pred, c2_pred = model(batch_param, batch_disp, batch_shear)

        # Compute test loss for each output
        test_loss_a1 = criterion(a1_pred, batch_a1)
        test_loss_c1 = criterion(c1_pred, batch_c1)
        test_loss_a2 = criterion(a2_pred, batch_a2)
        test_loss_c2 = criterion(c2_pred, batch_c2)

        # Total test loss
        batch_test_loss = test_loss_a1 + test_loss_c1 + test_loss_a2 + test_loss_c2
        test_loss += batch_test_loss.item()

        # Compute R² for each output using torch method
        test_r2_a1 = r2_score_torch(batch_a1, a1_pred)
        test_r2_c1 = r2_score_torch(batch_c1, c1_pred)
        test_r2_a2 = r2_score_torch(batch_a2, a2_pred)
        test_r2_c2 = r2_score_torch(batch_c2, c2_pred)

        # Average R²
        batch_test_r2 = (test_r2_a1 + test_r2_c1 + test_r2_a2 + test_r2_c2) / 4
        test_r2 += batch_test_r2

        # Compute MAE and MSE for each output
        batch_mae_a1 = mae_score_torch(batch_a1, a1_pred)
        batch_mae_c1 = mae_score_torch(batch_c1, c1_pred)
        batch_mae_a2 = mae_score_torch(batch_a2, a2_pred)
        batch_mae_c2 = mae_score_torch(batch_c2, c2_pred)

        batch_mse_a1 = mse_score_torch(batch_a1, a1_pred)
        batch_mse_c1 = mse_score_torch(batch_c1, c1_pred)
        batch_mse_a2 = mse_score_torch(batch_a2, a2_pred)
        batch_mse_c2 = mse_score_torch(batch_c2, c2_pred)

        # Total MAE and MSE for the batch
        batch_mae = (batch_mae_a1 + batch_mae_c1 + batch_mae_a2 + batch_mae_c2) / 4
        batch_mse = (batch_mse_a1 + batch_mse_c1 + batch_mse_a2 + batch_mse_c2) / 4

        # Accumulate MAE and MSE for average calculation
        test_mae += batch_mae.item()
        test_mse += batch_mse.item()

# Average test loss, R², MAE, and MSE
test_loss /= len(test_loader)
test_r2 /= len(test_loader)
test_mae /= len(test_loader)
test_mse /= len(test_loader)

# Print the final test performance
print(f'Final Model Performance - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}')

# # Plot loss & Plot R2 score
# plot_metric(train_losses, val_losses, best_epoch, test_loss, "Loss", "Loss", f"{type(model).__name__}", save_fig=True)
# plot_metric(train_r2_scores, val_r2_scores, best_epoch, test_r2, "R2 Score", "R2 Score", f"{type(model).__name__}", save_fig=True)

# Select a specific test index (e.g., 2)
test_index = 20
param_scaler, disp_scaler, shear_scaler, outa1_scaler, outc1_scaler, outa2_scaler, outc2_scaler = normalizer

# Loop over the loader and get the first batch
for data, displacement, shear, a1, c1, a2, c2 in test_loader:
    new_input_parameters = data[:test_index, :]
    new_input_displacement = displacement[:test_index, :]
    real_shear = shear[:test_index, :]
    real_a1 = a1[:test_index, :]
    real_c1 = c1[:test_index, :]
    real_a2 = a2[:test_index, :]
    real_c2 = c2[:test_index, :]
    break

# Restore best weights
trained_model = torch.load(f"checkpoints/{type(model).__name__}_best_full.pt", weights_only=False)
trained_model.eval()

with torch.no_grad():
    a1_pred, c1_pred, a2_pred, c2_pred = trained_model(new_input_parameters, new_input_displacement, real_shear)

# Move tensors to CPU for plotting and denormalization
new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
real_a1 = denormalize(real_a1.cpu().numpy(), outa1_scaler, sequence=True)
real_c1 = denormalize(real_c1.cpu().numpy(), outc1_scaler, sequence=True)
real_a2 = denormalize(real_a2.cpu().numpy(), outa2_scaler, sequence=True)
real_c2 = denormalize(real_c2.cpu().numpy(), outc2_scaler, sequence=True)

# Denormalize predicted values
a1_pred = denormalize(a1_pred.cpu().numpy(), outa1_scaler, sequence=True)
c1_pred = denormalize(c1_pred.cpu().numpy(), outc1_scaler, sequence=True)
a2_pred = denormalize(a2_pred.cpu().numpy(), outa2_scaler, sequence=True)
c2_pred = denormalize(c2_pred.cpu().numpy(), outc2_scaler, sequence=True)

# plotting
for i in range(test_index):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # A1 Comparison
    ax1.plot(a1_pred[i], label=f'A1 Predicted - {i + 1}', color='red')
    ax1.plot(real_a1[i], label=f'A1 Real - {i + 1}', color='blue', linestyle='--')
    ax1.set_xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax1.set_ylabel('A1 Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax1.set_title('A1 Comparison', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    ax1.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
    ax1.legend()
    ax1.grid()

    # C1 Comparison
    ax2.plot(c1_pred[i], label=f'C1 Predicted - {i + 1}', color='red')
    ax2.plot(real_c1[i], label=f'C1 Real - {i + 1}', color='blue', linestyle='--')
    ax2.set_xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax2.set_ylabel('C1 Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax2.set_title('C1 Comparison', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    ax2.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
    ax2.legend()
    ax2.grid()

    # A2 Comparison
    ax3.plot(a2_pred[i], label=f'A2 Predicted - {i + 1}', color='red')
    ax3.plot(real_a2[i], label=f'A2 Real - {i + 1}', color='blue', linestyle='--')
    ax3.set_xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax3.set_ylabel('A2 Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax3.set_title('A2 Comparison', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    ax3.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
    ax3.legend()
    ax3.grid()

    # C2 Comparison
    ax4.plot(c2_pred[i], label=f'C2 Predicted - {i + 1}', color='red')
    ax4.plot(real_c2[i], label=f'C2 Real - {i + 1}', color='blue', linestyle='--')
    ax4.set_xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax4.set_ylabel('C2 Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    ax4.set_title('C2 Comparison', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    ax4.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
    ax4.legend()
    ax4.grid()

    plt.tight_layout()
    plt.show()
