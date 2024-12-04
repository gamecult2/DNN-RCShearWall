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

from RCShearWall_V2.DNNModels import CrackTimeSeriesTransformer
from RCWall_Data_Processing import *
from utils.earlystopping import EarlyStopping
from DNNModels import *

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

# Define hyperparameters
DATA_FOLDER = "RCWall_Data/Run_Full/FullData"
DATA_SIZE = 200000
SEQUENCE_LENGTH = 500
DISPLACEMENT_FEATURES = 1
SHEAR_FEATURES = 1
PARAMETERS_FEATURES = 17
CRACK_LENGTH = 168
TEST_SIZE = 0.05
VAL_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
EPOCHS = 3
PATIENCE = 5

# Load and preprocess data
data, scaler = load_data_crack(DATA_SIZE, SEQUENCE_LENGTH, PARAMETERS_FEATURES, CRACK_LENGTH, DATA_FOLDER, True, True)

# Split and convert data
(train_splits, val_splits, test_splits) = split_and_convert(data, TEST_SIZE, VAL_SIZE, 44, device, True)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(*train_splits), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(*val_splits), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(*test_splits), batch_size=BATCH_SIZE, shuffle=False)

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
model = CrackTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SHEAR_FEATURES, SEQUENCE_LENGTH).to(device)
# model = EnhancedTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerShearModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = torch.compile(model)

# Visualize the computation graph
torchinfo.summary(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, min_lr=1e-6)
criterion = nn.SmoothL1Loss().to(device)  # nn.MSELoss().to(device)
early_stopping = EarlyStopping(PATIENCE, verbose=False, save_full_model=True, checkpoint_dir='checkpoints', model_name=f"{type(model).__name__}")

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
                             bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | Loss a1: {postfix[0][r2_a1]:.4f} | Loss c1: {postfix[0][r2_c1]:.4f} | Loss a2: {postfix[0][r2_a2]:.4f} | Loss c2: {postfix[0][r2_c2]:.4f} | Avg R²: {postfix[0][avg_r2]:.4f}"),
                             postfix=[{"r2_a1": 0.0, "r2_c1": 0.0, "r2_a2": 0.0, "r2_c2": 0.0, "avg_r2": 0.0}], leave=False)

    for batch_param, batch_disp, batch_shear, batch_a1, batch_c1, batch_a2, batch_c2 in train_loader_tqdm:
        batch_count += 1  # Increment batch counter
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad() Zero gradients

        # Forward pass with all inputs
        a1_pred, c1_pred, a2_pred, c2_pred = model(batch_param, batch_disp, batch_shear)

        # Compute loss for each output
        loss_a1 = criterion(a1_pred, batch_a1)
        loss_c1 = criterion(c1_pred, batch_c1)
        loss_a2 = criterion(a2_pred, batch_a2)
        loss_c2 = criterion(c2_pred, batch_c2)

        # Total loss (you can adjust weights if needed)
        loss = loss_a1 + loss_c1 + loss_a2 + loss_c2

        # Compute R² for each output (you might want to average or choose a specific metric)
        r2_a1 = r2_score(batch_a1.detach().cpu().numpy(), a1_pred.detach().cpu().numpy())
        r2_c1 = r2_score(batch_c1.detach().cpu().numpy(), c1_pred.detach().cpu().numpy())
        r2_a2 = r2_score(batch_a2.detach().cpu().numpy(), a2_pred.detach().cpu().numpy())
        r2_c2 = r2_score(batch_c2.detach().cpu().numpy(), c2_pred.detach().cpu().numpy())

        # Average R²
        r2 = (r2_a1 + r2_c1 + r2_a2 + r2_c2) / 4

        # Backward pass and optimizer step
        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()  # Update parameters

        # Update epoch metrics
        epoch_train_loss += loss.item()
        epoch_train_r2 += r2  # .item()

        # Update progress bar with real-time batch metrics
        train_loader_tqdm.postfix[0]["r2_a1"] = r2_a1.item()
        train_loader_tqdm.postfix[0]["r2_c1"] = r2_c1.item()
        train_loader_tqdm.postfix[0]["r2_a2"] = r2_a2.item()
        train_loader_tqdm.postfix[0]["r2_c2"] = r2_c2.item()
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
        for batch_param, batch_disp, batch_shear, batch_a1, batch_c1, batch_a2, batch_c2 in val_loader_tqdm:
            batch_count += 1

            # Forward pass with all inputs
            a1_pred, c1_pred, a2_pred, c2_pred = model(batch_param, batch_disp, batch_shear)

            # Compute loss for each output
            batch_loss_a1 = criterion(a1_pred, batch_a1)
            batch_loss_c1 = criterion(c1_pred, batch_c1)
            batch_loss_a2 = criterion(a2_pred, batch_a2)
            batch_loss_c2 = criterion(c2_pred, batch_c2)

            # Total loss (you can adjust weights if needed)
            batch_loss = batch_loss_a1 + batch_loss_c1 + batch_loss_a2 + batch_loss_c2

            # Compute R² for each output
            batch_r2_a1 = r2_score(batch_a1.detach().cpu().numpy(), a1_pred.detach().cpu().numpy())
            batch_r2_c1 = r2_score(batch_c1.detach().cpu().numpy(), c1_pred.detach().cpu().numpy())
            batch_r2_a2 = r2_score(batch_a2.detach().cpu().numpy(), a2_pred.detach().cpu().numpy())
            batch_r2_c2 = r2_score(batch_c2.detach().cpu().numpy(), c2_pred.detach().cpu().numpy())

            # Average R²
            batch_r2 = (batch_r2_a1 + batch_r2_c1 + batch_r2_a2 + batch_r2_c2) / 4

            # Update validation metrics
            val_loss += batch_loss.item()
            val_r2 += batch_r2

            # Update progress bar with real-time batch metrics
            val_loader_tqdm.postfix[0]["batch_loss"] = batch_loss.item()
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
    if early_stopping(val_loss, model):
        print("Early stopping triggered")
        break

best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed
print(f"Best Epoch: {best_epoch}")

# Final test evaluation
model.eval()
test_loss = test_r2 = 0

with torch.no_grad():
    for batch_param, batch_disp, batch_shear, batch_a1, batch_c1, batch_a2, batch_c2 in test_loader:
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

        # Compute R² for each output
        test_r2_a1 = r2_score(batch_a1.detach().cpu().numpy(), a1_pred.detach().cpu().numpy())
        test_r2_c1 = r2_score(batch_c1.detach().cpu().numpy(), c1_pred.detach().cpu().numpy())
        test_r2_a2 = r2_score(batch_a2.detach().cpu().numpy(), a2_pred.detach().cpu().numpy())
        test_r2_c2 = r2_score(batch_c2.detach().cpu().numpy(), c2_pred.detach().cpu().numpy())

        # Average R²
        batch_test_r2 = (test_r2_a1 + test_r2_c1 + test_r2_a2 + test_r2_c2) / 4
        test_r2 += batch_test_r2

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

# Plotting results
test_index = 2
input_param = test_splits[0][:test_index]
input_disp = test_splits[1][:test_index]
input_shear = test_splits[2][:test_index]

# Real target values
real_a1 = test_splits[3][:test_index]
real_c1 = test_splits[4][:test_index]
real_a2 = test_splits[5][:test_index]
real_c2 = test_splits[6][:test_index]

# Restore best weights
trained_model = torch.load(f"checkpoints/{type(model).__name__}_best_full.pt", weights_only=False)
trained_model.eval()

with torch.no_grad():
    # Predict all four outputs
    predicted_a1, predicted_c1, predicted_a2, predicted_c2 = trained_model(input_param, input_disp, input_shear)

param_scaler, disp_scaler, shear_scaler, outa1_scaler, outc1_scaler, outa2_scaler, outc2_scaler = scaler

# Move tensors to CPU for plotting and denormalization
input_param = denormalize2(input_param.cpu().numpy(), param_scaler,
                                    sequence=False, scaling_strategy='robust')
input_disp = denormalize2(input_disp.cpu().numpy(), disp_scaler,
                                      sequence=True, scaling_strategy='symmetric_log',
                                      handle_small_values=True, small_value_threshold=1e-3)
input_shear = denormalize2(input_shear.cpu().numpy(), shear_scaler,
                               sequence=True, scaling_strategy='symmetric_log',
                               handle_small_values=True, small_value_threshold=1e-3)

# Denormalize predictions and real values
predicted_a1 = denormalize2(predicted_a1.cpu().numpy(), outa1_scaler,
                            sequence=True, scaling_strategy='symmetric_log',
                            handle_small_values=True, small_value_threshold=1e-3)
predicted_c1 = denormalize2(predicted_c1.cpu().numpy(), outc1_scaler,
                            sequence=True, scaling_strategy='symmetric_log',
                            handle_small_values=True, small_value_threshold=1e-3)
predicted_a2 = denormalize2(predicted_a2.cpu().numpy(), outa2_scaler,
                            sequence=True, scaling_strategy='symmetric_log',
                            handle_small_values=True, small_value_threshold=1e-3)
predicted_c2 = denormalize2(predicted_c2.cpu().numpy(), outc2_scaler,
                            sequence=True, scaling_strategy='symmetric_log',
                            handle_small_values=True, small_value_threshold=1e-3)

real_a1 = denormalize2(real_a1.cpu().numpy(), outa1_scaler,
                       sequence=True, scaling_strategy='symmetric_log',
                       handle_small_values=True, small_value_threshold=1e-3)
real_c1 = denormalize2(real_c1.cpu().numpy(), outc1_scaler,
                       sequence=True, scaling_strategy='symmetric_log',
                       handle_small_values=True, small_value_threshold=1e-3)
real_a2 = denormalize2(real_a2.cpu().numpy(), outa2_scaler,
                       sequence=True, scaling_strategy='symmetric_log',
                       handle_small_values=True, small_value_threshold=1e-3)
real_c2 = denormalize2(real_c2.cpu().numpy(), outc2_scaler,
                       sequence=True, scaling_strategy='symmetric_log',
                       handle_small_values=True, small_value_threshold=1e-3)

# Plotting code
for i in range(test_index):
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_a1[i], label=f'Predicted Shear - {i + 1}')
    plt.plot(real_a1[i], label=f'Real Shear - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
