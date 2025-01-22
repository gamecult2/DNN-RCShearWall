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
DATA_SIZE = 50000000
SEQUENCE_LENGTH = 500
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
TEST_SIZE = 0.10
VAL_SIZE = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20
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
# model = LSTM_AE_Model_1(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3_Optimized(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = Transformer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTM_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMA2_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = AttentionLSTM_AEModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMAInspiredModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTMModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3_Optimized(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3_slice(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = EnhancedTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerShearModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
model = DecoderOnlyTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = torch.compile(model)

torchinfo.summary(model)

# Initialize training component
scaler = GradScaler('cuda')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6)
criterion = nn.SmoothL1Loss().to(device)  # nn.MSELoss().to(device)
# earlystop = EarlyStopping(PATIENCE, checkpoint_dir='checkpoints', model_name=f"{type(model).__name__}", save_full_model=True, verbose=False)

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
                             postfix=[{"lr": 0.0, "batch_loss": 0.0, "batch_r2": 0.0, "avg_r2": 0.0}], leave=False)

    for batch_param, batch_disp, batch_shear in train_loader_tqdm:
        batch_param = batch_param.to(device, non_blocking=True)
        batch_disp = batch_disp.to(device, non_blocking=True)
        batch_shear = batch_shear.to(device, non_blocking=True)
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
    # if earlystop(val_loss, model):
    #     print("Early stopping triggered")
    #     break

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
