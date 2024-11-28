import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torchviz import make_dot
from torch.utils.data import DataLoader, TensorDataset
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
from DNNModels import *



# Determine the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__} --- Using device: {device}")
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
PARAMETERS_FEATURES = 15
TEST_SIZE = 0.10
VAL_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 1
PATIENCE = 5

# Load and preprocess data
(InParams, InDisp, OutShear), (param_scaler, disp_scaler, shear_scaler) = load_data(DATA_SIZE,
                                                                                    SEQUENCE_LENGTH,
                                                                                    PARAMETERS_FEATURES,
                                                                                    DATA_FOLDER,
                                                                                    True,
                                                                                    True)

# Split and convert data
splits = split_and_convert((InParams, InDisp, OutShear), TEST_SIZE, VAL_SIZE, 40, device, True)

(X_param_train, X_disp_train, Y_shear_train,
 X_param_val, X_disp_val, Y_shear_val,
 X_param_test, X_disp_test, Y_shear_test) = splits

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_param_train, X_disp_train, Y_shear_train), BATCH_SIZE, shuffle=True) #, pin_memory=True, num_workers=4, prefetch_factor=2)
val_loader = DataLoader(TensorDataset(X_param_val, X_disp_val, Y_shear_val), BATCH_SIZE, shuffle=False) #, pin_memory=True, num_workers=4, prefetch_factor=2)
test_loader = DataLoader(TensorDataset(X_param_test, X_disp_test, Y_shear_test), BATCH_SIZE, shuffle=False) #, pin_memory=True, num_workers=4, prefetch_factor=2)

# Initialize model, loss, and optimizer
# model = LSTM_AE_Model_1(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_2(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LSTM_AE_Model_3(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
model = Transformer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = Informer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTM_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMA2_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = AttentionLSTM_AEModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMAInspiredModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTMModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = TimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = EnhancedTimeSeriesTransformer(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = InformerShearModel(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = torch.compile(model)


optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  #  weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion = nn.SmoothL1Loss().to(device)  # nn.MSELoss().to(device)
early_stopping = EarlyStopping(PATIENCE, verbose=False, save_full_model=True, checkpoint_dir='checkpoints', model_name=f"{type(model).__name__}")

# Visualize the computation graph
torchinfo.summary(model)

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

    for batch_param, batch_disp, batch_shear in train_loader_tqdm:
        batch_count += 1  # Increment batch counter
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad() Zero gradients
        outputs = model(batch_param, batch_disp)  # Forward pass
        loss = criterion(outputs, batch_shear)  # Loss computation
        r2 = r2_score(batch_shear.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        # Backward pass and optimizer step
        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()  # Update parameters

        # Update epoch metrics
        epoch_train_loss += loss.item()
        epoch_train_r2 += r2 # .item()

        # Update progress bar with real-time batch metrics
        train_loader_tqdm.postfix[0]["batch_loss"] = loss.item()
        train_loader_tqdm.postfix[0]["batch_r2"] = r2 # .item()
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
        for batch_param, batch_disp, batch_shear in val_loader_tqdm:
            batch_count += 1
            val_outputs = model(batch_param, batch_disp)
            batch_loss = criterion(val_outputs, batch_shear)
            batch_r2 = r2_score(batch_shear.detach().cpu().numpy(), val_outputs.detach().cpu().numpy())
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
    scheduler.step()

    # Print epoch summary
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Learining Rate: {lr}, Train Loss: {epoch_train_loss:.4f}, Train R²: {epoch_train_r2:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}\n')

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
    for batch_param, batch_disp, batch_shear in test_loader:
        test_outputs = model(batch_param, batch_disp)
        test_loss += criterion(batch_shear, test_outputs).item()
        test_r2 = r2_score(batch_shear.detach().cpu().numpy(), test_outputs.detach().cpu().numpy())

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
new_input_parameters = X_param_test[:test_index]
new_input_displacement = X_disp_test[:test_index]
real_shear = Y_shear_test[:test_index]

# Restore best weights
trained_model = torch.load(f"checkpoints/{type(model).__name__}_best_full.pt", weights_only=False)
trained_model.eval()

with torch.no_grad():
    predicted_shear = trained_model(new_input_parameters, new_input_displacement)

# Move tensors to CPU for plotting
# new_input_parameters = denormalize(new_input_parameters.cpu().numpy(), param_scaler, sequence=False)
# new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
# real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
# predicted_shear = denormalize(predicted_shear.cpu().numpy(), shear_scaler, sequence=True)

# Move tensors to CPU for plotting and denormalization
new_input_parameters = denormalize2(new_input_parameters.cpu().numpy(), param_scaler, scaling_strategy='robust', sequence=False)
new_input_displacement = denormalize2(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True, scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-3)
real_shear = denormalize2(real_shear.cpu().numpy(), shear_scaler, sequence=True, scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-3)
predicted_shear = denormalize2(predicted_shear.cpu().numpy(), shear_scaler, sequence=True, scaling_strategy='symmetric_log', handle_small_values=True, small_value_threshold=1e-3)

# Plotting code
for i in range(test_index):
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_shear[i], label=f'Predicted Shear - {i + 1}')
    plt.plot(real_shear[i], label=f'Real Shear - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
