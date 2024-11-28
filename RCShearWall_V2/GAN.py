import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
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

# Generator (Your TimeSeriesTransformer)
class Generator(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.series_encoder = nn.Sequential(
            nn.Linear(displacement_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(d_model, nhead=4, dropout=0.1)
            for _ in range(3)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, 1)  # Output shear value
        )
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.BatchNorm1d(1)
        )

    def forward(self, parameters, time_series):
        batch_size = parameters.shape[0]
        params_expanded = parameters.unsqueeze(1).expand(-1, self.sequence_length, -1)
        params_encoded = self.param_encoder(params_expanded)
        series_encoded = self.series_encoder(time_series.unsqueeze(-1))
        combined = torch.cat([params_encoded, series_encoded], dim=-1)
        combined = self.pos_encoder(combined)
        x = combined
        for block in self.processing_blocks:
            x = block(x)
        output = self.output_layer(x)
        smoothed_output = self.temporal_smoother(output.transpose(1, 2)).transpose(1, 2)
        return smoothed_output.squeeze(-1)


class Discriminator(nn.Module):
    def __init__(self, d_model, sequence_length):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, padding=2),  # Conv1d layer 1
            nn.LeakyReLU(0.2),
            nn.Conv1d(d_model, d_model * 2, kernel_size=5, padding=2),  # Conv1d layer 2
            nn.LeakyReLU(0.2),
            nn.Conv1d(d_model * 2, d_model * 4, kernel_size=5, padding=2),  # Conv1d layer 3
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(d_model * 4 * sequence_length, 1),  # Adjust based on output size of Conv1d layers
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)



# Define hyperparameters
DATA_FOLDER = "RCWall_Data/Run_Full/FullData"
DATA_SIZE = 1000
SEQUENCE_LENGTH = 100
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
TEST_SIZE = 0.10
VAL_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10
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

# Define hyperparameters
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for GAN

# Optimizers
lr = 0.0001
d_model=64
generator = Generator(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
discriminator = Discriminator(d_model, SEQUENCE_LENGTH).to(device)

optimizer_G = optim.AdamW(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# GAN Training Loop
for epoch in range(EPOCHS):
    # ===========================
    # Train Discriminator
    # ===========================
    discriminator.train()
    epoch_D_loss = 0.0
    batch_count = 0
    # Progress bar for training with real-time metrics
    train_loader_tqdm = tqdm(train_loader,
                             desc=f"Epoch {epoch + 1}/{EPOCHS} [GAN Train]",
                             bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | G Loss: {postfix[0][g_loss]:.4f} | D Loss: {postfix[0][d_loss]:.4f}"),
                             postfix=[{"g_loss": 0.0, "d_loss": 0.0}],
                             leave=False)

    for batch_param, batch_disp, batch_shear in train_loader_tqdm:
        batch_count += 1
        # Real data labels (1)
        real_data = batch_shear.unsqueeze(1).to(device)  # Add channel dimension

        # Fake data labels (0)
        fake_data = generator(batch_param.to(device), batch_disp.to(device))
        fake_data = fake_data.unsqueeze(1)  # Add channel dimension

        # Train Discriminator with real data
        optimizer_D.zero_grad()

        # Forward pass for real and fake data
        real_pred = discriminator(real_data)  # Output shape: [batch_size, 1]
        fake_pred = discriminator(fake_data)  # Output shape: [batch_size, 1]

        # Create labels (1 for real, 0 for fake)
        real_labels = torch.ones(real_pred.size(0), 1).to(device)
        fake_labels = torch.zeros(fake_pred.size(0), 1).to(device)

        # Check that the batch sizes match
        assert real_pred.size() == real_labels.size(), f"Shape mismatch: {real_pred.size()} vs {real_labels.size()}"
        assert fake_pred.size() == fake_labels.size(), f"Shape mismatch: {fake_pred.size()} vs {fake_labels.size()}"

        # Compute the discriminator loss
        D_loss_real = criterion(real_pred, real_labels)
        D_loss_fake = criterion(fake_pred, fake_labels)

        # Total Discriminator loss
        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        optimizer_D.step()

        epoch_D_loss += D_loss.item()

        # Update progress bar
        train_loader_tqdm.postfix[0]["g_loss"] = D_loss_real.item()
        train_loader_tqdm.postfix[0]["d_loss"] = D_loss_fake.item()
        train_loader_tqdm.update(1)

    # ===========================
    # Train Generator
    # ===========================
    # Train Generator
    generator.train()
    epoch_G_loss = 0.0
    val_loader_tqdm = tqdm(val_loader,
                           desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                           bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | G Loss: {postfix[0][g_loss]:.4f} | D Loss: {postfix[0][d_loss]:.4f}"),
                           postfix=[{"g_loss": 0.0, "d_loss": 0.0}],
                           leave=False)

    for batch_param, batch_disp, batch_shear in val_loader_tqdm:
        batch_count += 1
        # Train Generator to fool the Discriminator
        optimizer_G.zero_grad()
        fake_data = generator(batch_param.to(device), batch_disp.to(device))
        fake_data = fake_data.unsqueeze(1)  # Add channel dimension
        fake_pred = discriminator(fake_data)
        fake_labels = torch.zeros(fake_pred.size(0), 1).to(device)  # Shape: [BATCH_SIZE, 1]

        G_loss = criterion(fake_pred, torch.ones(fake_pred.size(0), 1).to(device))  # Shape: [BATCH_SIZE, 1]
        G_loss.backward()
        optimizer_G.step()

        epoch_G_loss += G_loss.item()
        # Convert predictions and labels to NumPy arrays for r2_score
        real_pred_np = real_pred.cpu().detach().numpy()  # Convert to numpy for r2_score
        fake_pred_np = fake_pred.cpu().detach().numpy()
        real_labels_np = real_labels.cpu().detach().numpy()
        fake_labels_np = fake_labels.cpu().detach().numpy()

        # Calculate R² score for real and fake data
        r2_real = r2_score(real_labels_np, real_pred_np)
        r2_fake = r2_score(fake_labels_np, fake_pred_np)

        # Print the R² scores
        print(f"R² score for real data: {r2_real:.4f}")
        print(f"R² score for fake data: {r2_fake:.4f}")

        # Continue with your loss calculations
        D_loss_real = criterion(real_pred, real_labels)  # Loss for real data
        D_loss_fake = criterion(fake_pred, fake_labels)  # Loss for fake data

        # Update progress bar
        val_loader_tqdm.postfix[0]["g_loss"] = G_loss.item()
        val_loader_tqdm.postfix[0]["d_loss"] = D_loss_fake.item()
        val_loader_tqdm.update(1)

    # Print training statistics
    print(f"Epoch [{epoch + 1}/{EPOCHS}], D Loss: {epoch_D_loss / batch_count:.4f}, G Loss: {epoch_G_loss / batch_count:.4f}")

    # Save model checkpoints and track best performance as needed
