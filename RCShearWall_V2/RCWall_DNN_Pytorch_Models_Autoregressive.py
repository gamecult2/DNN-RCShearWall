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


class Generator(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200, noise_dim=50):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        self.noise_dim = noise_dim

        # Parameter Processing Branch
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Noise Processing Branch
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, d_model // 2),  # Match the dimension of param_encoder output
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Processing Blocks
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(d_model, nhead=4, dropout=0.1)
            for _ in range(1)
        ])

        # Enhanced Output Generation
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            # nn.GELU(),
            # nn.Dropout(0.1),
            # nn.Linear(d_model * 2, d_model * 4),
            # nn.GELU(),
            # nn.Dropout(0.1),
            # nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, displacement_features)
        )

        # Temporal smoothing convolution
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(displacement_features, displacement_features,
                      kernel_size=5, padding=2, groups=displacement_features),
            nn.GELU(),
            nn.BatchNorm1d(displacement_features)
        )

    def forward(self, parameters, noise):
        batch_size = parameters.shape[0]

        # Expand parameters to match sequence length
        params_expanded = parameters.unsqueeze(1).expand(-1, self.sequence_length, -1)
        params_encoded = self.param_encoder(params_expanded)

        # Process noise
        noise_encoded = self.noise_encoder(noise).unsqueeze(1).expand(-1, self.sequence_length, -1)

        # Combine features
        combined = torch.cat([params_encoded, noise_encoded], dim=-1)

        # Add positional encoding
        combined = self.pos_encoder(combined)

        # Process through main blocks
        x = combined
        for block in self.processing_blocks:
            x = block(x)

        # Generate output sequence
        output = self.output_layer(x)

        # Apply temporal smoothing
        smoothed_output = self.temporal_smoother(output.transpose(1, 2)).transpose(1, 2)

        return smoothed_output.squeeze(-1)


class Discriminator(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200):
        super(Discriminator, self).__init__()

        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(parameters_features + displacement_features, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Processing blocks
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(d_model, nhead=4, dropout=0.1)
            for _ in range(2)
        ])

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, parameters, time_series):
        # Combine parameters and time series
        batch_size = parameters.shape[0]

        # Expand parameters to match sequence length
        params_expanded = parameters.unsqueeze(1).expand(-1, time_series.shape[1], -1)

        # Concatenate parameters and time series
        combined_input = torch.cat([params_expanded, time_series.unsqueeze(-1)], dim=-1)

        # Initial processing
        x = self.input_layer(combined_input)

        # Process through blocks
        for block in self.processing_blocks:
            x = block(x)

        # Global pooling (mean across sequence)
        x = x.mean(dim=1)

        # Classification
        validity = self.classification_head(x)

        return validity


class GAN(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200, noise_dim=50):
        super(GAN, self).__init__()
        self.generator = Generator(parameters_features, displacement_features, sequence_length, d_model, noise_dim)
        self.discriminator = Discriminator(parameters_features, displacement_features, sequence_length, d_model)

    def forward(self, parameters, noise):
        # Generate fake time series
        generated_series = self.generator(parameters, noise)
        return generated_series

# Define hyperparameters
DATA_FOLDER = "RCWall_Data/Run_Full/FullData"
DATA_SIZE = 5000
SEQUENCE_LENGTH = 100
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
TEST_SIZE = 0.10
VAL_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 2
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
NOISE_DIM = 50  # Should match the noise dimension in the Generator
gan_model = GAN(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH, noise_dim=NOISE_DIM).to(device)

# Optimizers for Generator and Discriminator
g_optimizer = optim.AdamW(gan_model.generator.parameters(), lr=LEARNING_RATE)
d_optimizer = optim.AdamW(gan_model.discriminator.parameters(), lr=LEARNING_RATE)

# Learning rate schedulers
g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.1)
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.1)
torchinfo.summary(gan_model)
# Initialize tracking variables
train_g_losses, train_d_losses, val_g_losses, val_d_losses = [], [], [], []
best_val_loss = float("inf")
best_epoch = 0

# ================ Training phase ================
for epoch in range(EPOCHS):
    gan_model.train()
    epoch_g_loss, epoch_d_loss = 0.0, 0.0
    batch_count = 0

    # Progress bar for training with real-time metrics
    train_loader_tqdm = tqdm(train_loader,
                             desc=f"Epoch {epoch + 1}/{EPOCHS} [GAN Train]",
                             bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | G Loss: {postfix[0][g_loss]:.4f} | D Loss: {postfix[0][d_loss]:.4f}"),
                             postfix=[{"g_loss": 0.0, "d_loss": 0.0}],
                             leave=False)

    for batch_param, batch_disp, batch_shear in train_loader_tqdm:
        batch_count += 1
        batch_size = batch_param.size(0)

        # Generate noise
        noise = torch.randn(batch_size, NOISE_DIM).to(device)

        # Train Discriminator
        d_optimizer.zero_grad(set_to_none=True)

        # Generate fake time series
        generated_series = gan_model.generator(batch_param, noise)

        # Discriminator forward pass
        real_validity = gan_model.discriminator(batch_param, batch_disp)
        fake_validity = gan_model.discriminator(batch_param, generated_series.detach())

        # Discriminator loss
        d_real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
        d_fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Discriminator backward pass
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(gan_model.discriminator.parameters(), max_norm=1.0)
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad(set_to_none=True)

        # Regenerate series to update generator
        generated_series = gan_model.generator(batch_param, noise)

        # Generator loss (fooling the discriminator)
        g_validity = gan_model.discriminator(batch_param, generated_series)
        g_loss = F.binary_cross_entropy(g_validity, torch.ones_like(g_validity))

        # Additional loss terms (optional)
        reconstruction_loss = F.mse_loss(generated_series, batch_shear)
        g_total_loss = g_loss + 0.1 * reconstruction_loss  # Balance adversarial and reconstruction loss

        # Generator backward pass
        g_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(gan_model.generator.parameters(), max_norm=1.0)
        g_optimizer.step()

        # Update epoch metrics
        epoch_g_loss += g_total_loss.item()
        epoch_d_loss += d_loss.item()

        # Update progress bar
        train_loader_tqdm.postfix[0]["g_loss"] = g_total_loss.item()
        train_loader_tqdm.postfix[0]["d_loss"] = d_loss.item()
        train_loader_tqdm.update(1)
        # Calculate R² score for real and fake data
        r2_real = r2_score(generated_series.detach().cpu().numpy(), batch_shear.detach().cpu().numpy())

        # Print the R² scores
        print(f"R² score for real data: {r2_real:.4f}")


    # Calculate average losses
    epoch_g_loss /= len(train_loader)
    epoch_d_loss /= len(train_loader)
    train_g_losses.append(epoch_g_loss)
    train_d_losses.append(epoch_d_loss)

    # ================ Validation phase ================
    gan_model.eval()
    val_g_loss, val_d_loss = 0.0, 0.0

    val_loader_tqdm = tqdm(val_loader,
                           desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
                           bar_format=("{l_bar}{bar} | Processed: {n_fmt}/{total_fmt} | Remaining: {remaining} | G Loss: {postfix[0][g_loss]:.4f} | D Loss: {postfix[0][d_loss]:.4f}"),
                           postfix=[{"g_loss": 0.0, "d_loss": 0.0}],
                           leave=False)

    with torch.no_grad():
        for batch_param, batch_disp, batch_shear in val_loader_tqdm:
            batch_size = batch_param.size(0)
            noise = torch.randn(batch_size, NOISE_DIM).to(device)

            # Generate series
            generated_series = gan_model.generator(batch_param, noise)

            # Discriminator validation
            real_validity = gan_model.discriminator(batch_param, batch_disp)
            fake_validity = gan_model.discriminator(batch_param, generated_series)

            # Validation losses
            d_real_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity))
            d_fake_loss = F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))
            val_d_loss += (d_real_loss + d_fake_loss).item() / 2

            # Generator validation loss
            g_validity = gan_model.discriminator(batch_param, generated_series)
            g_loss = F.binary_cross_entropy(g_validity, torch.ones_like(g_validity))
            reconstruction_loss = F.mse_loss(generated_series, batch_shear)
            val_g_loss += (g_loss + 0.1 * reconstruction_loss).item()
            r2_fake = r2_score(generated_series.detach().cpu().numpy(), batch_shear.detach().cpu().numpy())
            print(f"R² score for fake data: {r2_fake:.4f}")

            # Update progress bar
            val_loader_tqdm.postfix[0]["g_loss"] = g_loss.item()
            val_loader_tqdm.postfix[0]["d_loss"] = d_real_loss.item()
            val_loader_tqdm.update(1)

    # Calculate average validation losses
    val_g_loss /= len(val_loader)
    val_d_loss /= len(val_loader)
    val_g_losses.append(val_g_loss)
    val_d_losses.append(val_d_loss)

    # Update learning rates
    g_scheduler.step()
    d_scheduler.step()

    # Print epoch summary
    print(f'Epoch [{epoch + 1}/{EPOCHS}], G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}, '
          f'Val G Loss: {val_g_loss:.4f}, Val D Loss: {val_d_loss:.4f}\n')

    # Early Stopping (can be adapted for GAN)
    current_val_loss = val_g_loss + val_d_loss
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_epoch = epoch
        torch.save(gan_model.state_dict(), f"checkpoints/best_gan_model.pt")

# Optional: Load best model
gan_model.load_state_dict(torch.load(f"checkpoints/best_gan_model.pt"))


#  Plotting results
test_index = 2
new_input_parameters = X_param_test[:test_index]
new_input_displacement = X_disp_test[:test_index]
real_shear = Y_shear_test[:test_index]

with torch.no_grad():
    predicted_shear = gan_model.generator(new_input_parameters, new_input_displacement)

# Move tensors to CPU for plotting
new_input_parameters = denormalize(new_input_parameters.cpu().numpy(), param_scaler, sequence=False)
new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear.cpu().numpy(), shear_scaler, sequence=True)

# Move tensors to CPU for plotting and denormalization
# new_input_parameters = denormalize2(new_input_parameters.cpu().numpy(), param_scaler, scaling_strategy='robust')
# new_input_displacement = denormalize2(new_input_displacement.cpu().numpy(), disp_scaler, scaling_strategy='log_minmax', handle_small_values=True, small_value_threshold=1e-3)
# real_shear = denormalize2(real_shear.cpu().numpy(), shear_scaler, scaling_strategy='log_minmax', handle_small_values=True, small_value_threshold=1e-3)
# predicted_shear = denormalize2(predicted_shear.cpu().numpy(), shear_scaler, scaling_strategy='log_minmax', handle_small_values=True, small_value_threshold=1e-3)

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
