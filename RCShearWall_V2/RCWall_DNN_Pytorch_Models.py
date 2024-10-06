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

from xLSTM.model import xLSTM
from xLSTM.block import xLSTMBlock
from xLSTM.mlstm import mLSTM
from xLSTM.slstm import sLSTM

from informer.model import Informer

from LLaMA2.model import *

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from RCWall_Data_Processing import *
from utils.earlystopping import EarlyStopping

# Determine the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")
torch.backends.cuda.enable_flash_sdp(True)


def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Remove transpose
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Transformer_Model(nn.Module):
    def __init__(self, num_features_input_parameters, num_features_input_displacement, sequence_length,
                 d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.0):
        super(Transformer_Model, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Input embedding layers
        self.input_embedding = nn.Linear(num_features_input_displacement + num_features_input_parameters, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output layers
        self.dense1 = nn.Linear(d_model, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(100, 1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, parameters_input, displacement_input):
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # print('distributed_parameters', distributed_parameters.shape)

        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)
        # print('concatenated_tensor', concatenated_tensor.shape)

        # Embed and add positional encoding
        x = self.input_embedding(concatenated_tensor)
        # print('input_embedding', x.shape)
        x = self.positional_encoding(x)
        # print('positional_encoding', x.shape)

        # Create masks
        src_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        # print('src_mask', src_mask.shape)
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        # print('tgt_mask', tgt_mask.shape)
        # Transformer forward pass
        output = self.transformer(x, x, src_mask=src_mask, tgt_mask=tgt_mask)
        #print('output transformer', output.shape)

        # Dense layers
        x = self.dense1(output)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)

        # Output layer
        output_shear = self.output(x)
        output_shear = output_shear.squeeze(-1)  # (batch, seq_len)

        return output_shear


class Informer_Model(nn.Module):
    def __init__(self, num_features_input_parameters, num_features_input_displacement, sequence_length,
                 d_model=256, n_heads=4, e_layers=2, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, device=torch.device('cuda:0')):
        super(Informer_Model, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.device = device

        enc_in = num_features_input_parameters + num_features_input_displacement
        dec_in = num_features_input_parameters + num_features_input_displacement
        c_out = 1
        out_len = sequence_length

        # Informer model
        self.informer = Informer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=sequence_length,
            label_len=sequence_length,
            out_len=out_len,
            factor=5,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn=attn,
            embed=embed,
            freq=freq,
            activation=activation,
            output_attention=output_attention,
            distil=distil,
            mix=mix,
            device=device
        )

        # Output layers
        self.dense1 = nn.Linear(d_model, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(100, 1)

    def forward(self, parameters_input, displacement_input):
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # print('distributed_parameters', distributed_parameters.shape)
        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)
        # print('concatenated_tensor', concatenated_tensor.shape)
        # Create dummy temporal embeddings if needed
        x_mark_enc = torch.zeros(concatenated_tensor.size(0), self.sequence_length, concatenated_tensor.size(2)).to(self.device)
        x_mark_dec = torch.zeros(concatenated_tensor.size(0), self.sequence_length, concatenated_tensor.size(2)).to(self.device)
        # print('x_mark_enc', x_mark_enc.shape)
        # print('x_mark_dec', x_mark_dec.shape)

        # Informer forward pass
        dec_inp = torch.zeros(concatenated_tensor.size(0), self.sequence_length, concatenated_tensor.size(2)).to(self.device)  # Dummy decoder input for the initial step
        # print('dec_inp', dec_inp.shape)
        x = self.informer(concatenated_tensor, x_mark_enc, dec_inp, x_mark_dec)

        # print('output', x.shape)

        # Dense layers
        # x = self.dense1(output)
        # x = torch.tanh(x)
        # x = self.dropout1(x)
        # x = self.dense2(x)
        # x = torch.tanh(x)
        # x = self.dropout2(x)

        # Output layer
        # output_shear = self.output(x)
        # print('output_shear', output_shear.shape)
        output_shear = x.squeeze(-1)  # (batch, seq_len)
        # print('output_shear', output_shear.shape)

        return output_shear


class xLSTM_Model(nn.Module):
    def __init__(self, num_features_input_parameters, num_features_input_displacement, sequence_length):
        super(xLSTM_Model, self).__init__()
        self.sequence_length = sequence_length
        self.norm = nn.LayerNorm(200)
        self.norm2 = nn.LayerNorm(1)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(0.1)

        '''
        # #  xLSTM encoder
        self.lstm_encoder1 = xLSTMBlock(num_features_input_displacement + num_features_input_parameters, 200, num_layers=1, lstm_type="slstm")
        self.lstm_encoder2 = xLSTMBlock(200, 50, num_layers=1, lstm_type="slstm")
        # # xLSTM decoder
        self.lstm_decoder1 = xLSTMBlock(50, 200, num_layers=1, lstm_type="slstm")
        self.lstm_decoder2 = xLSTMBlock(200, num_features_input_displacement, num_layers=1, lstm_type="slstm")
        '''
        #  xLSTM encoder
        self.lstm_encoder1 = sLSTM(num_features_input_displacement + num_features_input_parameters, 200, num_layers=2)
        self.lstm_encoder2 = sLSTM(200, 50, num_layers=1)

        # xLSTM decoder
        self.lstm_decoder1 = sLSTM(50, 200, num_layers=1)
        self.lstm_decoder2 = sLSTM(200, num_features_input_displacement, num_layers=2)

        # Adjusting dimensions
        self.dense1 = nn.Linear(num_features_input_displacement, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(100, 1)

    def forward(self, parameters_input, displacement_input):
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, displacement_input.shape[1], 1)
        # print('distributed_parameters', distributed_parameters.shape)

        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)
        # print('concatenated_tensor', concatenated_tensor.shape)

        # Encoding
        lstm_out, _ = self.lstm_encoder1(concatenated_tensor)
        lstm_out = self.activation(lstm_out)
        # print('lstm_out activation', lstm_out.shape)
        # lstm_out = self.norm(lstm_out)
        # print('lstm_out norm', lstm_out.shape)
        # encoded_sequence, _ = self.lstm_encoder2(lstm_out)
        # encoded_sequence = self.activation(encoded_sequence)
        #
        # # Decoding
        # lstm_out, _ = self.lstm_decoder1(encoded_sequence)
        # lstm_out = self.activation(lstm_out)

        decoded_sequence, _ = self.lstm_decoder2(lstm_out)
        decoded_sequence = self.activation(decoded_sequence)
        # Dense layers
        x = self.dense1(decoded_sequence)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)

        # Output layer
        output_shear = self.output(x)
        # print('output_shear', output_shear.shape)
        output_shear = output_shear.reshape(output_shear.size(0), -1)

        return output_shear


class LSTM_AE_Model(nn.Module):
    def __init__(self, num_features_input_parameters, num_features_input_displacement, sequence_length):
        super(LSTM_AE_Model, self).__init__()
        self.sequence_length = sequence_length

        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)
        self.lstm_encoder1 = nn.LSTM(num_features_input_displacement + num_features_input_parameters, 200, batch_first=True)
        self.lstm_encoder2 = nn.LSTM(200, 50, batch_first=True)

        self.lstm_decoder1 = nn.LSTM(50, 200, batch_first=True)
        self.lstm_decoder2 = nn.LSTM(200, num_features_input_displacement, batch_first=True)

        self.dense1 = nn.Linear(num_features_input_displacement, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(200, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(100, 1)

    def forward(self, parameters_input, displacement_input):
        # print('parameters_input', parameters_input.shape)
        # print('displacement_input', displacement_input.shape)
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # print('distributed_parameters', distributed_parameters.shape)
        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)
        # print('concatenated_tensor', concatenated_tensor.shape)

        # Encoding
        lstm_out, _ = self.lstm_encoder1(concatenated_tensor)
        # print('lstm_out', lstm_out.shape)
        encoded_sequence, _ = self.lstm_encoder2(lstm_out)
        # print('encoded_sequence', encoded_sequence.shape)

        # Decoding
        lstm_out, _ = self.lstm_decoder1(encoded_sequence)
        # print('lstm_out', lstm_out.shape)
        decoded_sequence, _ = self.lstm_decoder2(lstm_out)
        # print('decoded_sequence', decoded_sequence.shape)

        # Dense layers
        x = self.dense1(decoded_sequence)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)

        # Output layer
        output_shear = self.output(x)
        # print('output_shear', output_shear.shape)
        output_shear = output_shear.reshape(output_shear.size(0), -1)

        return output_shear


class LLaMA2_Model(nn.Module):
    def __init__(self, num_features_input_parameters, num_features_input_displacement, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length

        # Fixed values
        dim = 512
        n_layers = 2
        n_heads = 4

        config = ModelConfig(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            max_batch_size=32,
            device=device,
            ffn_dim_multiplier=None,
            multiple_of=256
        )

        self.input_embedding = nn.Linear(num_features_input_displacement + num_features_input_parameters, dim)
        self.dropout = nn.Dropout(0.1)  # Fixed dropout rate

        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])

        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, 1)

    def forward(self, parameters_input, displacement_input):
        print(f"parameters_input shape: {parameters_input.shape}")
        print(f"displacement_input shape: {displacement_input.shape}")

        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        print(f"distributed_parameters shape: {distributed_parameters.shape}")

        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)
        print(f"concatenated_tensor shape: {concatenated_tensor.shape}")

        x = self.input_embedding(concatenated_tensor)
        print(f"After input_embedding shape: {x.shape}")

        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            print(f"Before layer {i} shape: {x.shape}")
            x = layer(x, 0)
            print(f"After layer {i} shape: {x.shape}")

        x = self.norm(x)
        output = self.output(x)

        return output.squeeze(-1)


# Define hyperparameters
DATA_SIZE = 30000
SEQUENCE_LENGTH = 500
NUM_FEATURES_INPUT_DISPLACEMENT = 1
NUM_FEATURES_INPUT_PARAMETERS = 13
PUSHOVER = False
TEST_SIZE = 0.2
VAL_SIZE = 0.2

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 5

# Load and preprocess data
data, scalers = load_data(DATA_SIZE, SEQUENCE_LENGTH, normalize_data=True, save_normalized_data=False, pushover=PUSHOVER)
InParams, InDisp, OutShear = data
param_scaler, disp_scaler, shear_scaler = scalers

# Split and convert data
X_param_train, X_disp_train, Y_shear_train, X_param_val, X_disp_val, Y_shear_val, X_param_test, X_disp_test, Y_shear_test = split_and_convert(
    (InParams, InDisp, OutShear), test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=42, device=device)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_param_train, X_disp_train, Y_shear_train), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_param_val, X_disp_val, Y_shear_val), BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_param_test, X_disp_test, Y_shear_test), BATCH_SIZE, shuffle=False)


# Initialize model, loss, and optimizer
# model = LSTM_AE_Model(NUM_FEATURES_INPUT_PARAMETERS, NUM_FEATURES_INPUT_DISPLACEMENT, SEQUENCE_LENGTH).to(device)
model = Transformer_Model(NUM_FEATURES_INPUT_PARAMETERS, NUM_FEATURES_INPUT_DISPLACEMENT, SEQUENCE_LENGTH).to(device)
# model = Informer_Model(NUM_FEATURES_INPUT_PARAMETERS, NUM_FEATURES_INPUT_DISPLACEMENT, SEQUENCE_LENGTH).to(device)
# model = xLSTM_Model(NUM_FEATURES_INPUT_PARAMETERS, NUM_FEATURES_INPUT_DISPLACEMENT, SEQUENCE_LENGTH).to(device)
# model = LLaMA2_Model(NUM_FEATURES_INPUT_PARAMETERS, NUM_FEATURES_INPUT_DISPLACEMENT, SEQUENCE_LENGTH).to(device)

# model = torch.compile(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
early_stopping = EarlyStopping(PATIENCE, verbose=False, path=f"checkpoints/{type(model).__name__}.pt")
torchinfo.summary(model)

# Training loop
train_losses, val_losses, train_r2_scores, val_r2_scores = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = epoch_train_r2 = 0
    for batch_param, batch_disp, batch_shear in train_loader:  # tqdm(train_loader, desc="Batch", leave=False):
        optimizer.zero_grad()
        outputs = model(batch_param, batch_disp)
        loss = criterion(outputs, batch_shear)
        r2 = r2_score(batch_shear, outputs)  # Calculate R2 score
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_r2 += r2.item()

    # Calculate average training loss and R2 for the epoch
    epoch_train_loss /= len(train_loader)
    epoch_train_r2 /= len(train_loader)
    train_losses.append(epoch_train_loss)
    train_r2_scores.append(epoch_train_r2)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = val_r2 = 0
        for batch_param, batch_disp, batch_shear in val_loader:
            val_outputs = model(batch_param, batch_disp)
            val_loss += criterion(val_outputs, batch_shear).item()
            val_r2 += r2_score(val_outputs, batch_shear).item()

        val_loss /= len(val_loader)
        val_r2 /= len(val_loader)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Train R2: {epoch_train_r2:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}')

    # Early Stopping
    if early_stopping(val_loss, model):
        print("Early stopping")
        break

# Restore best weights
model.load_state_dict(torch.load(f"checkpoints/{type(model).__name__}.pt"))

# Final evaluation
model.eval()
with torch.no_grad():
    test_loss = test_r2 = 0
    for batch_param, batch_disp, batch_shear in test_loader:
        test_outputs = model(batch_param, batch_disp)
        test_loss += criterion(test_outputs, batch_shear).item()
        test_r2 += r2_score(test_outputs, batch_shear).item()

    test_loss /= len(test_loader)
    test_r2 /= len(test_loader)

print(f'Final Model Performance - Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}')

# Plotting
best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed


def plot_metric(train_data, val_data, best_epoch, ylabel, title):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_data) + 1)
    plt.plot(epochs, train_data, label=f"Training {ylabel}")
    plt.plot(epochs, val_data, label=f"Validation {ylabel}")
    plt.scatter(best_epoch, val_data[best_epoch - 1], color='red', s=100, label="Best Model")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{title} Over Epochs")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Plot loss
plot_metric(train_losses, val_losses, best_epoch, "MSE Loss", "Training and Validation Loss")

# Plot R2 score
plot_metric(train_r2_scores, val_r2_scores, best_epoch, "R2 Score", "Training and Validation R2 Score")

# Plotting results
test_index = 5
new_input_parameters = X_param_test[:test_index]
new_input_displacement = X_disp_test[:test_index]
real_shear = Y_shear_test[:test_index]

with torch.no_grad():
    predicted_shear = model(new_input_parameters, new_input_displacement)

# Move tensors to CPU for plotting
new_input_parameters = denormalize(new_input_parameters.cpu().numpy(), param_scaler, sequence=False)
new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear.cpu().numpy(), shear_scaler, sequence=True)

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
