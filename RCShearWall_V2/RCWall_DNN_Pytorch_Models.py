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

from xLSTM.model import xLSTM
from xLSTM.block import xLSTMBlock
from xLSTM.mlstm import mLSTM
from xLSTM.slstm import sLSTM

from informer.model import Informer

from LLaMA2.model import *

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
    def __init__(self, d_model, max_seq_length=500):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer_Model(nn.Module):
    def __init__(self, num_features_params, num_features_disp, sequence_length,
                 d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.sequence_length = sequence_length
        self.d_model = d_model  # Changed to 256 to be divisible by nhead=8

        # Combined input features (displacement + parameters)
        total_features = num_features_params + 1  # +1 for displacement

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(total_features, d_model),
            nn.LayerNorm(d_model),  # Added LayerNorm for better stability
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, sequence_length)

        # Transformer encoder-decoder architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),  # Added LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, parameters_input, displacement_input):
        # Distribute parameters across sequence length
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Create embeddings from concatenated input
        embedded = self.input_embedding(concatenated_tensor)  # [batch_size, seq_length, d_model]

        # Add positional encoding
        src = self.pos_encoder(embedded)
        tgt = src.clone()

        # Create attention mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(self.sequence_length).to(src.device)

        # Encoder-Decoder processing
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

        # Process output sequence
        predictions = self.output_layer(output)  # [batch_size, seq_length, 1]
        predictions = predictions.squeeze(-1)  # [batch_size, seq_length]

        return predictions
'''
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
    def __init__(self, parameters_features, displacement_features, sequence_length,
                 d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.0):
        super(Transformer_Model, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Input embedding layers
        self.input_embedding = nn.Linear(displacement_features + parameters_features, d_model)
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
'''

class Informer_Model(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length,
                 d_model=256, n_heads=4, e_layers=2, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, device=torch.device('cpu')):
        super(Informer_Model, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.device = device

        enc_in = parameters_features + displacement_features
        dec_in = parameters_features + displacement_features
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

# class xLSTM_Model(nn.Module):
#     def __init__(self, parameters_features, displacement_features, sequence_length):
#         super(xLSTM_Model, self).__init__()
#         self.sequence_length = sequence_length
#         self.norm = nn.LayerNorm(200)
#         self.norm2 = nn.LayerNorm(1)
#         self.activation = nn.GELU()
#         self.dropout_layer = nn.Dropout(0.1)
#
#         '''
#         # #  xLSTM encoder
#         self.lstm_encoder1 = xLSTMBlock(displacement_features + parameters_features, 200, num_layers=1, lstm_type="slstm")
#         self.lstm_encoder2 = xLSTMBlock(200, 50, num_layers=1, lstm_type="slstm")
#         # # xLSTM decoder
#         self.lstm_decoder1 = xLSTMBlock(50, 200, num_layers=1, lstm_type="slstm")
#         self.lstm_decoder2 = xLSTMBlock(200, displacement_features, num_layers=1, lstm_type="slstm")
#         '''
#         #  xLSTM encoder
#         self.lstm_encoder1 = sLSTM(displacement_features + parameters_features, 200, num_layers=2)
#         self.lstm_encoder2 = sLSTM(200, 50, num_layers=1)
#
#         # xLSTM decoder
#         self.lstm_decoder1 = sLSTM(50, 200, num_layers=1)
#         self.lstm_decoder2 = sLSTM(200, displacement_features, num_layers=2)
#
#         # Adjusting dimensions
#         self.dense1 = nn.Linear(displacement_features, 200)
#         self.dropout1 = nn.Dropout(0.2)
#         self.dense2 = nn.Linear(200, 100)
#         self.dropout2 = nn.Dropout(0.2)
#         self.output = nn.Linear(100, 1)
#
#     def forward(self, parameters_input, displacement_input):
#         # Repeat parameters for each time step
#         distributed_parameters = parameters_input.unsqueeze(1).repeat(1, displacement_input.shape[1], 1)
#         # print('distributed_parameters', distributed_parameters.shape)
#
#         # Concatenate displacement and parameters
#         concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)
#         # print('concatenated_tensor', concatenated_tensor.shape)
#
#         # Encoding
#         lstm_out, _ = self.lstm_encoder1(concatenated_tensor)
#         lstm_out = self.activation(lstm_out)
#         # print('lstm_out activation', lstm_out.shape)
#         # lstm_out = self.norm(lstm_out)
#         # print('lstm_out norm', lstm_out.shape)
#         # encoded_sequence, _ = self.lstm_encoder2(lstm_out)
#         # encoded_sequence = self.activation(encoded_sequence)
#         #
#         # # Decoding
#         # lstm_out, _ = self.lstm_decoder1(encoded_sequence)
#         # lstm_out = self.activation(lstm_out)
#
#         decoded_sequence, _ = self.lstm_decoder2(lstm_out)
#         decoded_sequence = self.activation(decoded_sequence)
#         # Dense layers
#         x = self.dense1(decoded_sequence)
#         x = torch.tanh(x)
#         x = self.dropout1(x)
#         x = self.dense2(x)
#         x = torch.tanh(x)
#         x = self.dropout2(x)
#
#         # Output layer
#         output_shear = self.output(x)
#         # print('output_shear', output_shear.shape)
#         output_shear = output_shear.reshape(output_shear.size(0), -1)
#
#         return output_shear

class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out

class ExtendedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Traditional LSTM gates
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        # Additional components
        self.input_norm = nn.LayerNorm(input_size)
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.state_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism for cell state
        self.cell_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden):
        h, c = hidden

        # Apply normalization
        x = self.input_norm(x)
        h = self.hidden_norm(h)

        # Concatenate input and hidden state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)

        # Apply non-linearities
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)  # cell gate
        o = torch.sigmoid(o)  # output gate

        # Compute cell attention
        cell_importance = self.cell_attention(c)

        # Update cell state with attention
        c = f * c + i * g
        c = c * cell_importance
        c = self.state_norm(c)

        # Compute hidden state
        h = o * torch.tanh(c)

        return h, c

class xLSTM_Model2(nn.Module):
    def __init__(self, num_features_params, sequence_length,
                 hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input processing
        total_features = num_features_params + 1  # +1 for displacement
        self.input_embed = nn.Sequential(
            nn.Linear(total_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multiple xLSTM layers
        self.lstm_layers = nn.ModuleList([
            ExtendedLSTMCell(hidden_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])

        # Residual connections
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size * 2, dropout)
            for _ in range(num_layers)
        ])

        # Output processing
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        # Skip connection from input to output
        self.skip_connection = nn.Sequential(
            nn.Linear(total_features, hidden_size // 2),
            nn.ReLU()
        )

    def init_hidden(self, batch_size, device):
        hidden = []
        for _ in range(self.num_layers):
            hidden.append((
                torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device)
            ))
        return hidden

    def forward(self, parameters_input, displacement_input):
        batch_size = parameters_input.size(0)
        device = parameters_input.device

        # Initialize hidden states
        hidden_states = self.init_hidden(batch_size, device)
        outputs = []

        # Distribute parameters across sequence length
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Process each time step
        for t in range(self.sequence_length):
            x = concatenated_input[:, t, :]

            # Skip connection input
            skip_out = self.skip_connection(x)

            # Main input embedding
            x = self.input_embed(x)

            # Process through LSTM layers with residual connections
            for layer_idx in range(self.num_layers):
                residual = x

                # LSTM processing
                h, c = self.lstm_layers[layer_idx](x, hidden_states[layer_idx])
                hidden_states[layer_idx] = (h, c)

                # Residual connection
                x = h
                x = self.residual_blocks[layer_idx](x)
                x = x + residual

            # Combine with skip connection and generate output
            x = torch.cat([x, skip_out], dim=-1)
            out = self.output_layer(x)
            outputs.append(out)

        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        return outputs.squeeze(-1)

class LSTM_AE_Model(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(LSTM_AE_Model, self).__init__()
        self.sequence_length = sequence_length

        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout)
        self.lstm_encoder1 = nn.LSTM(displacement_features + parameters_features, 300, batch_first=True)
        self.lstm_encoder2 = nn.LSTM(300, 50, batch_first=True)

        self.lstm_decoder1 = nn.LSTM(50, 300, batch_first=True)
        self.lstm_decoder2 = nn.LSTM(300, displacement_features, batch_first=True)

        self.dense1 = nn.Linear(displacement_features, 300)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(300, 200)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(200, 1)

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

class LSTM_AE_Model2(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(LSTM_AE_Model2, self).__init__()
        self.sequence_length = sequence_length

        # Encoder
        self.lstm_encoder1 = nn.LSTM(displacement_features + parameters_features, 300, batch_first=True)
        self.lstm_encoder2 = nn.LSTM(300, 50, batch_first=True)

        # Decoder
        self.lstm_decoder1 = nn.LSTM(50, 300, batch_first=True)
        self.lstm_decoder2 = nn.LSTM(300, displacement_features, batch_first=True)

        # Dense layers with modified architecture
        self.dense1 = nn.Linear(displacement_features, 300)
        self.batch_norm1 = nn.BatchNorm1d(300)
        self.dropout1 = nn.Dropout(0.1)

        self.dense2 = nn.Linear(300, 200)
        self.batch_norm2 = nn.BatchNorm1d(200)
        self.dropout2 = nn.Dropout(0.1)

        # Modified output layer for better small value handling
        self.pre_output = nn.Linear(200, 50)
        self.output = nn.Linear(50, 1)

        # Activation for small values
        self.small_activation = nn.ReLU()  # or custom activation

    def small_value_activation(self, x):
        # Custom activation function for small values
        return x * torch.sigmoid(x)  # Smooth transition near zero

    def forward(self, parameters_input, displacement_input):
        # Distribute parameters
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate inputs
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Encoding
        lstm_out, _ = self.lstm_encoder1(concatenated_tensor)
        encoded_sequence, _ = self.lstm_encoder2(lstm_out)

        # Decoding
        lstm_out, _ = self.lstm_decoder1(encoded_sequence)
        decoded_sequence, _ = self.lstm_decoder2(lstm_out)

        # Dense layers with batch normalization
        batch_size = decoded_sequence.size(0)
        time_steps = decoded_sequence.size(1)

        # Reshape for batch norm
        x = decoded_sequence.reshape(-1, decoded_sequence.size(-1))

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = torch.sigmoid(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = torch.sigmoid(x)
        x = self.dropout2(x)

        # Pre-output processing
        x = self.pre_output(x)
        x = self.small_value_activation(x)  # Custom activation for small values
        # x = self.small_activation(x)

        # Final output
        output_shear = self.output(x)
        output_shear = output_shear.reshape(batch_size, -1)

        return output_shear

class LLaMA2_Model(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
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

        self.input_embedding = nn.Linear(displacement_features + parameters_features, dim)
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
DATA_SIZE = 1100
SEQUENCE_LENGTH = 200
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
ANALYSIS = 'CYCLIC'
TEST_SIZE = 0.20
VAL_SIZE = 0.20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 200
PATIENCE = 50

# Load and preprocess data
(InParams, InDisp, OutShear), (param_scaler, disp_scaler, shear_scaler) = load_data(DATA_SIZE,
                                                                                    SEQUENCE_LENGTH,
                                                                                    PARAMETERS_FEATURES, 
                                                                                    True,
                                                                                    ANALYSIS)

# Split and convert data
(X_param_train, X_disp_train, Y_shear_train, X_param_val, X_disp_val, Y_shear_val, X_param_test, X_disp_test, Y_shear_test) = split_and_convert((InParams, InDisp, OutShear),
                                                                                                                                                TEST_SIZE,
                                                                                                                                                VAL_SIZE,
                                                                                                                                                42,
                                                                                                                                                device,
                                                                                                                                                True )

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_param_train, X_disp_train, Y_shear_train), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_param_val, X_disp_val, Y_shear_val), BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_param_test, X_disp_test, Y_shear_test), BATCH_SIZE, shuffle=False)


# Initialize model, loss, and optimizer
model = LSTM_AE_Model2(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = Transformer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = Informer_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = xLSTM_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)
# model = LLaMA2_Model(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).to(device)

# model = torch.compile(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
early_stopping = EarlyStopping(PATIENCE, verbose=False, path=f"checkpoints/{type(model).__name__}.pt")
torchinfo.summary(model)

# Initialize tracking variables
train_losses, val_losses, train_r2_scores, val_r2_scores = [], [], [], []
best_val_loss = float("inf")  # Track the best validation loss
best_epoch = 0  # Track the epoch number corresponding to the best validation loss

# Training phase
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss, epoch_train_r2 = 0.0, 0.0

    for batch_param, batch_disp, batch_shear in train_loader:
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        outputs = model(batch_param, batch_disp) # Forward pass
        loss = criterion(outputs, batch_shear)
        r2 = r2_score(batch_shear.cpu(), outputs.cpu())
        loss.backward() # Backward pass
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_r2 += r2.item()

    # Calculate average training loss and R2 for the epoch
    epoch_train_loss /= len(train_loader)
    epoch_train_r2 /= len(train_loader)
    train_losses.append(epoch_train_loss)
    train_r2_scores.append(epoch_train_r2)

    # Validation phase
    model.eval()
    val_loss, val_r2 = 0.0, 0.0
    with torch.no_grad():
        for batch_param, batch_disp, batch_shear in val_loader:
            val_outputs = model(batch_param, batch_disp)
            val_loss += criterion(val_outputs, batch_shear).item()
            val_r2 += r2_score(batch_shear.cpu(), val_outputs.cpu())

        val_loss /= len(val_loader)
        val_r2 /= len(val_loader)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Train R2: {epoch_train_r2:.4f}, Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}')

    # Early Stopping
    if early_stopping(val_loss, model):
        print("Early stopping triggered")
        break

# Restore best weights
best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed
print(f"Best Epoch: {best_epoch}")
model.load_state_dict(torch.load(f"checkpoints/{type(model).__name__}.pt", weights_only=True))

# Final test evaluation
model.eval()
test_loss = test_r2 = 0

with torch.no_grad():
    for batch_param, batch_disp, batch_shear in test_loader:
        test_outputs = model(batch_param, batch_disp)
        test_loss += criterion(test_outputs, batch_shear).item()
        test_r2 += r2_score(test_outputs, batch_shear).item()

    test_loss /= len(test_loader)
    test_r2 /= len(test_loader)

print(f'Final Model Performance - Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}')

# Plotting
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
# new_input_parameters = denormalize(new_input_parameters.cpu().numpy(), param_scaler, sequence=False)
# new_input_displacement = denormalize(new_input_displacement.cpu().numpy(), disp_scaler, sequence=True)
# real_shear = denormalize(real_shear.cpu().numpy(), shear_scaler, sequence=True)
# predicted_shear = denormalize(predicted_shear.cpu().numpy(), shear_scaler, sequence=True)

# Move tensors to CPU for plotting and denormalization
new_input_parameters = denormalize2(new_input_parameters.cpu().numpy(), param_scaler, scaling_strategy='robust')
new_input_displacement = denormalize2(new_input_displacement.cpu().numpy(), disp_scaler, scaling_strategy='log_minmax', handle_small_values=True, small_value_threshold=1e-3)
real_shear = denormalize2(real_shear.cpu().numpy(), shear_scaler, scaling_strategy='log_minmax', handle_small_values=True, small_value_threshold=1e-3)
predicted_shear = denormalize2(predicted_shear.cpu().numpy(), shear_scaler, scaling_strategy='log_minmax', handle_small_values=True, small_value_threshold=1e-3)

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
