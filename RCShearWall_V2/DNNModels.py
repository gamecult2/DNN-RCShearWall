import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from xLSTM.model import xLSTM
from xLSTM.block import xLSTMBlock
from xLSTM.mlstm import mLSTM
from xLSTM.slstm import sLSTM
from informer.model import Informer
from LLaMA2.model import *

from informer.embed import DataEmbedding
from informer.attn import FullAttention, ProbAttention, AttentionLayer
from informer.encoder import Encoder, EncoderLayer
from informer.decoder import Decoder, DecoderLayer

# ======= Global Use Layers  ====================================================================
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(ProbSparseAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.1,
            batch_first=True  # Set batch_first=True
        )

    def forward(self, x):
        return self.attention(x, x, x)[0]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x):
        return self.attention(x, x, x)[0]

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty([x.shape[0], 1, 1]).bernoulli_(keep_prob)
        return x / keep_prob * mask

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x)
        return residual + x

class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.adaptive_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.adaptive_scale * self.pe[:x.size(1), :].unsqueeze(0)
# ===============================================================================================

# ======= Divers Model  =========================================================================
class AttentionLSTM_AEModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(AttentionLSTM_AEModel, self).__init__()
        self.sequence_length = sequence_length

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=300, num_heads=6, dropout=0.1)

        # LSTM Encoder
        self.lstm_encoder1 = nn.LSTM(displacement_features + parameters_features, 300, batch_first=True)
        self.lstm_encoder2 = nn.LSTM(300, 100, batch_first=True)

        # LSTM Decoder
        self.lstm_decoder1 = nn.LSTM(100, 300, batch_first=True)
        self.lstm_decoder2 = nn.LSTM(300, displacement_features, batch_first=True)

        # Reconstruction layers
        self.reconstruction = nn.Sequential(
            nn.Linear(displacement_features, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 1)
        )

    def forward(self, parameters_input, displacement_input):
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Encoding
        lstm_out, _ = self.lstm_encoder1(concatenated_tensor)

        # Attention Mechanism
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Further Encoding
        encoded_sequence, _ = self.lstm_encoder2(attention_out)

        # Decoding
        lstm_out, _ = self.lstm_decoder1(encoded_sequence)
        decoded_sequence, _ = self.lstm_decoder2(lstm_out)

        # Reconstruction and Output
        output_shear = self.reconstruction(decoded_sequence)

        return output_shear.reshape(output_shear.size(0), -1)

class xLSTM_Model(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(xLSTM_Model, self).__init__()
        self.sequence_length = sequence_length
        self.norm = nn.LayerNorm(200)
        self.norm2 = nn.LayerNorm(1)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(0.1)

        '''
        # #  xLSTM encoder
        self.lstm_encoder1 = xLSTMBlock(displacement_features + parameters_features, 200, num_layers=1, lstm_type="slstm")
        self.lstm_encoder2 = xLSTMBlock(200, 50, num_layers=1, lstm_type="slstm")
        # # xLSTM decoder
        self.lstm_decoder1 = xLSTMBlock(50, 200, num_layers=1, lstm_type="slstm")
        self.lstm_decoder2 = xLSTMBlock(200, displacement_features, num_layers=1, lstm_type="slstm")
        '''
        #  xLSTM encoder
        self.lstm_encoder1 = sLSTM(displacement_features + parameters_features, 200, num_layers=2)
        self.lstm_encoder2 = sLSTM(200, 50, num_layers=1)

        # xLSTM decoder
        self.lstm_decoder1 = sLSTM(50, 200, num_layers=1)
        self.lstm_decoder2 = sLSTM(200, displacement_features, num_layers=2)

        # Adjusting dimensions
        self.dense1 = nn.Linear(displacement_features, 200)
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
# ================================================================================================

# ======= Transformer Based Model  ==============================================================
class Transformer_Model(nn.Module):
    def __init__(self, num_features_params, num_features_disp, sequence_length,
                 d_model=256, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.sequence_length = sequence_length
        self.d_model = d_model  # Changed to 256 to be divisible by nhead=8

        # Combined input features (displacement + parameters)
        total_features = num_features_params + num_features_disp  # +1 for displacement

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

        # Output projection + Add smoothing techniques
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),  # Added LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
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


        # Add additional smoothing strategies
        predictions = F.interpolate(
            predictions,
            scale_factor=1,
            mode='linear',  # or 'nearest'
            align_corners=False
        ).squeeze(-1)  # [batch_size, seq_length]
        return predictions

class Transformer_Model_2(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length,
                 d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.0):
        super(Transformer_Model_2, self).__init__()
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

class TransformerAEModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length,
                 n_heads=4, n_encoder_layers=2, n_decoder_layers=2):
        super(TransformerAEModel, self).__init__()
        self.sequence_length = sequence_length

        # Input embedding
        self.input_embedding = nn.Linear(displacement_features + parameters_features, 256)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, 256))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=n_heads, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=n_heads, dim_feedforward=512, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Reconstruction and output layers
        self.reconstruction = nn.Sequential(
            nn.Linear(256, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 1)
        )

    def forward(self, parameters_input, displacement_input):
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Input embedding
        x = self.input_embedding(concatenated_tensor)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer Encoding
        x = x.transpose(0, 1)  # Sequence first for transformer
        encoded = self.transformer_encoder(x)

        # Transformer Decoding (using a memory-like approach)
        decoded = self.transformer_decoder(x, encoded)

        # Transpose back and pass through reconstruction layers
        decoded = decoded.transpose(0, 1)
        output_shear = self.reconstruction(decoded)

        return output_shear.reshape(output_shear.size(0), -1)
# ==============================================================================================

# ======= LSTM AutoEncoder ======================================================================
# LSTM-AE for joint parameter & time series processing (baseline)
class LSTM_AE_Model_1(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(LSTM_AE_Model_1, self).__init__()
        self.sequence_length = sequence_length

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
        # Repeat parameters for each time step
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate displacement and parameters
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Encoding & Decoding
        lstm_out, _ = self.lstm_encoder1(concatenated_tensor)
        encoded_sequence, _ = self.lstm_encoder2(lstm_out)
        lstm_out, _ = self.lstm_decoder1(encoded_sequence)
        decoded_sequence, _ = self.lstm_decoder2(lstm_out)

        # Dense layers
        x = self.dense1(decoded_sequence)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)

        # Output layer
        output_shear = self.output(x)
        output_shear = output_shear.reshape(output_shear.size(0), -1)

        return output_shear

# LSTM-AE with batch norm and GELU activation for improved stability and small value
class LSTM_AE_Model_2(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200):
        super(LSTM_AE_Model_2, self).__init__()
        self.sequence_length = sequence_length

        self.lstm_encoder1 = nn.LSTM(displacement_features + parameters_features, 300, batch_first=True)
        self.lstm_encoder2 = nn.LSTM(300, 50, batch_first=True)

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
        self.small_activation = nn.GELU()  # or custom activation

    def small_value_activation(self, x):
        return x * torch.sigmoid(x)  # Smooth transition near zero (Custom activation function for small values)

    def forward(self, parameters_input, displacement_input):
        # Distribute parameters
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate inputs
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Encoding & Decoding
        lstm_out, _ = self.lstm_encoder1(concatenated_tensor)
        encoded_sequence, _ = self.lstm_encoder2(lstm_out)
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
        # x = self.small_value_activation(x)  # Custom activation for small values
        x = self.small_activation(x)

        # Final output
        output_shear = self.output(x)
        output_shear = output_shear.reshape(batch_size, -1)

        return output_shear

# Hybrid LSTM-AE with separate processing branches and positional encoding for enhanced feature
class LSTM_AE_Model_3(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=400):
        super(LSTM_AE_Model_3, self).__init__()
        self.sequence_length = sequence_length
        self.parameters_features = parameters_features
        self.displacement_features = displacement_features

        # Parameter Processing Branch
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Time Series Processing Branch with multiple Conv1d layers
        self.series_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_model // 6, kernel_size=5, padding=2),  # First Conv1d layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=d_model // 6, out_channels=d_model // 4, kernel_size=5, padding=2),  # Second Conv1d layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=d_model // 4, out_channels=d_model // 2, kernel_size=5, padding=2),  # Third Conv1d layer
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Encoder with smoother transitions
        self.lstm_encoder1 = nn.LSTM(d_model, 150, batch_first=True)  # Reduced from 300
        self.lstm_encoder2 = nn.LSTM(150, 50, batch_first=True)

        # Decoder with smoother transitions
        self.lstm_decoder1 = nn.LSTM(50, 150, batch_first=True)
        self.lstm_decoder2 = nn.LSTM(150, d_model, batch_first=True)

        # Dense layers with reduced dimensions for parameter efficiency
        self.dense1 = nn.Linear(d_model, d_model // 2)
        self.layer_norm1 = nn.LayerNorm(d_model // 2)
        self.dropout1 = nn.Dropout(0.1)

        self.dense2 = nn.Linear(d_model // 2, 100)
        self.layer_norm2 = nn.LayerNorm(100)
        self.dropout2 = nn.Dropout(0.1)

        # Modified output layers
        self.pre_output = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, parameters_input, displacement_input):
        batch_size = parameters_input.size(0)

        # Process parameters: expand to match sequence length
        params_encoded = self.param_encoder(parameters_input)
        params_expanded = params_encoded.unsqueeze(1).expand(-1, self.sequence_length, -1)
        # print('params_expanded', params_expanded.shape)
        # Process displacement input: reduce dimensionality first
        displacement_input = displacement_input.unsqueeze(1)
        series_encoded = self.series_encoder(displacement_input)
        # print('series_encoded', series_encoded.shape)

        series_encoded = series_encoded.permute(0, 2, 1)
        # print('series_encoded', series_encoded.shape)

        # Combine features: concatenation + projection
        combined = torch.cat([params_expanded, series_encoded], dim=-1)

        # Add positional encoding
        combined = self.pos_encoder(combined)

        # Encoding and decoding
        lstm_out, _ = self.lstm_encoder1(combined)
        encoded_sequence, _ = self.lstm_encoder2(lstm_out)
        lstm_out, _ = self.lstm_decoder1(encoded_sequence)
        decoded_sequence, _ = self.lstm_decoder2(lstm_out)

        # Dense layers with layer normalization
        x = decoded_sequence.reshape(-1, decoded_sequence.size(-1))

        x = self.dense1(x)
        x = self.layer_norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Pre-output processing
        x = self.pre_output(x)

        # Final output
        output_shear = self.output(x)
        output_shear = output_shear.reshape(batch_size, -1)

        return output_shear
# =================================================================================================

# ======= TimeSeriesTransformer ===================================================================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200):
        super(TimeSeriesTransformer, self).__init__()
        self.sequence_length = sequence_length

        # Parameter Processing Branch
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Time Series Processing Branch with multiple Conv1d layers
        self.series_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_model // 6, kernel_size=5, padding=2),  # First Conv1d layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=d_model // 6, out_channels=d_model // 4, kernel_size=5, padding=2),  # Second Conv1d layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=d_model // 4, out_channels=d_model // 2, kernel_size=5, padding=2),  # Third Conv1d layer
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Positional Encoding or Rotary Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        # self.pos_encoder = RotaryPositionalEmbedding(d_model, max_len=sequence_length)

        # Main Processing Blocks
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(d_model, nhead=4, dropout=0.1)
            # EnhancedProcessingBlock(d_model, nhead=8, dropout=0.1)
            for _ in range(3)
        ])

        # Output Generation
        # self.output_layer = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(d_model // 2, displacement_features)
        # )
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # Reduced first layer size
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),  # Reduced further
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, d_model // 2),  # Increased back
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, displacement_features)  # Final output layer
        )

        # Add temporal smoothing convolution
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(displacement_features, displacement_features,
                      kernel_size=5, padding=2, groups=displacement_features),
            nn.GELU(),
            nn.BatchNorm1d(displacement_features)
        )

    def forward(self, parameters, time_series):
        # Expand parameters to match sequence length
        params_expanded = parameters.unsqueeze(1).expand(-1, self.sequence_length, -1)
        params_encoded = self.param_encoder(params_expanded)
        # print('params_encoded', params_encoded.shape)
        # Process time series
        # print('time_series', time_series.shape)
        # series_encoded = series_encoded
        series_encoded = self.series_encoder(time_series.unsqueeze(1)).permute(0, 2, 1)  # For Convolution
        # series_encoded = self.series_encoder(time_series.unsqueeze(-1))  # For Linear
        # print('series_encoded', series_encoded.shape)
        # Combine features
        combined = torch.cat([params_encoded, series_encoded], dim=-1)
        # print('combined', combined.shape)
        # Add positional encoding
        combined = self.pos_encoder(combined)

        # Process through main blocks
        x = combined
        for block in self.processing_blocks:
            x = block(x)

        # Generate output sequence
        output = self.output_layer(x)

        # Add time series as residual connection
        residual_output = output + time_series.unsqueeze(-1)

        # Apply temporal smoothing  Reshape for 1D convolution
        smoothed_output = self.temporal_smoother(residual_output.transpose(1, 2)).transpose(1, 2)

        return smoothed_output.squeeze(-1)

class ProcessingBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(ProcessingBlock, self).__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Convolutional FFN
        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1)
        )
        # self.conv_ffn = nn.Sequential(
        #     nn.Conv1d(d_model, d_model * 4, kernel_size=3, padding=1),  # Increased expansion
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(d_model * 4, d_model * 2, kernel_size=3, padding=1),  # Additional intermediate layer
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1)
        # )

        # LSTM layer
        self.lstm = nn.LSTM(
            d_model, d_model, bidirectional=False,
            batch_first=True
        )

        # LSTM layer
        self.lstm2 = nn.LSTM(
            d_model, d_model, bidirectional=False,
            batch_first=True
        )
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with skip connection
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Convolutional FFN with skip connection
        conv_input = x.transpose(1, 2)
        conv_output = self.conv_ffn(conv_input).transpose(1, 2)
        x = self.norm2(x + self.dropout(conv_output))

        # Bidirectional LSTM with skip connection
        lstm_output, _ = self.lstm(x)
        x = self.norm3(x + self.dropout(lstm_output))

        # Bidirectional LSTM with skip connection
        lstm_output, _ = self.lstm2(x)
        x = self.norm4(x + self.dropout(lstm_output))

        return x
# =================================================================================================

class ShearTransformer(nn.Module):
    def __init__(self,
                 parameter_features=17,
                 displacement_features=1,
                 sequence_length=500,
                 d_model=128,
                 nhead=8,
                 num_layers=3,
                 window_size=5,
                 future_steps=2,  # Number of future steps to predict
                 overlap_ratio=0.5):
        super(ShearTransformer, self).__init__()

        # Input feature dimensions
        self.parameter_features = parameter_features
        self.displacement_features = displacement_features
        self.total_input_features = parameter_features + displacement_features

        # Sliding window parameters
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio
        self.future_steps = future_steps

        # Calculate overlap size
        self.overlap_size = int(window_size * overlap_ratio)

        # Embedding layers
        self.parameter_embedding = nn.Linear(parameter_features, d_model)
        self.displacement_embedding = nn.Linear(displacement_features, d_model)

        # Positional Encoding
        self.positional_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer for predictions
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        # Temporal smoothing
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(1)
        )

    def forward(self, parameters, displacement):
        batch_size = parameters.size(0)

        # Prepare input tensors
        parameters = parameters
        displacement = displacement.unsqueeze(-1)

        # Initialize output tensor to store predictions
        full_output = torch.zeros(
            batch_size,
            self.sequence_length + self.future_steps,
            device=displacement.device
        )

        # Sliding window prediction with overlap
        for start in range(0, self.sequence_length - self.window_size + 1, self.window_size - self.overlap_size):
            # Determine the end of the current prediction window
            window_end = min(start + self.window_size + self.future_steps, self.sequence_length + self.future_steps)

            # Select current window of displacement
            window_displacement = displacement[:, start:start + self.window_size, :]

            # Repeat parameters to match window length
            window_parameters = parameters.unsqueeze(1).expand(-1, self.window_size, -1)

            # Combine parameters and displacement
            combined_input = torch.cat([window_parameters, window_displacement], dim=-1)

            # Embed inputs
            param_embedded = self.parameter_embedding(window_parameters)
            displ_embedded = self.displacement_embedding(window_displacement)

            # Combine embeddings
            combined_embedded = param_embedded + displ_embedded

            # Add positional encoding
            encoded_input = self.positional_encoder(combined_embedded)

            # Transformer encoding
            transformer_output = self.transformer_encoder(encoded_input)

            # Project to output
            window_prediction = self.output_projection(transformer_output)

            # Smooth prediction
            smoothed_prediction = self.temporal_smoother(
                window_prediction.transpose(1, 2)
            ).transpose(1, 2).squeeze(-1)

            # print('smoothed_prediction', smoothed_prediction.shape)
            # Update full output with prediction
            if start > 0:
                # Overlap blending
                overlap_region = min(self.overlap_size, full_output.shape[1] - start)
                weights = torch.linspace(1, 0, overlap_region, device=full_output.device)

                # Blend overlapping predictions
                full_output[:, start:start + overlap_region] = (
                        full_output[:, start:start + overlap_region] * (1 - weights) +
                        smoothed_prediction[:, :overlap_region] * weights
                )

            # Update predictions for current window
            prediction_slice = smoothed_prediction[:, :min(window_end - start, self.window_size + self.future_steps)]
            # print('prediction_slice', prediction_slice.shape)
            full_output[:, start:start + len(prediction_slice[0])] = prediction_slice

        # Return predictions up to sequence length + future steps
        return full_output[:, :self.sequence_length]

# ======= CrackTransformer ========================================================================
class CrackTimeSeriesTransformer(nn.Module):
    def __init__(self,
                 parameters_features=17,
                 displacement_features=1,
                 shear_features=1,
                 sequence_length=500,
                 crack_length=168,
                 d_model=256,
                 dropout=0.1,
                 num_processing_blocks=3):
        super(CrackTimeSeriesTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.crack_length = crack_length

        # Enhanced feature encoders with adaptive normalization
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.displacement_encoder = nn.Sequential(
            nn.Linear(displacement_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.shear_encoder = nn.Sequential(
            nn.Linear(shear_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Adaptive Positional Encoding
        self.pos_encoder = AdaptivePositionalEncoding(d_model, max_len=sequence_length)

        # Feature interaction projection
        self.feature_interaction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Configurable Processing Blocks
        self.processing_blocks = nn.ModuleList([
            CrackProcessingBlock(d_model, nhead=4, dropout=dropout)
            for _ in range(num_processing_blocks)
        ])

        # Output prediction heads with uncertainty estimation
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(d_model, dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, crack_length),
                nn.Softplus()  # Added for uncertainty estimation
            ) for _ in range(4)
        ])

    def forward(self, parameters, displacement_series, shear_series):
        # Expand parameters to match sequence length
        params_expanded = parameters.unsqueeze(1).expand(-1, self.sequence_length, -1)
        params_encoded = self.param_encoder(params_expanded)
        displacement_encoded = self.displacement_encoder(displacement_series.unsqueeze(-1))
        shear_encoded = self.shear_encoder(shear_series.unsqueeze(-1))

        # Combine features
        combined = torch.cat([params_encoded, displacement_encoded, shear_encoded], dim=-1)
        x = self.pos_encoder(combined)
        x = self.feature_interaction(x)

        # Process through configurable blocks
        for block in self.processing_blocks:
            x = block(x)

        # Generate output sequences with mean aggregation and uncertainty estimation
        outputs = [
            layer(x).mean(dim=1)
            for layer in self.output_layers
        ]

        return outputs[0], outputs[1], outputs[2], outputs[3]

class CrackProcessingBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrackProcessingBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model * 4),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 4, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, bidirectional=True,
            batch_first=True, num_layers=2
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.residual_block = ResidualBlock(d_model, dropout)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Convolutional feed-forward network
        conv_input = x.transpose(1, 2)
        conv_output = self.conv_ffn(conv_input).transpose(1, 2)
        x = self.norm2(x + conv_output)

        # Bidirectional LSTM
        lstm_output, _ = self.lstm(x)
        x = self.norm3(x + lstm_output)

        # Additional residual processing
        x = self.residual_block(x)
        return x
# =================================================================================================

# ======= EnhancedTimeSeriesTransformer ===========================================================
class EnhancedTimeSeriesTransformer(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=256):
        super(EnhancedTimeSeriesTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.parameters_features = parameters_features
        self.displacement_features = displacement_features

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Time series encoder
        self.series_encoder = nn.Sequential(
            nn.Linear(displacement_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Positional Encoding
        # self.positional_encoding = PositionalEncoding(d_model)
        self.positional_encoding = RotaryPositionalEmbedding(d_model, max_len=sequence_length)

        # Multiple processing blocks with progressive complexity
        self.processing_blocks = nn.ModuleList([
            EnhancedProcessingBlock(d_model, nhead=4, dropout=0.1)
            for _ in range(3)
        ])

        # Global feature extraction
        # self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)

        # Output Generation
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, displacement_features),
        )

    def forward(self, parameters, time_series):
        # Expand parameters to match sequence length
        params_expanded = parameters.unsqueeze(1).expand(-1, self.sequence_length, -1)
        params_encoded = self.param_encoder(params_expanded)

        # Process time series
        series_encoded = self.series_encoder(time_series.unsqueeze(-1))

        # Combine features
        combined = torch.cat([params_encoded, series_encoded], dim=-1)

        # Add positional encoding
        combined = self.positional_encoding(combined)

        # Process through enhanced blocks
        x = combined
        for block in self.processing_blocks:
            x = block(x)
        # print('x processing_blocks', x.shape)

        # Global feature extraction
        global_features = self.global_pooling(x.transpose(1, 2)).squeeze(-1)
        global_features_expanded = global_features.unsqueeze(1).expand(-1, x.size(1), -1)  # Shape: [32, 500, 256]
        x = x + global_features_expanded  # Combine global and sequence-level featuresc

        output = self.output_layer(x).squeeze(-1)
        # print('output', output.shape)

        return output

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class EnhancedProcessingBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(EnhancedProcessingBlock, self).__init__()

        # Advanced self-attention with adaptive layer normalization
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Adaptive layer normalization
        self.adaptive_norm = nn.LayerNorm(d_model)

        # Channel attention
        self.se_block = SEBlock(d_model)

        # Dilated convolutional FFN
        self.dilated_conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=2, dilation=2)
        )

        # Stochastic depth (drop path)
        self.drop_path_rate = dropout

        # Convolutional layer for temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Stochastic depth
        if self.training and self.drop_path_rate > 0:
            keep_prob = 1 - self.drop_path_rate
            mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep_prob)
            if keep_prob > 0:
                x = x / keep_prob * mask

        # Self-attention with adaptive normalization
        attn_output, _ = self.self_attn(x, x, x)
        x = self.adaptive_norm(x + self.dropout(attn_output))

        # Channel-wise attention
        x = self.se_block(x.transpose(1, 2)).transpose(1, 2)

        # Dilated convolutional FFN
        conv_input = x.transpose(1, 2)
        conv_output = self.dilated_conv_ffn(conv_input).transpose(1, 2)
        x = self.norm1(x + self.dropout(conv_output))

        # Temporal convolution
        temp_conv_input = x.transpose(1, 2)
        temp_conv_output = self.temporal_conv(temp_conv_input).transpose(1, 2)
        x = self.norm2(x + self.dropout(temp_conv_output))

        return x
# ==================================================================================================

# ======= Informer Model ===========================================================================
# 1. Transformer-based Informer Architecture
class InformerModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=512):
        super(InformerModel, self).__init__()
        self.sequence_length = sequence_length

        # Input processing
        self.input_projection = nn.Linear(displacement_features + parameters_features, d_model)

        # Informer-specific attention
        self.prob_sparse_attn = ProbSparseAttention(d_model, n_heads=8)

        # Encoder layers with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True,  # Set batch_first=True
            norm_first=True  # Apply normalization before attention
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
            enable_nested_tensor=True  # Enable nested tensor optimization
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),  # Add LayerNorm for stability
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),  # Add LayerNorm for stability
            nn.Linear(64, 1)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=sequence_length)

    def forward(self, parameters_input, displacement_input):
        # Distribute parameters across sequence
        distributed_params = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate inputs
        x = torch.cat([displacement_input.unsqueeze(-1), distributed_params], dim=-1)

        # Project to d_model dimensions
        x = self.input_projection(x)

        # Apply Informer attention and encoding
        x = self.prob_sparse_attn(x)
        x = self.encoder(x)

        # Decode to final output
        output = self.decoder(x)
        return output.squeeze(-1)

class InformerShearModel(nn.Module):
    def __init__(self,
                 parameters_features,
                 displacement_features,
                 sequence_length,
                 output_sequence_length=500,  # New parameter to match expected output
                 d_model=512,
                 n_heads=4,
                 e_layers=2,
                 d_layers=2,
                 d_ff=512,
                 dropout=0.1,
                 attn='prob',
                 activation='gelu'):
        super(InformerShearModel, self).__init__()

        # Model configuration
        self.sequence_length = sequence_length
        self.output_sequence_length = output_sequence_length

        # Create temporal marks for embedding
        temporal_marks = torch.zeros(1, sequence_length, 5).long()
        for i in range(sequence_length):
            temporal_marks[0, i, 0] = i % 12  # month
            temporal_marks[0, i, 1] = i % 31  # day
            temporal_marks[0, i, 2] = i % 7  # weekday
            temporal_marks[0, i, 3] = i % 24  # hour
            temporal_marks[0, i, 4] = i % 60  # minute
        self.register_buffer('temporal_marks', temporal_marks)

        # Input embedding
        self.embedding = DataEmbedding(
            displacement_features + parameters_features,
            d_model,
            embed_type='fixed',
            freq='h',
            dropout=dropout
        )

        # Attention mechanism
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor=5, attention_dropout=dropout),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor=5, attention_dropout=dropout),
                        d_model,
                        n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(200, output_sequence_length)  # Changed to match output_sequence_length
        )

    def forward(self, parameters_input, displacement_input):
        # Prepare input
        batch_size = parameters_input.size(0)

        # Distribute parameters across sequence
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate inputs
        x_enc = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Prepare temporal marks for the batch
        temporal_marks = self.temporal_marks.repeat(batch_size, 1, 1)

        # Embedding
        x_enc = self.embedding(x_enc, temporal_marks)

        # Encoder
        enc_out, _ = self.encoder(x_enc)

        # Decoder (simplified for single output prediction)
        # We'll use encoder output as both input and cross-attention input
        dec_out = self.decoder(x_enc, enc_out)

        # Project to output with specified sequence length
        output = self.projection(dec_out[:, -1, :])

        return output

class Informer_AE_Model(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, model_dim=512, factor=5, heads=8, dropout=0.1):
        super(Informer_AE_Model, self).__init__()
        self.sequence_length = sequence_length
        self.model_dim = model_dim

        # Informer model
        self.informer = Informer(
            enc_in=parameters_features + displacement_features,  # Input feature size
            dec_in=parameters_features + displacement_features,  # Input feature size
            c_out=displacement_features,  # Output feature size
            seq_len=sequence_length,
            label_len=sequence_length // 2,  # Typically half the sequence length
            out_len=sequence_length,  # Prediction length
            d_model=model_dim,
            n_heads=heads,
            e_layers=2,  # Encoder layers
            d_layers=1,  # Decoder layers
            dropout=dropout,
            factor=factor,
            activation="gelu",  # Informer activation
            output_attention=False,
        )

        # Dense layers for fine-tuning the output
        self.dense1 = nn.Linear(displacement_features, 300)
        self.batch_norm1 = nn.BatchNorm1d(300)
        self.dropout1 = nn.Dropout(0.1)

        self.dense2 = nn.Linear(300, 200)
        self.batch_norm2 = nn.BatchNorm1d(200)
        self.dropout2 = nn.Dropout(0.1)

        self.pre_output = nn.Linear(200, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, parameters_input, displacement_input):
        # Distribute parameters
        distributed_parameters = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Concatenate inputs
        concatenated_tensor = torch.cat([displacement_input.unsqueeze(-1), distributed_parameters], dim=-1)

        # Split for encoder and decoder
        x_enc = concatenated_tensor  # Encoder input
        x_mark_enc = None  # Optional, depends on Informer implementation
        x_dec = concatenated_tensor  # Decoder input (or future time steps if forecasting)
        x_mark_dec = None  # Optional, depends on Informer implementation

        # Pass through Informer
        informer_output = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)  # Ensure correct inputs

        # Dense layers with batch normalization
        batch_size = informer_output.size(0)
        time_steps = informer_output.size(1)

        # Reshape for batch norm
        x = informer_output.reshape(-1, informer_output.size(-1))

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
        x = torch.sigmoid(x)  # Use sigmoid activation for small values

        # Final output
        output_shear = self.output(x)
        output_shear = output_shear.reshape(batch_size, -1)

        return output_shear
# ==================================================================================================

# ======= LLaMA2 Model =============================================================================
class LLaMA2_Model(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length

        # Fixed values
        dim = 512
        n_layers = 2
        n_heads = 4

        config = ModelConfig(dim=dim,
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

# 2. LLaMA-inspired Architecture
class LLaMAInspiredModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(LLaMAInspiredModel, self).__init__()
        self.sequence_length = sequence_length
        hidden_dim = 512

        # Input processing
        self.input_proj = nn.Linear(displacement_features + parameters_features, hidden_dim)

        # RoPE positional embeddings
        self.pos_encoder = RotaryPositionalEmbedding(hidden_dim)

        # Multi-head attention with RoPE
        self.attention = MultiHeadAttention(hidden_dim, num_heads=8)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, parameters_input, displacement_input):
        distributed_params = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x = torch.cat([displacement_input.unsqueeze(-1), distributed_params], dim=-1)

        # Project and apply positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Self-attention and feed-forward
        x = self.attention(x)
        x = self.feed_forward(x)

        # Output processing
        output = self.output_layers(x)
        return output.squeeze(-1)
# ==================================================================================================

class LSTM_AE_Model_3_slice(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length, window_size, d_model=400):
        super(LSTM_AE_Model_3_slice, self).__init__()
        self.sequence_length = sequence_length
        self.parameters_features = parameters_features
        self.displacement_features = displacement_features
        self.window_size = window_size  # Window size for sliding window approach
        self.overlap_size = window_size // 2  # Half of the window size for overlap

        # Parameter Processing Branch
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Time Series Processing Branch
        self.series_encoder = nn.Sequential(
            nn.Linear(displacement_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)

        # Encoder
        self.lstm_encoder1 = nn.LSTM(d_model, 150, batch_first=True)
        self.lstm_encoder2 = nn.LSTM(150, 50, batch_first=True)

        # Decoder
        self.lstm_decoder1 = nn.LSTM(50, 150, batch_first=True)
        self.lstm_decoder2 = nn.LSTM(150, d_model, batch_first=True)

        # Dense layers reduction
        self.dense1 = nn.Linear(d_model, d_model // 2)
        self.layer_norm1 = nn.LayerNorm(d_model // 2)
        self.dropout1 = nn.Dropout(0.1)

        # Output layers
        self.dense2 = nn.Linear(d_model // 2, 100)
        self.layer_norm2 = nn.LayerNorm(100)
        self.dropout2 = nn.Dropout(0.1)

        self.pre_output = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, parameters_input, displacement_input):
        batch_size = parameters_input.size(0)
        full_output = torch.zeros(batch_size, self.sequence_length, device=parameters_input.device)

        # Adjust the sliding window to ensure we cover the entire sequence
        for start in range(0, self.sequence_length, self.window_size - self.overlap_size):
            # Ensure we don't go beyond the sequence length
            end = min(start + self.window_size, self.sequence_length)

            disp_window = displacement_input[:, start:end]

            # Pad the window if it's shorter than window_size
            if disp_window.size(1) < self.window_size:
                pad_size = self.window_size - disp_window.size(1)
                disp_window = F.pad(disp_window, (0, pad_size))

            disp_window = disp_window.unsqueeze(-1)

            # Encode parameters
            params_encoded = self.param_encoder(parameters_input)
            params_expanded = params_encoded.unsqueeze(1).repeat(1, self.window_size, 1)

            # Encode displacement
            series_encoded = self.series_encoder(disp_window)

            # Combine encoded params and series
            combined = torch.cat([params_expanded, series_encoded], dim=-1)
            combined = self.pos_encoder(combined)

            # LSTM Encoder and Decoder transition
            lstm_out, _ = self.lstm_encoder1(combined)
            encoded_sequence, _ = self.lstm_encoder2(lstm_out)
            lstm_out, _ = self.lstm_decoder1(encoded_sequence)
            decoded_sequence, _ = self.lstm_decoder2(lstm_out)

            # Dense Transformation
            x = decoded_sequence.reshape(-1, decoded_sequence.size(-1))
            x = self.dense1(x)
            x = self.layer_norm1(x)
            x = torch.relu(x)
            x = self.dropout1(x)

            x = self.dense2(x)
            x = self.layer_norm2(x)
            x = torch.relu(x)
            x = self.dropout2(x)

            x = self.pre_output(x)

            # Generating shear output for current window
            smoothed_prediction = self.output(x)
            smoothed_prediction = smoothed_prediction.view(batch_size, self.window_size)

            # Weighted overlap blending
            if start > 0:
                overlap_region = min(self.overlap_size, full_output.shape[1] - start)
                weights = torch.linspace(1, 0, overlap_region, device=full_output.device)

                # Blend overlapping predictions
                full_output[:, start:start + overlap_region] = (
                        full_output[:, start:start + overlap_region] * (1 - weights) +
                        smoothed_prediction[:, :overlap_region] * weights
                )

            # Add non-overlapping or remaining parts
            non_overlap_start = start + (self.overlap_size if start > 0 else 0)
            non_overlap_end = min(start + self.window_size, self.sequence_length)

            full_output[:, non_overlap_start:non_overlap_end] = smoothed_prediction[
                                                                :,
                                                                (self.overlap_size if start > 0 else 0):(non_overlap_end - non_overlap_start + (self.overlap_size if start > 0 else 0))
                                                                ]

        # Trim or pad to ensure exactly 500 points
        if full_output.size(1) > 500:
            full_output = full_output[:, :500]
        elif full_output.size(1) < 500:
            pad_size = 500 - full_output.size(1)
            full_output = F.pad(full_output, (0, pad_size))

        return full_output


class ShearTransformer(nn.Module):
    def __init__(self,
                 parameter_features=17,
                 displacement_features=1,
                 sequence_length=500,
                 d_model=128,
                 nhead=8,
                 num_layers=3,
                 window_size=10,
                 overlap_ratio=0.5,
                 prediction_horizon=3):
        super(ShearTransformer, self).__init__()

        # Input feature dimensions
        self.parameter_features = parameter_features
        self.displacement_features = displacement_features
        self.total_input_features = parameter_features + displacement_features

        # Sliding window parameters
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio  # Proportion of window that overlaps
        self.prediction_horizon = prediction_horizon  # Number of time steps to predict beyond the window

        # Calculate overlap size
        self.overlap_size = int(window_size * overlap_ratio)

        # Embedding layers
        self.parameter_embedding = nn.Linear(parameter_features, d_model)
        self.displacement_embedding = nn.Linear(displacement_features, d_model)

        # Positional Encoding
        self.positional_encoder = PositionalEncoding(d_model)

        # Main Processing Blocks
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(d_model, nhead=4, dropout=0.1)
            # EnhancedProcessingBlock(d_model, nhead=8, dropout=0.1)
            for _ in range(3)
        ])

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        # Temporal smoothing
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(1)
        )

    def forward(self, parameters, displacement):
        batch_size = parameters.size(0)

        # Expand parameters to match sequence length
        parameters = parameters
        displacement = displacement.unsqueeze(-1)

        # Initialize full_output with space for the sequence length + prediction horizon
        full_output = torch.zeros(batch_size, self.sequence_length + self.prediction_horizon, 1, device=displacement.device)

        # Sliding window prediction with overlap
        for start in range(0, self.sequence_length - self.window_size + 1, self.window_size - self.overlap_size):
            # Select window of displacement
            window_displacement = displacement[:, start:start + self.window_size, :]

            # Repeat parameters to match window length
            window_parameters = parameters.unsqueeze(1).expand(-1, self.window_size, -1)

            # Combine parameters and displacement
            combined_input = torch.cat([window_parameters, window_displacement], dim=-1)

            # Embed inputs
            param_embedded = self.parameter_embedding(window_parameters)
            displ_embedded = self.displacement_embedding(window_displacement)

            # Combine embeddings
            combined_embedded = param_embedded + displ_embedded

            # Add positional encoding
            encoded_combined = self.positional_encoder(combined_embedded)
            x = encoded_combined
            # Transformer encoding
            for block in self.processing_blocks:
                x = block(x)
            # transformer_output = self.transformer_encoder(encoded_input)

            # Project to output
            window_prediction = self.output_projection(x)

            # Smooth prediction
            window_prediction = self.temporal_smoother(window_prediction.transpose(1, 2)).transpose(1, 2)

            # Update full output with prediction
            # For future predictions (outside window), predict the next steps
            future_prediction = window_prediction[:, -self.prediction_horizon:, :]  # Get future steps

            # Ensure that the index range aligns
            end_index = start + self.window_size + self.prediction_horizon
            if end_index > full_output.shape[1]:
                end_index = full_output.shape[1]

            # Update the tensor with the correct ranges for the current window and future predictions
            full_output[:, start:end_index, :] = torch.cat(
                (window_prediction, future_prediction), dim=1
            )

            # When overlapping, we'll use a weighted average
            if start > 0:
                overlap_region = min(self.overlap_size, full_output.shape[1] - start)

                # Create weights for blending (linear interpolation)
                weights = torch.linspace(1, 0, overlap_region, device=full_output.device)
                weights = weights.view(1, -1, 1)

                # Blend the overlapping regions
                full_output[:, start:start + overlap_region, :] = (
                        full_output[:, start:start + overlap_region, :] * (1 - weights) +
                        window_prediction[:, :overlap_region, :] * weights
                )

        return full_output[:, :self.sequence_length, :].squeeze(-1)


class ShearTransformer2(nn.Module):
    def __init__(self,
                 parameter_features=17,
                 displacement_features=1,
                 sequence_length=500,
                 d_model=128,
                 nhead=8,
                 num_layers=3,
                 window_size=5,
                 future_steps=2,  # Number of future steps to predict
                 overlap_ratio=0.5):
        super(ShearTransformer2, self).__init__()

        # Input feature dimensions
        self.parameter_features = parameter_features
        self.displacement_features = displacement_features
        self.total_input_features = parameter_features + displacement_features

        # Sliding window parameters
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio
        self.future_steps = future_steps

        # Calculate overlap size
        self.overlap_size = int(window_size * overlap_ratio)

        # Embedding layers
        self.parameter_embedding = nn.Linear(parameter_features, d_model)
        self.displacement_embedding = nn.Linear(displacement_features, d_model)

        # Positional Encoding
        self.positional_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer for predictions
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        # Temporal smoothing
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(1)
        )

    def forward(self, parameters, displacement):
        batch_size = parameters.size(0)

        # Prepare input tensors
        parameters = parameters
        displacement = displacement.unsqueeze(-1)

        # Initialize output tensor to store predictions
        full_output = torch.zeros(
            batch_size,
            self.sequence_length + self.future_steps,
            device=displacement.device
        )

        # Sliding window prediction with overlap
        for start in range(0, self.sequence_length - self.window_size + 1, self.window_size - self.overlap_size):
            # Determine the end of the current prediction window
            window_end = min(start + self.window_size + self.future_steps, self.sequence_length + self.future_steps)

            # Select current window of displacement
            window_displacement = displacement[:, start:start + self.window_size, :]

            # Repeat parameters to match window length
            window_parameters = parameters.unsqueeze(1).expand(-1, self.window_size, -1)

            # Combine parameters and displacement
            combined_input = torch.cat([window_parameters, window_displacement], dim=-1)

            # Embed inputs
            param_embedded = self.parameter_embedding(window_parameters)
            displ_embedded = self.displacement_embedding(window_displacement)

            # Combine embeddings
            combined_embedded = param_embedded + displ_embedded

            # Add positional encoding
            encoded_input = self.positional_encoder(combined_embedded)

            # Transformer encoding
            transformer_output = self.transformer_encoder(encoded_input)

            # Project to output
            window_prediction = self.output_projection(transformer_output)

            # Smooth prediction
            smoothed_prediction = self.temporal_smoother(
                window_prediction.transpose(1, 2)
            ).transpose(1, 2).squeeze(-1)

            # print('smoothed_prediction', smoothed_prediction.shape)
            # Update full output with prediction
            if start > 0:
                # Overlap blending
                overlap_region = min(self.overlap_size, full_output.shape[1] - start)
                weights = torch.linspace(1, 0, overlap_region, device=full_output.device)

                # Blend overlapping predictions
                full_output[:, start:start + overlap_region] = (
                        full_output[:, start:start + overlap_region] * (1 - weights) +
                        smoothed_prediction[:, :overlap_region] * weights
                )

            # Update predictions for current window
            prediction_slice = smoothed_prediction[:, :min(window_end - start, self.window_size + self.future_steps)]
            # print('prediction_slice', prediction_slice.shape)
            full_output[:, start:start + len(prediction_slice[0])] = prediction_slice

        # Return predictions up to sequence length + future steps
        return full_output[:, :self.sequence_length]