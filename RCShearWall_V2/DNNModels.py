import torch
import torch.nn as nn
import math

from transformers import AutoModelForCausalLM, AutoTokenizer


# from informer import Informer


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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
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

'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # Convert to batch_first format
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
'''

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


# 2. LLaMA-inspired Architecture
class LLaMAInspiredModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(LLaMAInspiredModel, self).__init__()
        self.sequence_length = sequence_length
        hidden_dim = 512

        # RoPE positional embeddings
        self.pos_encoder = RotaryPositionalEmbedding(hidden_dim)

        # Input processing
        self.input_proj = nn.Linear(displacement_features + parameters_features, hidden_dim)

        # Multi-head attention with RoPE
        self.attention = RoPEMultiHeadAttention(hidden_dim, num_heads=8)

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


# 3. Extended LSTM (xLSTM) Architecture
class xLSTMModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, sequence_length):
        super(xLSTMModel, self).__init__()
        self.sequence_length = sequence_length
        hidden_dim = 256

        # Extended LSTM with multiple attention mechanisms
        self.xlstm = nn.ModuleList([
            nn.LSTM(displacement_features + parameters_features, hidden_dim, batch_first=True),
            MultiHeadAttention(hidden_dim, 4),
            nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        ])

        # Skip connections and layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim // 2)

        # Output processing
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, parameters_input, displacement_input):
        distributed_params = parameters_input.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x = torch.cat([displacement_input.unsqueeze(-1), distributed_params], dim=-1)

        # First LSTM layer
        x, _ = self.xlstm[0](x)

        # Multi-head attention
        attention_out = self.xlstm[1](x)
        x = x + attention_out  # Skip connection

        # Second LSTM layer
        x, _ = self.xlstm[2](x)
        x = self.layer_norm(x)

        # Output processing
        output = self.output_network(x)
        return output.squeeze(-1)


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


# suggest best
class TimeSeriesTransformer(nn.Module):
    def __init__(self, param_features=17, sequence_length=500, d_model=128):
        super(TimeSeriesTransformer, self).__init__()
        self.sequence_length = sequence_length

        # Parameter Processing Branch
        self.param_encoder = nn.Sequential(
            nn.Linear(param_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Time Series Processing Branch
        self.series_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Main Processing Blocks
        self.processing_blocks = nn.ModuleList([
            ProcessingBlock(d_model, nhead=4, dropout=0.1)
            for _ in range(2)
        ])

        # Output Generation
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, parameters, time_series):
        batch_size = parameters.shape[0]

        # Process parameters
        # Expand parameters to match sequence length
        params_expanded = parameters.unsqueeze(1).expand(-1, self.sequence_length, -1)
        params_encoded = self.param_encoder(params_expanded)

        # Process time series
        series_encoded = self.series_encoder(time_series.unsqueeze(-1))

        # Combine features
        combined = torch.cat([params_encoded, series_encoded], dim=-1)

        # Add positional encoding
        combined = self.pos_encoder(combined)

        # Process through main blocks
        x = combined
        for block in self.processing_blocks:
            x = block(x)

        # Generate output sequence
        output = self.output_layer(x)

        return output.squeeze(-1)


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

        # LSTM layer
        self.lstm = nn.LSTM(
            d_model, d_model // 2, bidirectional=True,
            batch_first=True
        )

        # LSTM layer
        self.lstm2 = nn.LSTM(
            d_model, d_model // 2, bidirectional=True,
            batch_first=True
        )
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with skip connection
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Convolutional FFN with skip connection
        # conv_input = x.transpose(1, 2)
        # conv_output = self.conv_ffn(conv_input).transpose(1, 2)
        # x = self.norm2(x + self.dropout(conv_output))

        # Bidirectional LSTM with skip connection
        lstm_output, _ = self.lstm(x)
        x = self.norm3(x + self.dropout(lstm_output))

        # Bidirectional LSTM with skip connection
        lstm_output, _ = self.lstm2(x)
        x = self.norm3(x + self.dropout(lstm_output))

        return x


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