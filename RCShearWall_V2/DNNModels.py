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

# ======= Global Use Layers  =================================================================
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
# ===============================================================================================

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
            # Add smoothing layers
            nn.Conv1d(1, 1, kernel_size=3, padding=1),  # Temporal smoothing
            nn.GroupNorm(1, 1)  # Normalize across sequence
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
            predictions.unsqueeze(1),
            scale_factor=1,
            mode='linear',  # or 'nearest'
            align_corners=False
        ).squeeze(1)  # [batch_size, seq_length]

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


# ======= TimeSeriesTransformer =================================================================
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

        # Time Series Processing Branch
        self.series_encoder = nn.Sequential(
            nn.Linear(displacement_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1)
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
            nn.Linear(d_model, d_model * 2),  # Increased first layer size
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 4),  # Added an intermediate layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model * 2),  # Another intermediate layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, displacement_features)  # Final output layer
        )

        # Add temporal smoothing convolution
        self.temporal_smoother = nn.Sequential(
            nn.Conv1d(displacement_features, displacement_features,
                      kernel_size=5, padding=2, groups=displacement_features),
            nn.GELU(),
            nn.BatchNorm1d(displacement_features)
        )

    def forward(self, parameters, time_series):
        batch_size = parameters.shape[0]

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

        # Apply temporal smoothing  Reshape for 1D convolution
        smoothed_output = self.temporal_smoother(output.transpose(1, 2)).transpose(1, 2)

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
# ===============================================================================================

# ======= EnhancedTimeSeriesTransformer =========================================================
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
# ================================================================================================
