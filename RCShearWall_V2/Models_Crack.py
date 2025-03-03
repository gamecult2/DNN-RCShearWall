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


class LocalAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size, dropout=0.05):
        super(LocalAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.dropout = dropout

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x):
        # Get the sequence length
        seq_len = x.size(1)

        # Create local attention mask (size [seq_len, seq_len])
        attn_mask = self.create_local_mask(seq_len)

        # Self-attention with local mask
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        return attn_output

    def create_local_mask(self, seq_len):
        # Create a mask that only allows attention to local neighbors within the window size
        mask = torch.zeros(seq_len, seq_len).to(torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        return mask


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

class CrackTimeSeriesTransformer2(nn.Module):
    def __init__(self,
                 num_horizontal=14,
                 num_vertical=12,
                 sequence_length=500,
                 parameters_features=17,
                 d_model=128,
                 dropout=0.1,
                 num_processing_blocks=2):
        super().__init__()
        self.num_horizontal = num_horizontal
        self.num_vertical = num_vertical
        self.total_panels = num_horizontal * num_vertical
        self.sequence_length = sequence_length

        # Create 2D panel position tensor
        self.panel_positions = self._create_panel_positions()

        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.displacement_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.shear_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Processing blocks
        self.processing_blocks = nn.ModuleList([
            CrackProcessingBlock2(d_model, nhead=4, dropout=dropout)
            for _ in range(num_processing_blocks)
        ])

        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, self.total_panels)
            ) for _ in range(4)
        ])

    def _create_panel_positions(self):
        # Create 2D tensor representing panel positions
        panel_positions = []
        for i in range(self.num_horizontal):
            for k in range(self.num_vertical):
                panel_positions.append((i, k))
        return torch.tensor(panel_positions, dtype=torch.float32)

    def forward(self, parameters, displacement_series, shear_series):
        # Encode inputs
        params_encoded = self.param_encoder(parameters.unsqueeze(1)).expand(-1, self.total_panels, -1)
        print("params_encoded shape:", params_encoded.shape)

        # Reshape and encode displacement and shear series
        displacement_encoded = self.displacement_encoder(displacement_series.unsqueeze(1))
        print("displacement_encoded shape:", displacement_encoded.shape)
        shear_encoded = self.shear_encoder(shear_series.unsqueeze(1))
        print("shear_encoded shape:", shear_encoded.shape)

        # Combine inputs
        combined_input = torch.cat([
            params_encoded,
            displacement_encoded,
            shear_encoded
        ], dim=1)
        print("combined_input shape:", combined_input.shape)

        # Encode combined input and expand to sequence length
        x = self.input_encoder(combined_input)
        print("encoded_input shape:", x.shape)
        x = x.unsqueeze(1).expand(-1, self.total_panels, -1)
        print("expanded_input shape:", x.shape)

        # Process through blocks
        for block in self.processing_blocks:
            x = block(x)
            print("block_output shape:", x.shape)

        # Generate outputs repeated over panel positions
        outputs = []
        for layer in self.output_layers:
            # Process features and apply mean aggregation
            output = layer(x).mean(dim=1)
            print("layer_output shape:", output.shape)

            # Repeat output over panel positions
            output = output.unsqueeze(1).expand(-1, self.total_panels, -1)
            print("repeated_output shape:", output.shape)
            outputs.append(output)
            print("repeated_output shape:", output.shape)
        print("repeated_output shape:", output[0].shape)
        return tuple(outputs)


class CrackProcessingBlock2(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


# =====================================================================================================
class PanelPositionEmbedding(nn.Module):
    def __init__(self, num_horizontal=14, num_vertical=12, embedding_dim=200):
        super().__init__()
        self.horizontal_embedding = nn.Embedding(num_horizontal, embedding_dim // 2)
        self.vertical_embedding = nn.Embedding(num_vertical, embedding_dim // 2)

    def forward(self, batch_size):
        horizontal_coords = torch.arange(14, device=self.horizontal_embedding.weight.device).repeat(12)
        vertical_coords = torch.repeat_interleave(
            torch.arange(12, device=self.vertical_embedding.weight.device), 14
        )

        horizontal_emb = self.horizontal_embedding(horizontal_coords)
        vertical_emb = self.vertical_embedding(vertical_coords)

        # print("horizontal_emb shape:", horizontal_emb.shape)
        # print("vertical_emb shape:", vertical_emb.shape)
        position_emb = torch.cat([horizontal_emb, vertical_emb], dim=-1)
        # print("position_emb shape:", position_emb.shape)
        position_emb = position_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # print("position_emb after shape:", position_emb.shape)
        return position_emb.permute(0, 2, 1)


class SpatialAwarenessAttention(nn.Module):
    def __init__(self, d_model, num_horizontal=14, num_vertical=12):
        super().__init__()
        self.num_panels = num_horizontal * num_vertical
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.spatial_correlation = nn.Sequential(
            nn.Linear(336, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, x, position_emb):
        # print("x shape:", x.shape)
        # print("position_emb shape:", position_emb.shape)
        x_with_position = torch.cat([x, position_emb], dim=-1)
        # print("x_with_position shape:", x_with_position.shape)
        x_spatial = self.spatial_correlation(x_with_position)
        # print("x_spatial shape:", x_spatial.shape)
        spatial_attn_output, _ = self.spatial_attention(x_spatial, x_spatial, x_spatial)
        return spatial_attn_output


class CrackProcessingBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
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
        self.conv = nn.Conv1d(in_channels=256, out_channels=168, kernel_size=1)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        conv_input = x.transpose(1, 2)
        conv_output = self.conv_ffn(conv_input).transpose(1, 2)
        x = self.norm2(x + conv_output)

        lstm_output, _ = self.lstm(x)
        x = self.norm3(x + lstm_output)

        x = self.residual_block(x)
        x_downsampled = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x_downsampled


class SpatialAwareCrackTimeSeriesTransformer(nn.Module):
    def __init__(self,
                 parameters_features=17,
                 displacement_features=1,
                 shear_features=1,
                 sequence_length=500,
                 crack_length=168,
                 num_horizontal=14,
                 num_vertical=12,
                 d_model=256,
                 dropout=0.2,
                 num_processing_blocks=2):
        super().__init__()
        self.sequence_length = sequence_length

        # Feature encoders
        self.panel_position_encoder = PanelPositionEmbedding(
            num_horizontal, num_vertical, embedding_dim=500
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.displacement_encoder = nn.Sequential(
            nn.Conv1d(displacement_features, d_model // 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model // 4)
        )

        self.shear_encoder = nn.Sequential(
            nn.Conv1d(shear_features, d_model // 4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model // 4)
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Spatial awareness modules
        self.processing_blocks = nn.ModuleList([
            nn.ModuleDict({
                'processing': CrackProcessingBlock(d_model, nhead=2, dropout=dropout),
                'spatial_attention': SpatialAwarenessAttention(d_model, num_horizontal, num_vertical)
            }) for _ in range(num_processing_blocks)
        ])

        # Output prediction heads
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(d_model),
                nn.Linear(d_model, crack_length)
            ) for _ in range(4)
        ])

    def forward(self, parameters, displacement_series, shear_series):
        batch_size = parameters.size(0)

        # Encode inputs
        params_encoded = self.param_encoder(parameters)
        # print("params_encoded shape:", params_encoded.shape)

        # Reshape and encode displacement and shear series
        displacement_encoded = self.displacement_encoder(displacement_series.unsqueeze(1)).transpose(1, 2)
        # print("displacement_encoded shape:", displacement_encoded.shape)
        shear_encoded = self.shear_encoder(shear_series.unsqueeze(1)).transpose(1, 2)
        # print("shear_encoded shape:", shear_encoded.shape)

        # Combine features
        combined = torch.cat([
            params_encoded.unsqueeze(1).expand(-1, self.sequence_length, -1),
            displacement_encoded,
            shear_encoded
        ], dim=-1)

        # Apply positional encoding
        x = self.pos_encoder(combined)

        # Get panel position embeddings
        position_emb = self.panel_position_encoder(batch_size)

        # Process through spatial-aware blocks
        for i, block in enumerate(self.processing_blocks):
            x = block['processing'](x)
            x = block['spatial_attention'](x, position_emb)

        # Generate outputs
        outputs = [layer(x).mean(dim=1) for layer in self.output_layers]

        return outputs[0], outputs[1], outputs[2], outputs[3]


# =====================================================================================================

# =====================================================================================================
'''
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=14, width=12):
        super().__init__()

        # Create position indices
        y_pos = torch.arange(height).float()
        x_pos = torch.arange(width).float()

        # Calculate number of dimensions for each coordinate
        dim_per_coord = d_model // 2
        div_term = torch.exp(torch.arange(0, dim_per_coord, 2).float() * (-math.log(10000.0) / dim_per_coord))

        # Create positional encodings
        pos_h = torch.zeros(height, dim_per_coord)
        pos_w = torch.zeros(width, dim_per_coord)

        pos_h[:, 0::2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pos_h[:, 1::2] = torch.cos(y_pos.unsqueeze(1) * div_term)

        pos_w[:, 0::2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pos_w[:, 1::2] = torch.cos(x_pos.unsqueeze(1) * div_term)

        # Create grid of positions
        pos_grid = []
        for i in range(height):
            for j in range(width):
                pos_grid.append(torch.cat([pos_h[i], pos_w[j]]))

        # Stack all positions
        pos_grid = torch.stack(pos_grid)  # Shape: [height*width, d_model]

        # Register as buffer (persistent state)
        self.register_buffer('pe', pos_grid.unsqueeze(0))  # Shape: [1, height*width, d_model]

        # Store grid dimensions
        self.height = height
        self.width = width

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        output = x + self.pe[:, :x.size(1)]
        return output


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.mlp(x))
        return x


class OutputBranch(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_layers=3):
        super().__init__()
        self.output_dim = output_dim
        self.pos_encoder = PositionalEncoding2D(input_dim)
        self.attention_layers = nn.ModuleList([
            AttentionBlock(dim=input_dim) for _ in range(num_attention_layers)
        ])

        # Final projections
        self.final_norm = nn.LayerNorm(input_dim)
        # self.final_linear = nn.Linear(input_dim, output_dim)
        # Final output projection
        self.final_linear = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        # Repeat the input for each position in the grid
        x = x.repeat(1, self.output_dim, 1)  # 168 = 14 * 12

        # Add positional encoding
        x = self.pos_encoder(x)

        # Process through attention layers
        for i, attention_layer in enumerate(self.attention_layers):
            x = attention_layer(x)

        # Final processing
        x = self.final_norm(x)

        # Average across the sequence dimension
        x = x.mean(dim=1)

        # Project to final output dimension
        x = self.final_linear(x)

        return x


class CrackDetectionModel3(nn.Module):
    def __init__(self, parameters_features, displacement_features, shear_features, crack_length, sequence_length, num_horizontal=14, num_vertical=12, d_model=512, dropout=0.1, num_attention_layers=3):
        super().__init__()

        # Updated encoder architecture, but output dimension remains the same as original
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.disp_encoder = nn.Sequential(
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

        self.input_norm = nn.LayerNorm(d_model)
        self.input_activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.output_branches = nn.ModuleList([
            OutputBranch(
                input_dim=d_model,
                output_dim=crack_length,  # 14 * 12 panels
                num_attention_layers=num_attention_layers
            ) for _ in range(4)
        ])

    def forward(self, parameters, displacement, shear):
        # Encode inputs
        p = self.param_encoder(parameters)  # [batch_size, 128]
        d = self.disp_encoder(displacement)  # [batch_size, 64]
        s = self.shear_encoder(shear)  # [batch_size, 64]

        # Combine features
        x = torch.cat([p, d, s], dim=-1)  # [batch_size, 256]

        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, 256]

        # Initial processing
        x = self.input_norm(x)
        x = self.input_activation(x)
        x = self.dropout(x)

        # Process through independent branches
        outputs = []
        for i, branch in enumerate(self.output_branches):
            outputs.append(branch(x))

        return tuple(outputs)
'''


# =====================================================================================================


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=14, width=12):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        dim_per_coord = d_model // 2

        # Pre-compute position matrices and division term
        position_h = torch.arange(height).float().unsqueeze(1)
        position_w = torch.arange(width).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_per_coord, 2).float() * (-math.log(10000.0) / dim_per_coord))

        # Compute height and width encodings
        pos_h = torch.zeros(height, dim_per_coord)
        pos_w = torch.zeros(width, dim_per_coord)

        pos_h[:, 0::2] = torch.sin(position_h * div_term)
        pos_h[:, 1::2] = torch.cos(position_h * div_term)
        pos_w[:, 0::2] = torch.sin(position_w * div_term)
        pos_w[:, 1::2] = torch.cos(position_w * div_term)

        # Create position grid using broadcasting
        h_grid = pos_h.unsqueeze(1).expand(-1, width, -1)
        w_grid = pos_w.unsqueeze(0).expand(height, -1, -1)

        # Combine the height and width encodings
        pos_grid = torch.cat([h_grid.reshape(height * width, -1), w_grid.reshape(height * width, -1)], dim=-1)

        # Register positional encoding as a buffer
        self.register_buffer('pe', pos_grid.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Combine normalization layers with residual connections
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Optimize MLP with a single dropout at the end
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm architecture for better training stability
        normed_x = self.norm1(x)
        x = x + self.attention(normed_x, normed_x, normed_x)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class OutputBranch(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_layers=3):
        super().__init__()
        self.pos_encoder = PositionalEncoding2D(input_dim)

        # Use ModuleList for better memory efficiency
        self.attention_layers = nn.ModuleList([
            AttentionBlock(input_dim) for _ in range(num_attention_layers)
        ])

        # Optimize final projections with fewer layers
        self.final_norm = nn.LayerNorm(input_dim)
        self.final_linear = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # More efficient repeat operation
        x = x.expand(-1, self.pos_encoder.output_size, -1)
        x = self.pos_encoder(x)

        # Process through attention layers
        for layer in self.attention_layers:
            x = layer(x)

        # Efficient final processing
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.final_linear(x)


class CrackDetectionModel3(nn.Module):
    def __init__(
            self,
            parameters_features,
            displacement_features,
            shear_features,
            crack_length,
            sequence_length,
            num_horizontal=14,
            num_vertical=12,
            d_model=512,
            dropout=0.1,
            num_attention_layers=3
    ):
        super().__init__()

        # Optimize encoder architectures
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )

        # Shared architecture for displacement and shear
        feature_encoder = lambda in_features: nn.Sequential(
            nn.Linear(in_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )

        self.disp_encoder = feature_encoder(displacement_features)
        self.shear_encoder = feature_encoder(shear_features)

        # Combined normalization and activation
        self.input_processor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Parallel output branches
        self.output_branches = nn.ModuleList([
            OutputBranch(
                input_dim=d_model,
                output_dim=crack_length,
                num_attention_layers=num_attention_layers
            ) for _ in range(4)
        ])

    def forward(self, parameters, displacement, shear):
        # Encode inputs in parallel
        x = torch.cat([
            self.param_encoder(parameters),
            self.disp_encoder(displacement),
            self.shear_encoder(shear)
        ], dim=-1)

        # Process input
        x = self.input_processor(x.unsqueeze(1))

        # Process through branches in parallel
        return tuple(branch(x) for branch in self.output_branches)


# =====================================================================================================

class CrackPatternCNN(nn.Module):
    def __init__(self, parameter_dim=17, displacement_dim=500, shear_dim=500):
        super(CrackPatternCNN, self).__init__()

        # Define constants for the model architecture
        self.parameter_dim = parameter_dim
        self.displacement_dim = displacement_dim
        self.shear_dim = shear_dim

        # Parameter embedding network (MLP with more layers)
        self.parameter_embedding = nn.Sequential(
            nn.Linear(parameter_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Combine all inputs and reshape for convolutional processing
        combined_features = 128 + displacement_dim + shear_dim
        self.initial_features = combined_features

        # Enhanced CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.initial_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        # Skip connection adapters
        self.skip_adapter1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.skip_adapter2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.skip_adapter3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(128)

        # Shared feature extraction before splitting into separate output branches
        self.shared_features1 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1)
        self.shared_features2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1)
        self.shared_bn1 = nn.BatchNorm2d(96)
        self.shared_bn2 = nn.BatchNorm2d(64)

        # Output branches for the four predictions with additional layers for refinement
        # Angle 1 branch
        self.a1_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Width 1 branch
        self.c1_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Angle 2 branch
        self.a2_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Width 2 branch
        self.c2_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, parameters_input, displacement_input, shear_input):
        """
        Forward pass through the enhanced CNN

        Args:
            parameters_input: Tensor of shape (batch, 17) - geometric and material parameters
            displacement_input: Tensor of shape (batch, 2) - min/max displacement
            shear_input: Tensor of shape (batch, 2) - min/max shear

        Returns:
            Tuple of 4 tensors (a1_pred, c1_pred, a2_pred, c2_pred), each with shape (batch, 168)
        """
        batch_size = parameters_input.size(0)

        # Process parameters through embedding network
        param_embedding = self.parameter_embedding(parameters_input)  # (batch, 128)

        # Combine all inputs
        combined = torch.cat([param_embedding, displacement_input, shear_input], dim=1)  # (batch, 128+2+2)

        # Reshape to spatial format for convolution - initialize with a spatially uniform representation
        h, w = 14, 12  # Target spatial dimensions
        x = combined.view(batch_size, combined.size(1), 1, 1).expand(batch_size, combined.size(1), h, w)

        # Enhanced CNN layers with skip connections
        x1 = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, 14, 12)
        x1_adapted = self.skip_adapter1(x1)  # (batch, 128, 14, 12)

        x2 = F.relu(self.bn2(self.conv2(x1)))  # (batch, 128, 14, 12)
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (batch, 128, 14, 12)
        x4 = F.relu(self.bn4(self.conv4(x3 + x1_adapted)))  # (batch, 128, 14, 12)

        # Apply dropout for regularization
        x4 = self.dropout(x4)

        # Second block with skip connection
        x4_adapted = self.skip_adapter2(x4)  # (batch, 256, 14, 12)
        x5 = F.relu(self.bn5(self.conv5(x4)))  # (batch, 256, 14, 12)
        x6 = F.relu(self.bn6(self.conv6(x5 + x4_adapted)))  # (batch, 256, 14, 12)

        # Apply dropout again
        x6 = self.dropout(x6)

        # Third block with skip connection
        x6_adapted = self.skip_adapter3(x6)  # (batch, 128, 14, 12)
        x7 = F.relu(self.bn7(self.conv7(x6)))  # (batch, 128, 14, 12)

        # Enhanced shared feature extraction with multiple layers
        shared = F.relu(self.shared_bn1(self.shared_features1(x7 + x6_adapted)))  # (batch, 96, 14, 12)
        shared = F.relu(self.shared_bn2(self.shared_features2(shared)))  # (batch, 64, 14, 12)

        # Apply final dropout
        shared = self.dropout(shared)

        # Generate the four separate predictions with refined output branches
        a1 = self.a1_refine(shared)  # (batch, 1, 14, 12)
        c1 = self.c1_refine(shared)  # (batch, 1, 14, 12)
        a2 = self.a2_refine(shared)  # (batch, 1, 14, 12)
        c2 = self.c2_refine(shared)  # (batch, 1, 14, 12)

        # Reshape to target dimensions (batch, 168)
        a1_pred = a1.view(batch_size, -1)  # (batch, 168)
        c1_pred = c1.view(batch_size, -1)  # (batch, 168)
        a2_pred = a2.view(batch_size, -1)  # (batch, 168)
        c2_pred = c2.view(batch_size, -1)  # (batch, 168)

        return a1_pred, c1_pred, a2_pred, c2_pred

# ===== GOOD ===
class CrackPatternCNN(nn.Module):
    def __init__(self, parameter_dim=17, displacement_dim=500, shear_dim=500):
        super(CrackPatternCNN, self).__init__()

        # Define constants for the model architecture
        self.parameter_dim = parameter_dim
        self.displacement_dim = displacement_dim
        self.shear_dim = shear_dim

        # Parameter embedding network (MLP with more layers)
        self.parameter_embedding = nn.Sequential(
            nn.Linear(parameter_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )

        # Separate embeddings for displacement and shear
        self.disp_embedding = nn.Sequential(
            nn.Linear(displacement_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )

        self.shear_embedding = nn.Sequential(
            nn.Linear(shear_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )

        # Combine all inputs and reshape for convolutional processing
        combined_features = 128 + 128 + 128  # 128 from parameter, displacement, and shear embeddings
        self.initial_features = combined_features

        # Enhanced CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.initial_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        # Skip connection adapters
        self.skip_adapter1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.skip_adapter2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.skip_adapter3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(128)

        # Shared feature extraction before splitting into separate output branches
        self.shared_features1 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1)
        self.shared_features2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1)
        self.shared_bn1 = nn.BatchNorm2d(96)
        self.shared_bn2 = nn.BatchNorm2d(64)

        # Output branches for the four predictions with additional layers for refinement
        # Angle 1 branch
        self.a1_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Width 1 branch
        self.c1_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Angle 2 branch
        self.a2_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Width 2 branch
        self.c2_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, parameters_input, displacement_input, shear_input):
        """
        Forward pass through the enhanced CNN

        Args:
            parameters_input: Tensor of shape (batch, 17) - geometric and material parameters
            displacement_input: Tensor of shape (batch, 500) - displacement values
            shear_input: Tensor of shape (batch, 500) - shear values

        Returns:
            Tuple of 4 tensors (a1_pred, c1_pred, a2_pred, c2_pred), each with shape (batch, 168)
        """
        batch_size = parameters_input.size(0)

        # Process parameters through embedding network
        param_embedding = self.parameter_embedding(parameters_input)  # (batch, 128)

        # Process displacement and shear inputs through their respective embeddings
        disp_embedding = self.disp_embedding(displacement_input)  # (batch, 128)
        shear_embedding = self.shear_embedding(shear_input)  # (batch, 128)

        # Combine all inputs
        combined = torch.cat([param_embedding, disp_embedding, shear_embedding], dim=1)  # (batch, 128+128+128)

        # Reshape to spatial format for convolution - initialize with a spatially uniform representation
        h, w = 14, 12  # Target spatial dimensions
        x = combined.view(batch_size, combined.size(1), 1, 1).expand(batch_size, combined.size(1), h, w)

        # Enhanced CNN layers with skip connections
        x1 = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, 14, 12)
        x1_adapted = self.skip_adapter1(x1)  # (batch, 128, 14, 12)

        x2 = F.relu(self.bn2(self.conv2(x1)))  # (batch, 128, 14, 12)
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (batch, 128, 14, 12)
        x4 = F.relu(self.bn4(self.conv4(x3 + x1_adapted)))  # (batch, 128, 14, 12)

        # Apply dropout for regularization
        x4 = self.dropout(x4)

        # Second block with skip connection
        x4_adapted = self.skip_adapter2(x4)  # (batch, 256, 14, 12)
        x5 = F.relu(self.bn5(self.conv5(x4)))  # (batch, 256, 14, 12)
        x6 = F.relu(self.bn6(self.conv6(x5 + x4_adapted)))  # (batch, 256, 14, 12)

        # Apply dropout again
        x6 = self.dropout(x6)

        # Third block with skip connection
        x6_adapted = self.skip_adapter3(x6)  # (batch, 128, 14, 12)
        x7 = F.relu(self.bn7(self.conv7(x6)))  # (batch, 128, 14, 12)

        # Enhanced shared feature extraction with multiple layers
        shared = F.relu(self.shared_bn1(self.shared_features1(x7 + x6_adapted)))  # (batch, 96, 14, 12)
        shared = F.relu(self.shared_bn2(self.shared_features2(shared)))  # (batch, 64, 14, 12)

        # Apply final dropout
        shared = self.dropout(shared)

        # Generate the four separate predictions with refined output branches
        a1 = self.a1_refine(shared)  # (batch, 1, 14, 12)
        c1 = self.c1_refine(shared)  # (batch, 1, 14, 12)
        a2 = self.a2_refine(shared)  # (batch, 1, 14, 12)
        c2 = self.c2_refine(shared)  # (batch, 1, 14, 12)

        # Reshape to target dimensions (batch, 168)
        a1_pred = a1.view(batch_size, -1)  # (batch, 168)
        c1_pred = c1.view(batch_size, -1)  # (batch, 168)
        a2_pred = a2.view(batch_size, -1)  # (batch, 168)
        c2_pred = c2.view(batch_size, -1)  # (batch, 168)

        return a1_pred, c1_pred, a2_pred, c2_pred

'''
class CrackPatternCNN(nn.Module):
    def __init__(self, parameter_dim=17, displacement_dim=500, shear_dim=500, time_steps=100):
        super(CrackPatternCNN, self).__init__()

        # Define constants for the model architecture
        self.parameter_dim = parameter_dim
        self.displacement_dim = displacement_dim
        self.shear_dim = shear_dim
        self.time_steps = time_steps

        # Parameter embedding network (MLP with more layers)
        self.parameter_embedding = nn.Sequential(
            nn.Linear(parameter_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Time-series embedding for displacement and shear using 1D convolutions
        self.disp_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.shear_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Combine all inputs and reshape for convolutional processing
        combined_features = 128 + 128 + 128  # 128 from parameter, displacement, and shear embeddings
        self.initial_features = combined_features

        # Enhanced CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.initial_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        # Skip connection adapters
        self.skip_adapter1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.skip_adapter2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.skip_adapter3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(128)

        # Shared feature extraction before splitting into separate output branches
        self.shared_features1 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1)
        self.shared_features2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1)
        self.shared_bn1 = nn.BatchNorm2d(96)
        self.shared_bn2 = nn.BatchNorm2d(64)

        # Output branches for the four predictions with additional layers for refinement
        # Angle 1 branch
        self.a1_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Width 1 branch
        self.c1_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Angle 2 branch
        self.a2_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Width 2 branch
        self.c2_refine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, parameters_input, displacement_input, shear_input):
        """
        Forward pass through the enhanced CNN

        Args:
            parameters_input: Tensor of shape (batch, 17) - geometric and material parameters
            displacement_input: Tensor of shape (batch, time_steps) - displacement time-series
            shear_input: Tensor of shape (batch, time_steps) - shear time-series

        Returns:
            Tuple of 4 tensors (a1_pred, c1_pred, a2_pred, c2_pred), each with shape (batch, 168)
        """
        batch_size = parameters_input.size(0)

        # Process parameters through embedding network
        param_embedding = self.parameter_embedding(parameters_input)  # (batch, 128)

        # Process displacement and shear inputs through their respective embeddings
        disp_embedding = self.disp_embedding(displacement_input.unsqueeze(1))  # (batch, 128, time_steps)
        shear_embedding = self.shear_embedding(shear_input.unsqueeze(1))  # (batch, 128, time_steps)

        # Take the last timestep's output from the embeddings (or apply pooling if desired)
        disp_embedding = disp_embedding[:, :, -1]  # (batch, 128)
        shear_embedding = shear_embedding[:, :, -1]  # (batch, 128)

        # Combine all inputs
        combined = torch.cat([param_embedding, disp_embedding, shear_embedding], dim=1)  # (batch, 128+128+128)

        # Reshape to spatial format for convolution - initialize with a spatially uniform representation
        h, w = 14, 12  # Target spatial dimensions
        x = combined.view(batch_size, combined.size(1), 1, 1).expand(batch_size, combined.size(1), h, w)

        # Enhanced CNN layers with skip connections
        x1 = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, 14, 12)
        x1_adapted = self.skip_adapter1(x1)  # (batch, 128, 14, 12)

        x2 = F.relu(self.bn2(self.conv2(x1)))  # (batch, 128, 14, 12)
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (batch, 128, 14, 12)
        x4 = F.relu(self.bn4(self.conv4(x3 + x1_adapted)))  # (batch, 128, 14, 12)

        # Apply dropout for regularization
        x4 = self.dropout(x4)

        # Second block with skip connection
        x4_adapted = self.skip_adapter2(x4)  # (batch, 256, 14, 12)
        x5 = F.relu(self.bn5(self.conv5(x4)))  # (batch, 256, 14, 12)
        x6 = F.relu(self.bn6(self.conv6(x5 + x4_adapted)))  # (batch, 256, 14, 12)

        # Apply dropout again
        x6 = self.dropout(x6)

        # Third block with skip connection
        x6_adapted = self.skip_adapter3(x6)  # (batch, 128, 14, 12)
        x7 = F.relu(self.bn7(self.conv7(x6)))  # (batch, 128, 14, 12)

        # Enhanced shared feature extraction with multiple layers
        shared = F.relu(self.shared_bn1(self.shared_features1(x7 + x6_adapted)))  # (batch, 96, 14, 12)
        shared = F.relu(self.shared_bn2(self.shared_features2(shared)))  # (batch, 64, 14, 12)

        # Apply final dropout
        shared = self.dropout(shared)

        # Generate the four separate predictions with refined output branches
        a1 = self.a1_refine(shared)  # (batch, 1, 14, 12)
        c1 = self.c1_refine(shared)  # (batch, 1, 14, 12)
        a2 = self.a2_refine(shared)  # (batch, 1, 14, 12)
        c2 = self.c2_refine(shared)  # (batch, 1, 14, 12)

        # Reshape to target dimensions (batch, 168)
        a1_pred = a1.view(batch_size, -1)  # (batch, 168)
        c1_pred = c1.view(batch_size, -1)  # (batch, 168)
        a2_pred = a2.view(batch_size, -1)  # (batch, 168)
        c2_pred = c2.view(batch_size, -1)  # (batch, 168)

        return a1_pred, c1_pred, a2_pred, c2_pred

'''

class CrackPatternTransformer(nn.Module):
    def __init__(self, parameter_dim=17, displacement_dim=500, shear_dim=500,
                 d_model=128, nhead=8, num_decoder_layers=6, dim_feedforward=512,
                 dropout=0.1, output_size=168):
        super(CrackPatternTransformer, self).__init__()

        # Define constants for the model architecture
        self.parameter_dim = parameter_dim
        self.displacement_dim = displacement_dim
        self.shear_dim = shear_dim
        self.d_model = d_model
        self.output_size = output_size

        # Parameter embedding network (MLP)
        self.parameter_embedding = nn.Sequential(
            nn.Linear(parameter_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU()
        )

        # Embedding for displacement and shear inputs
        self.disp_shear_embedding = nn.Linear(displacement_dim + shear_dim, d_model)

        # Positional encoding (learned)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))

        # Using TransformerEncoderLayer for a decoder-only architecture
        # (this simulates a GPT-style decoder-only transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_decoder_layers
        )

        # Output heads for each prediction
        self.a1_head = nn.Linear(d_model, output_size)
        self.c1_head = nn.Linear(d_model, output_size)
        self.a2_head = nn.Linear(d_model, output_size)
        self.c2_head = nn.Linear(d_model, output_size)

    def generate_mask(self, seq_len, device):
        # Generate causal mask for decoder-only transformer
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, parameters_input, displacement_input, shear_input):
        """
        Forward pass through the transformer encoder (used as decoder-only style transformer)

        Args:
            parameters_input: Tensor of shape (batch, 17) - geometric and material parameters
            displacement_input: Tensor of shape (batch, 2) - min/max displacement
            shear_input: Tensor of shape (batch, 2) - min/max shear

        Returns:
            Tuple of 4 tensors (a1_pred, c1_pred, a2_pred, c2_pred), each with shape (batch, 168)
        """
        batch_size = parameters_input.size(0)
        device = parameters_input.device

        # Process parameters through embedding network
        param_embedding = self.parameter_embedding(parameters_input)  # (batch, d_model)

        # Process displacement and shear inputs
        disp_shear_combined = torch.cat([displacement_input, shear_input], dim=1)  # (batch, 4)
        disp_shear_embedding = self.disp_shear_embedding(disp_shear_combined)  # (batch, d_model)

        # Combine embeddings and reshape for transformer input
        # Creating a sequence with length 2: [param_embedding, disp_shear_embedding]
        sequence = torch.stack([param_embedding, disp_shear_embedding], dim=1)  # (batch, 2, d_model)

        # Add positional encoding
        sequence = sequence + self.pos_encoder

        # Generate causal mask for decoder-only transformer
        mask = self.generate_mask(sequence.size(1), device)

        # Pass through transformer encoder with causal masking (simulating decoder-only behavior)
        transformer_output = self.transformer_encoder(
            sequence,
            mask=mask
        )  # (batch, 2, d_model)

        # Extract the final hidden state
        hidden = transformer_output[:, -1, :]  # (batch, d_model)

        # Generate predictions through output heads
        a1_pred = self.a1_head(hidden)  # (batch, output_size)
        c1_pred = self.c1_head(hidden)  # (batch, output_size)
        a2_pred = self.a2_head(hidden)  # (batch, output_size)
        c2_pred = self.c2_head(hidden)  # (batch, output_size)

        return a1_pred, c1_pred, a2_pred, c2_pred

# =====================================================================================================

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),  # Increased width
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output

        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x


class CrackDetectionModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, shear_features, crack_length=168,
                 d_model=512, dropout=0.1, num_attention_layers=3):
        super().__init__()

        # Input encoders
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
            nn.Linear(d_model // 4, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.shear_encoder = nn.Sequential(
            nn.Linear(shear_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.crack_length = crack_length

        # Rest of the model remains the same
        self.attention_branch1 = nn.ModuleList([
            AttentionBlock(d_model) for _ in range(num_attention_layers)
        ])

        self.attention_branch2 = nn.ModuleList([
            AttentionBlock(d_model) for _ in range(num_attention_layers)
        ])

        self.final_norm1 = nn.LayerNorm(d_model)
        self.final_norm2 = nn.LayerNorm(d_model)
        # Output MLPs
        self.mlp_a1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Changed to output single value per panel
        )

        self.mlp_a2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Changed to output single value per panel
        )

        self.mlp_c1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Changed to output single value per panel
        )

        self.mlp_c2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Changed to output single value per panel
        )
    def forward(self, parameters, displacement, shear):
        # Encode inputs
        param_encoded = self.param_encoder(parameters)  # [batch, d_model//2]
        disp_encoded = self.displacement_encoder(displacement)  # [batch, d_model//4]
        shear_encoded = self.shear_encoder(shear)  # [batch, d_model//4]

        # Combine encoded inputs
        combined = torch.cat([param_encoded, disp_encoded, shear_encoded], dim=-1)  # [batch, d_model]

        # Project to full dimension and repeat for each panel
        # combined = self.panel_projection(combined)  # [batch, d_model]
        combined = combined.unsqueeze(1).expand(-1, self.crack_length, -1)  # [batch, 168, d_model]

        # Add positional encoding
        x = self.pos_encoder(combined)  # [batch, 168, d_model]

        # Process through attention branches
        x1 = x
        for layer in self.attention_branch1:
            x1 = layer(x1)
        x1 = self.final_norm1(x1)  # [batch, 168, d_model]

        x2 = x
        for layer in self.attention_branch2:
            x2 = layer(x2)
        x2 = self.final_norm2(x2)  # [batch, 168, d_model]

        # Generate predictions
        a1_pred = self.mlp_a1(x1).squeeze(-1)  # [batch, 168]
        a2_pred = self.mlp_a2(x1).squeeze(-1)  # [batch, 168]
        c1_pred = self.mlp_c1(x2).squeeze(-1)  # [batch, 168]
        c2_pred = self.mlp_c2(x2).squeeze(-1)  # [batch, 168]

        return a1_pred, c1_pred, a2_pred, c2_pred


class CrackDetectionModel2(nn.Module):
    def __init__(self, parameters_features, displacement_features, shear_features, crack_length,
                 num_horizontal=14, num_vertical=12, d_model=512, dropout=0.1, num_attention_layers=6,
                 lstm_hidden_size=128, num_lstm_layers=2):
        super().__init__()

        # Input encoders
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )

        # LSTM for displacement - note the input_size is now 1
        self.displacement_encoder = nn.LSTM(
            input_size=1,  # Since we'll process one feature at a time
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.displacement_projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )

        # LSTM for shear - note the input_size is now 1
        self.shear_encoder = nn.LSTM(
            input_size=1,  # Since we'll process one feature at a time
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.shear_projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model)

        # Two separate attention branches
        self.attention_branch1 = nn.ModuleList([
            AttentionBlock(d_model) for _ in range(num_attention_layers)
        ])  # For a1_pred and a2_pred

        self.attention_branch2 = nn.ModuleList([
            AttentionBlock(d_model) for _ in range(num_attention_layers)
        ])  # For c1_pred and c2_pred

        self.final_norm = nn.LayerNorm(d_model)

        # Separate MLPs for each output with softmax
        self.mlp_a1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, crack_length)
        )

        self.mlp_a2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, crack_length)
        )

        self.mlp_c1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, crack_length)
        )

        self.mlp_c2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, crack_length)
        )

    def forward(self, parameters, displacement, shear):
        # Encode parameters
        param_encoded = self.param_encoder(parameters)  # [32, 256]

        # Reshape displacement and shear for LSTM
        displacement = displacement.unsqueeze(-1)  # [32, 500, 1]
        shear = shear.unsqueeze(-1)  # [32, 500, 1]

        # Process through LSTMs
        _, (disp_hidden, _) = self.displacement_encoder(displacement)
        disp_encoded = self.displacement_projection(disp_hidden[-1])  # [32, d_model//4]

        _, (shear_hidden, _) = self.shear_encoder(shear)
        shear_encoded = self.shear_projection(shear_hidden[-1])  # [32, d_model//4]

        # Combine encoded inputs
        combined = torch.cat([param_encoded, disp_encoded, shear_encoded], dim=-1)
        x = combined.unsqueeze(1)
        x = self.pos_encoder(x)

        # Process through attention branches
        x1 = x
        for layer in self.attention_branch1:
            x1 = layer(x1)
        x1 = self.final_norm(x1)
        x1 = x1.squeeze(1)

        x2 = x
        for layer in self.attention_branch2:
            x2 = layer(x2)
        x2 = self.final_norm(x2)
        x2 = x2.squeeze(1)

        # Generate predictions
        a1_pred = self.mlp_a1(x1)
        a2_pred = self.mlp_a2(x1)
        c1_pred = self.mlp_c1(x2)
        c2_pred = self.mlp_c2(x2)

        return a1_pred, c1_pred, a2_pred, c2_pred


# =====================================================================================================



# =====================================================================================================
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        # Pre-norm architecture instead of post-norm for better training stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Wider MLP with better activation
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),  # Increased width
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm architecture
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout1(attn_output)
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x


class CrackDetectionModel(nn.Module):
    def __init__(self, parameters_features, displacement_features, shear_features,
                 crack_length=168, d_model=512, dropout=0.1, num_attention_layers=3):
        super().__init__()

        # Deeper input encoders with residual connections
        self.param_encoder = nn.Sequential(
            nn.Linear(parameters_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # ResidualBlock(d_model // 2),
            ResidualBlock(d_model // 2)
        )

        self.displacement_encoder = nn.Sequential(
            nn.Linear(displacement_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            # ResidualBlock(d_model // 4),
            ResidualBlock(d_model // 4)
        )

        self.shear_encoder = nn.Sequential(
            nn.Linear(shear_features, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            # ResidualBlock(d_model // 4),
            ResidualBlock(d_model // 4)
        )

        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.crack_length = crack_length

        # Shared attention layers followed by specialized branches
        # self.shared_attention = nn.ModuleList([
        #     AttentionBlock(d_model, num_heads=8, dropout=dropout)
        #     for _ in range(num_attention_layers // 2)
        # ])

        # Specialized attention branches with different number of heads
        self.attention_branch1 = nn.ModuleList([
            AttentionBlock(d_model, num_heads=8, dropout=dropout)  # More heads for finer detail
            for _ in range(num_attention_layers)
        ])

        self.attention_branch2 = nn.ModuleList([
            AttentionBlock(d_model, num_heads=8, dropout=dropout)  # Fewer heads for global patterns
            for _ in range(num_attention_layers)
        ])

        self.final_norm1 = nn.LayerNorm(d_model)
        self.final_norm2 = nn.LayerNorm(d_model)

        # Deeper output MLPs with residual connections
        self.mlp_a1 = OutputMLP(d_model, dropout)
        self.mlp_a2 = OutputMLP(d_model, dropout)
        self.mlp_c1 = OutputMLP(d_model, dropout)
        self.mlp_c2 = OutputMLP(d_model, dropout)

    def forward(self, parameters, displacement, shear):
        # Encode inputs
        param_encoded = self.param_encoder(parameters)
        disp_encoded = self.displacement_encoder(displacement)
        shear_encoded = self.shear_encoder(shear)

        # Combine and fuse encoded inputs
        combined = torch.cat([param_encoded, disp_encoded, shear_encoded], dim=-1)
        combined = self.fusion_layer(combined)

        # Expand for sequence length
        combined = combined.unsqueeze(1).expand(-1, self.crack_length, -1)

        # Add positional encoding
        x = self.pos_encoder(combined)

        # Shared attention processing
        # for layer in self.shared_attention:
        #     x = layer(x)

        # Split into specialized branches
        x1, x2 = x, x

        # Branch 1 processing (finer details)
        for layer in self.attention_branch1:
            x1 = layer(x1)
        x1 = self.final_norm1(x1)

        # Branch 2 processing (global patterns)
        for layer in self.attention_branch2:
            x2 = layer(x2)
        x2 = self.final_norm2(x2)

        # Generate predictions
        a1_pred = self.mlp_a1(x1).squeeze(-1)
        a2_pred = self.mlp_a2(x1).squeeze(-1)
        c1_pred = self.mlp_c1(x2).squeeze(-1)
        c2_pred = self.mlp_c2(x2).squeeze(-1)

        return a1_pred, c1_pred, a2_pred, c2_pred


# Helper classes
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        return x + self.layers(x)


class OutputMLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(d_model),
            # ResidualBlock(d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        return self.net(x)
