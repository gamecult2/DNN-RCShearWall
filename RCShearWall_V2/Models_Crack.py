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
        self.final_linear = nn.Linear(input_dim, output_dim)

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
    def __init__(self,
                 parameters_features,
                 displacement_features,
                 shear_features,
                 crack_length,
                 sequence_length,
                 num_horizontal=14,
                 num_vertical=12,
                 d_model=256,
                 dropout=0.1,
                 num_attention_layers=4):
        super().__init__()

        self.param_encoder = nn.Linear(parameters_features, d_model//2)
        self.disp_encoder = nn.Linear(displacement_features, d_model//4)
        self.shear_encoder = nn.Linear(shear_features, d_model//4)

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
# =====================================================================================================
