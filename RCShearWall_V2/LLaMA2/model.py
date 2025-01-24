import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union


@dataclass
class ModelConfig:
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4  # Corrected to match n_heads
    vocab_size: int = -1  # Not used in LLaMA2_Model
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 512  # Not used in LLaMA2_Model
    device: str = 'cuda'  # Set device to 'cuda'
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, device: str) -> None:
        super().__init__()
        self.dim = head_dim
        assert self.dim % 2 == 0, "head_dim must be divisible by 2"

        theta_numerator = torch.arange(0, self.dim, 2, dtype=torch.float32)
        theta = 1.0 / torch.pow(10000, theta_numerator / self.dim).to(device)

        m = torch.arange(seq_len, dtype=torch.float32).to(device)
        freqs = torch.outer(m, theta).float()

        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_complex)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, seq_len, n_heads, dim = x.shape
        assert dim == self.dim, "dim must be equal to self.dim"

        x_complex = torch.view_as_complex(x.float().reshape(batch_size, seq_len, n_heads, -1, 2))
        freq_complex = self.freqs_cis[start_pos:start_pos + seq_len]
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)

        x_rotated = x_complex * freq_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch_size, seq_len, n_heads, dim)

        return x_out.type_as(x).to(x.device)



class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps  # Epsilon value for numerical stability
        self.gamma = nn.Parameter(torch.ones(dim))  # Learnable parameter for scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Input tensor of shape (Batch_Size, SeqLen, Dim)

        # Calculate the root-mean-square norm along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the root-mean-square norm and scale with gamma
        normalized_x = (x / rms) * self.gamma

        return normalized_x  # Return the normalized tensor


class SelfAttention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim = args.dim
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.Wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim)).to(args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim)).to(args.device)

        self.rope = RotaryPositionEmbedding(self.head_dim, args.max_seq_len, args.device)

    @staticmethod
    def repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (x[:, :, :, None, :]
                    .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                    .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
                    )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim, "dim must be equal to self.dim"

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = self.rope(xq, start_pos)
        xk = self.rope(xk, start_pos)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        keys = self.repeat_heads(keys, self.n_rep)
        values = self.repeat_heads(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        context = torch.matmul(scores, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.Wo(context)

        return output


class FeedForward(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.fc1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.fc3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.fc1(x))
        x_V = self.fc3(x)
        x = swish * x_V
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int):
        h = x + self.attention(self.norm1(x), start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))  # Corrected to use ffn_norm
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelConfig) -> None:
        super().__init__()

        # Check if vocab_size is specified
        assert args.vocab_size != -1, "vocab_size must be specified"

        # Store model configuration and necessary parameters
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Embedding layer for token embeddings
        self.embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Create a list of transformer encoder blocks
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # Layer normalization for the output
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # Output linear layer
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # Input shape: (Batch_Size, SeqLen)

        # Ensure seq_len is 1
        assert x.shape[1] == 1, "seq_len must be 1"

        # Embedding lookup
        x = self.embeddings(x)

        # Pass through each transformer encoder block
        for layer in self.layers:
            x = layer(x, start_pos)

        # Layer normalization
        x = self.norm(x)

        # Output prediction
        x = self.output(x)

        return x  # Return the output
