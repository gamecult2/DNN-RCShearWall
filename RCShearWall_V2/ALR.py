import torch
import torch.nn as nn
import math


class Massive175BModel(nn.Module):
    def __init__(self,
                 vocab_size=50257,  # Standard GPT tokenizer size
                 d_model=12288,  # Massive model dimension
                 num_heads=96,  # Extremely high number of attention heads
                 num_layers=96):  # Massive number of transformer layers
        super().__init__()

        # Store key dimensions
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(2048, d_model)

        # Attention projection layers
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        # Output and up/down projections
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * 4, bias=False)
        self.down_proj = nn.Linear(d_model * 4, d_model, bias=False)

        # Unembedding layer
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def count_component_parameters(self):
        # Detailed parameter counting function
        def param_count(layer):
            return sum(p.numel() for p in layer.parameters())

        print("\n--- Parameter Breakdown ---")
        print(f"Token Embedding: {param_count(self.token_embedding):,}")
        print(f"Position Embedding: {param_count(self.position_embedding):,}")

        print("\nAttention Projections:")
        print(f"Query Projection: {param_count(self.query_proj):,}")
        print(f"Key Projection: {param_count(self.key_proj):,}")
        print(f"Value Projection: {param_count(self.value_proj):,}")

        print("\nOther Projections:")
        print(f"Output Projection: {param_count(self.output_proj):,}")
        print(f"Up Projection: {param_count(self.up_proj):,}")
        print(f"Down Projection: {param_count(self.down_proj):,}")

        print("\nUnembedding Layer:")
        print(f"Unembed Projection: {param_count(self.unembed):,}")

        total = (
                param_count(self.token_embedding) +
                param_count(self.position_embedding) +
                param_count(self.query_proj) +
                param_count(self.key_proj) +
                param_count(self.value_proj) +
                param_count(self.output_proj) +
                param_count(self.up_proj) +
                param_count(self.down_proj) +
                param_count(self.unembed)
        )
        print(f"\nTotal Parameters: {total:,}")

        return total


# Instantiate the model
model = Massive175BModel()

# Count and print parameters
model.count_component_parameters()