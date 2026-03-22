"""Configuration for the Rosetta transformer."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RosettaConfig:
    """
    Configuration for the Rosetta genomic transformer.

    The architecture is designed around the thesis that DNA has an inherent
    "endianness" -- strand direction, reading frame, and codon position
    significance -- that must be modeled explicitly.
    """

    # --- Vocabulary & Embedding ---
    vocab_size: int = 7  # A=0, C=1, G=2, T=3, N=4, [CLS]=5, [MASK]=6
    d_model: int = 512
    max_seq_len: int = 8192  # nucleotides (not tokens -- character-level)

    # --- Multi-Frame Attention ---
    n_frames: int = 6          # 3 forward + 3 reverse complement reading frames
    n_heads: int = 8           # attention heads per frame-aware layer
    n_layers: int = 12         # total transformer layers
    n_frame_layers: int = 4    # layers with explicit multi-frame attention (first N)
    d_ff: int = 2048           # feedforward dimension
    dropout: float = 0.1

    # --- RC Equivariance ---
    rc_equivariant: bool = True  # enforce reverse-complement equivariance

    # --- Codon Position Encoding ---
    codon_position_dim: int = 32   # dimension for codon-position embeddings (pos 1, 2, 3)
    use_wobble_weighting: bool = True  # weight loss by codon position significance

    # --- Hierarchical Positional Encoding ---
    positional_encoding: Literal["hierarchical", "alibi", "sinusoidal"] = "hierarchical"
    n_position_scales: int = 4  # nucleotide, codon, gene-scale (~1kb), TAD-scale (~100kb)
    position_scale_factors: list[int] = field(default_factory=lambda: [1, 3, 1000, 100000])

    # --- Generation ---
    generative: bool = True  # include autoregressive generation head
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.95

    # --- Training ---
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_steps: int = 500000
    batch_size: int = 32
    gradient_accumulation_steps: int = 4

    # --- Wobble-Aware Loss ---
    # Information significance per codon position:
    # Position 1: ~2 bits (high significance)
    # Position 2: ~2 bits (high significance)
    # Position 3: ~0.5-1 bit (wobble -- lower significance for amino acid identity,
    #              but carries secondary regulatory info)
    wobble_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])  # fallback
    use_codon_weights: bool = True  # per-codon degeneracy weights (overrides wobble_weights)
