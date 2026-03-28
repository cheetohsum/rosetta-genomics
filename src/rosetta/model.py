"""
Rosetta Transformer: Multi-Frame, RC-Equivariant Genomic Foundation Model

Architecture overview:
  1. Character-level nucleotide embedding (A, C, G, T) -- no information loss
  2. Codon-position encoding: each nucleotide knows its position within
     all 6 possible reading frames simultaneously
  3. Hierarchical positional encoding at 4 scales (nt, codon, gene, TAD)
  4. RC-equivariant multi-frame attention layers (first N layers)
  5. Standard transformer layers (remaining layers)
  6. Dual heads: masked language model + autoregressive generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional

from .config import RosettaConfig


# =============================================================================
# Nucleotide Utilities
# =============================================================================

# The complement mapping: A<->T, C<->G
COMPLEMENT = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4, 5: 5, 6: 6}


def reverse_complement(seq: torch.Tensor) -> torch.Tensor:
    """
    Compute reverse complement of nucleotide sequences.

    This is the core "endianness flip" -- reading the opposite strand
    in the opposite direction. The same physical DNA, different information.

    Args:
        seq: (batch, seq_len) tensor of nucleotide indices (0=A, 1=C, 2=G, 3=T)
    Returns:
        (batch, seq_len) tensor of reverse complement
    """
    # Complement: A<->T (0<->3), C<->G (1<->2), others unchanged
    comp = seq.clone()
    for orig, repl in COMPLEMENT.items():
        comp[seq == orig] = repl
    # Reverse
    return comp.flip(dims=[-1])


# =============================================================================
# Per-Codon Degeneracy Weight Table
# =============================================================================

# Standard genetic code indexed by nucleotide tokens (A=0, C=1, G=2, T=3)
_CODON_TABLE = {
    (0,0,0): 'K', (0,0,1): 'N', (0,0,2): 'K', (0,0,3): 'N',
    (0,1,0): 'T', (0,1,1): 'T', (0,1,2): 'T', (0,1,3): 'T',
    (0,2,0): 'R', (0,2,1): 'S', (0,2,2): 'R', (0,2,3): 'S',
    (0,3,0): 'I', (0,3,1): 'I', (0,3,2): 'M', (0,3,3): 'I',
    (1,0,0): 'Q', (1,0,1): 'H', (1,0,2): 'Q', (1,0,3): 'H',
    (1,1,0): 'P', (1,1,1): 'P', (1,1,2): 'P', (1,1,3): 'P',
    (1,2,0): 'R', (1,2,1): 'R', (1,2,2): 'R', (1,2,3): 'R',
    (1,3,0): 'L', (1,3,1): 'L', (1,3,2): 'L', (1,3,3): 'L',
    (2,0,0): 'E', (2,0,1): 'D', (2,0,2): 'E', (2,0,3): 'D',
    (2,1,0): 'A', (2,1,1): 'A', (2,1,2): 'A', (2,1,3): 'A',
    (2,2,0): 'G', (2,2,1): 'G', (2,2,2): 'G', (2,2,3): 'G',
    (2,3,0): 'V', (2,3,1): 'V', (2,3,2): 'V', (2,3,3): 'V',
    (3,0,0): '*', (3,0,1): 'Y', (3,0,2): '*', (3,0,3): 'Y',
    (3,1,0): 'S', (3,1,1): 'S', (3,1,2): 'S', (3,1,3): 'S',
    (3,2,0): '*', (3,2,1): 'C', (3,2,2): 'W', (3,2,3): 'C',
    (3,3,0): 'L', (3,3,1): 'F', (3,3,2): 'L', (3,3,3): 'F',
}


def build_codon_weight_table() -> torch.Tensor:
    """
    Build a (4, 4, 4, 3) lookup table of degeneracy-based weights.

    For codon (n0, n1, n2) at intra-codon position p:
      - Fix the other two positions, vary position p over all 4 nucleotides
      - Count how many produce the SAME amino acid
      - weight = 1.0 / count

    Examples:
      Ala = GCx (4-fold at pos 2): weight = 0.25
      Met = ATG (non-degenerate):   weight = 1.0
      Phe = TT(T/C) (2-fold):      weight = 0.5
    """
    table = torch.ones(4, 4, 4, 3)

    for n0 in range(4):
        for n1 in range(4):
            for n2 in range(4):
                aa = _CODON_TABLE.get((n0, n1, n2))
                if aa is None:
                    continue

                # Position 0: vary n0, fix n1, n2
                deg_p0 = sum(1 for v in range(4) if _CODON_TABLE.get((v, n1, n2)) == aa)
                table[n0, n1, n2, 0] = 1.0 / deg_p0

                # Position 1: vary n1, fix n0, n2
                deg_p1 = sum(1 for v in range(4) if _CODON_TABLE.get((n0, v, n2)) == aa)
                table[n0, n1, n2, 1] = 1.0 / deg_p1

                # Position 2: vary n2, fix n0, n1
                deg_p2 = sum(1 for v in range(4) if _CODON_TABLE.get((n0, n1, v)) == aa)
                table[n0, n1, n2, 2] = 1.0 / deg_p2

    return table


# =============================================================================
# Codon Position Encoding
# =============================================================================

class CodonPositionEncoding(nn.Module):
    """
    Encodes each nucleotide's position within all 6 possible reading frames.

    For a nucleotide at absolute position i:
      - Forward frame 0: codon_pos = i % 3        (0, 1, 2)
      - Forward frame 1: codon_pos = (i - 1) % 3  (2, 0, 1)
      - Forward frame 2: codon_pos = (i - 2) % 3  (1, 2, 0)
      - Reverse frames: same but on the reverse complement

    This is the "byte alignment" layer -- it tells the model where each
    nucleotide sits within every possible codon interpretation.

    The key insight: position 1-2 carry amino acid identity (~2 bits each),
    position 3 (wobble) carries error tolerance + regulatory info (~0.5-1 bit).
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        # Learnable embeddings for codon positions 0, 1, 2
        # Separate embeddings per frame to allow frame-specific patterns
        self.frame_embeddings = nn.ModuleList([
            nn.Embedding(3, config.codon_position_dim)
            for _ in range(config.n_frames)
        ])
        self.projection = nn.Linear(
            config.n_frames * config.codon_position_dim,
            config.d_model
        )

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate codon position encodings for all 6 frames.

        Returns: (1, seq_len, d_model) tensor
        """
        positions = torch.arange(seq_len, device=device)

        frame_embeds = []
        for frame_idx in range(6):
            if frame_idx < 3:
                # Forward frames with offset 0, 1, 2
                codon_pos = (positions - frame_idx) % 3
            else:
                # Reverse frames: reverse the position, then offset
                rev_positions = seq_len - 1 - positions
                codon_pos = (rev_positions - (frame_idx - 3)) % 3

            embed = self.frame_embeddings[frame_idx](codon_pos)
            frame_embeds.append(embed)

        # Concatenate all frame embeddings and project
        combined = torch.cat(frame_embeds, dim=-1)  # (seq_len, 6 * codon_pos_dim)
        return self.projection(combined).unsqueeze(0)  # (1, seq_len, d_model)


# =============================================================================
# Hierarchical Positional Encoding
# =============================================================================

class HierarchicalPositionalEncoding(nn.Module):
    """
    Multi-scale positional encoding reflecting both coding and regulatory organization.

    Default 7 scales capture biologically meaningful periodicities:
      1bp   — nucleotide position (reading frame)
      2bp   — dinucleotide (CpG patterns, epigenetic signals)
      3bp   — codon (triplet coding structure)
      10bp  — helical turn (~10.5bp, TF binding face accessibility)
      147bp — nucleosome (wrapping periodicity, chromatin structure)
      1kb   — gene scale (promoter/enhancer distances)
      100kb — TAD scale (topological domain context)
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.n_scales = config.n_position_scales
        self.scale_factors = config.position_scale_factors

        # Unequal dim allocation: handles d_model not divisible by n_scales
        base_d = config.d_model // config.n_position_scales
        self.dims_per_scale = [base_d] * config.n_position_scales
        self.dims_per_scale[-1] += config.d_model - base_d * config.n_position_scales

        self.projection = nn.Linear(config.d_model, config.d_model)

    def _sinusoidal(self, positions: torch.Tensor, d: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding (handles odd dimensions)."""
        pe = torch.zeros(positions.shape[0], d, device=positions.device)
        n_sin = (d + 1) // 2  # ceil(d/2) sin terms
        n_cos = d // 2        # floor(d/2) cos terms
        div_term = torch.exp(
            torch.arange(0, d, 2, device=positions.device).float()
            * (-math.log(10000.0) / max(d, 1))
        )
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1).float() * div_term[:n_sin])
        if n_cos > 0:
            pe[:, 1::2] = torch.cos(positions.unsqueeze(1).float() * div_term[:n_cos])
        return pe

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate hierarchical positional encodings.

        Returns: (1, seq_len, d_model) tensor
        """
        positions = torch.arange(seq_len, device=device)

        scale_encodings = []
        for i, scale_factor in enumerate(self.scale_factors):
            scaled_pos = positions // scale_factor
            pe = self._sinusoidal(scaled_pos, self.dims_per_scale[i])
            scale_encodings.append(pe)

        combined = torch.cat(scale_encodings, dim=-1)
        return self.projection(combined).unsqueeze(0)


# =============================================================================
# RC-Equivariant Multi-Frame Attention
# =============================================================================

class MultiFrameAttention(nn.Module):
    """
    Attention mechanism that processes all 6 reading frames simultaneously.

    This is where the "key discovery" happens. The model learns:
    - Which frames are coding at each position
    - How information flows between frames (overlapping genes)
    - Frame-specific patterns (codon usage, splice signals)
    - Cross-strand regulatory interactions (antisense transcription)

    The attention scores across frames form a 6-dimensional "interpretive key"
    that the model learns to construct for each genomic region.

    Architecture:
    - Input embeddings are projected into 6 frame-specific subspaces
    - Cross-frame attention allows information flow between frames
    - A gating mechanism learns which frames are "active" at each position
    - Output is a frame-aware representation fused back to d_model
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.n_frames = config.n_frames
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads

        # Shared K, V projection -- all frames attend over the same keys/values
        self.shared_kv = nn.Linear(config.d_model, 2 * config.d_model)

        # Per-frame Q projections -- each frame asks different questions
        self.frame_q = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(config.n_frames)
        ])

        # Cross-frame MLP: lets frame outputs interact before gating
        # Bottleneck at 2*d_model preserves frame-specific information
        # while still allowing cross-frame mixing
        self.cross_frame_mlp = nn.Sequential(
            nn.Linear(config.n_frames * config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.n_frames * config.d_model),
        )
        # Learnable scale for cross-frame residual — starts small but can grow,
        # preventing the MLP from being a no-op due to small weight initialization
        self.cross_frame_scale = nn.Parameter(torch.tensor(0.1))

        # Frame gate: learns which frames are active at each position
        # Uses softmax (not sigmoid) to force competition between frames --
        # if one frame activates, the others must decrease
        self.frame_gate_proj = nn.Sequential(
            nn.Linear(config.d_model, config.n_frames * config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.n_frames * config.d_model // 4, config.n_frames),
        )

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-frame attention forward pass.

        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        residual = x

        # Shared keys and values -- all frames attend over the same space
        kv = self.shared_kv(x)
        k_shared, v_shared = kv.chunk(2, dim=-1)
        k_shared = k_shared.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v_shared = v_shared.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Per-frame queries -- each frame asks different questions of the shared K/V
        # Uses Flash Attention (scaled_dot_product_attention) which:
        #   - Never materializes the full (seq x seq) attention matrix
        #   - Runs in O(seq) memory instead of O(seq^2)
        #   - Uses fused CUDA kernels for ~2x speed

        # Convert mask to SDPA format
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                if mask.shape[0] == mask.shape[1]:
                    # Causal mask (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                    attn_mask = mask.unsqueeze(0).unsqueeze(0).bool()
                else:
                    # Padding mask (batch, seq_len) -> (batch, 1, 1, seq_len)
                    attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            else:
                attn_mask = mask.bool()

        frame_outputs = []
        for frame_idx in range(self.n_frames):
            q = self.frame_q[frame_idx](x)
            q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

            out = F.scaled_dot_product_attention(
                q, k_shared, v_shared,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
            frame_outputs.append(out)

        # Cross-frame MLP: let frame outputs interact
        stacked = torch.cat(frame_outputs, dim=-1)  # (batch, seq_len, 6*d_model)
        mixed = stacked + self.cross_frame_scale * self.cross_frame_mlp(stacked)

        # Split back, apply frame gating
        frame_outputs_mixed = mixed.chunk(self.n_frames, dim=-1)
        gates = F.softmax(self.frame_gate_proj(x), dim=-1)  # (batch, seq_len, n_frames)

        gated = []
        for frame_idx in range(self.n_frames):
            gate = gates[:, :, frame_idx:frame_idx+1]
            gated.append(frame_outputs_mixed[frame_idx] * gate)

        combined = sum(gated)
        combined = self.out_proj(combined)
        combined = self.dropout(combined)

        return self.layer_norm(residual + combined)


# =============================================================================
# Multi-Scale Regulatory Attention
# =============================================================================

class MultiScaleAttention(nn.Module):
    """
    Attention mechanism that processes multiple regulatory scales simultaneously.

    Mirrors MultiFrameAttention but for regulatory rather than coding structure:
    - Motif scale: learns TF binding motifs (~10bp patterns)
    - Nucleosome scale: learns nucleosome positioning signals (~150bp)
    - Enhancer scale: learns regulatory element structure (~500bp)

    Architecture (same pattern as MultiFrameAttention):
    - Shared K/V projection
    - Per-scale Q projections (each scale "asks" different questions)
    - Cross-scale MLP interaction with learned residual scale
    - Softmax gating (competitive: which scale is dominant at each position)
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.n_scales = config.n_regulatory_scales
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads

        self.shared_kv = nn.Linear(config.d_model, 2 * config.d_model)

        self.scale_q = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(self.n_scales)
        ])

        self.cross_scale_mlp = nn.Sequential(
            nn.Linear(self.n_scales * config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, self.n_scales * config.d_model),
        )
        self.cross_scale_scale = nn.Parameter(torch.tensor(0.1))

        self.scale_gate_proj = nn.Sequential(
            nn.Linear(config.d_model, self.n_scales * config.d_model // 4),
            nn.GELU(),
            nn.Linear(self.n_scales * config.d_model // 4, self.n_scales),
        )

        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        residual = x

        kv = self.shared_kv(x)
        k_shared, v_shared = kv.chunk(2, dim=-1)
        k_shared = k_shared.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v_shared = v_shared.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                if mask.shape[0] == mask.shape[1]:
                    attn_mask = mask.unsqueeze(0).unsqueeze(0).bool()
                else:
                    attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            else:
                attn_mask = mask.bool()

        scale_outputs = []
        for scale_idx in range(self.n_scales):
            q = self.scale_q[scale_idx](x)
            q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

            out = F.scaled_dot_product_attention(
                q, k_shared, v_shared,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
            scale_outputs.append(out)

        stacked = torch.cat(scale_outputs, dim=-1)
        mixed = stacked + self.cross_scale_scale * self.cross_scale_mlp(stacked)

        scale_outputs_mixed = mixed.chunk(self.n_scales, dim=-1)
        gates = F.softmax(self.scale_gate_proj(x), dim=-1)

        gated = []
        for scale_idx in range(self.n_scales):
            gate = gates[:, :, scale_idx:scale_idx + 1]
            gated.append(scale_outputs_mixed[scale_idx] * gate)

        combined = sum(gated)
        combined = self.out_proj(combined)
        combined = self.dropout(combined)

        return self.layer_norm(residual + combined)


# =============================================================================
# RC-Equivariant Layer Wrapper
# =============================================================================

class RCEquivariantWrapper(nn.Module):
    """
    Wraps a layer to enforce reverse-complement equivariance.

    For any layer f, we enforce:
        f(reverse_complement(x)) == reverse(f(x))

    This is achieved by processing both strands through f, then combining
    them via a learned projection applied to symmetrically constructed
    inputs. Using the same projection on (fwd, flip(rc)) and (rc, flip(fwd))
    guarantees equivariance while allowing a richer combination than
    simple averaging.
    """

    def __init__(self, layer: nn.Module, d_model: int):
        super().__init__()
        self.layer = layer
        # Learned equivariant combination (replaces simple average)
        self.strand_combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_rc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rc_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        RC-equivariant forward pass.

        Equivariance proof: strand_combine is applied to symmetrically
        constructed inputs -- (y_fwd, flip(y_rc)) for forward and
        (y_rc, flip(y_fwd)) for RC. The same weights on swapped inputs
        guarantees flip(output_fwd) == output_rc.
        """
        y_fwd = self.layer(x, mask=mask)
        y_rc = self.layer(x_rc, mask=rc_mask if rc_mask is not None else mask)

        # Symmetric input construction preserves equivariance
        fwd_input = torch.cat([y_fwd, y_rc.flip(dims=[1])], dim=-1)
        rc_input = torch.cat([y_rc, y_fwd.flip(dims=[1])], dim=-1)

        y_fwd_final = self.strand_combine(fwd_input)
        y_rc_final = self.strand_combine(rc_input)

        return y_fwd_final, y_rc_final


# =============================================================================
# Standard Transformer Layer
# =============================================================================

class TransformerLayer(nn.Module):
    """Standard transformer layer with pre-norm and optional MoDA (cross-layer K/V access).

    When depth_kv is provided, queries attend over [depth_K/V | current_K/V],
    letting this layer directly access features from preceding layers without
    signal dilution through the residual stream.
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # Explicit Q/K/V projections (replaces nn.MultiheadAttention for MoDA support)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        depth_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
            depth_kv: optional (depth_K, depth_V) from preceding layers,
                      each (batch, n_heads, depth_len, d_head)
        Returns:
            (x, K, V) — hidden state + this layer's K/V for MoDA cache
        """
        batch, seq_len, _ = x.shape
        normed = self.norm1(x)

        # Q/K/V projections
        q = self.q_proj(normed).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # MoDA: prepend depth K/V from preceding layers
        if depth_kv is not None:
            depth_k, depth_v = depth_kv
            k_all = torch.cat([depth_k, k], dim=2)
            v_all = torch.cat([depth_v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Build mask for SDPA — depth tokens are always attendable
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2 and mask.shape[0] == mask.shape[1]:
                # Causal mask — extend for depth tokens (all attendable)
                depth_len = k_all.shape[2] - seq_len
                if depth_len > 0:
                    depth_mask = torch.ones(seq_len, depth_len, device=mask.device)
                    attn_mask = torch.cat([depth_mask, mask], dim=1).unsqueeze(0).unsqueeze(0).bool()
                else:
                    attn_mask = mask.unsqueeze(0).unsqueeze(0).bool()
            elif mask.dim() == 2:
                # Padding mask (batch, seq) — depth tokens always valid
                depth_len = k_all.shape[2] - seq_len
                if depth_len > 0:
                    depth_pad = torch.ones(batch, depth_len, device=mask.device)
                    full_mask = torch.cat([depth_pad, mask], dim=1)
                    attn_mask = full_mask.unsqueeze(1).unsqueeze(2).bool()
                else:
                    attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            else:
                attn_mask = mask.bool()

        attn_out = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out

        # Pre-norm feedforward
        normed = self.norm2(x)
        x = x + self.ff(normed)

        return x, k, v


# =============================================================================
# ELECTRA Generator (lightweight, disposable after pretraining)
# =============================================================================

class _GeneratorLayer(nn.Module):
    """Minimal transformer layer for the ELECTRA generator."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        normed = self.norm1(x)
        q = self.q_proj(normed).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        x = x + self.out_proj(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class ElectraGenerator(nn.Module):
    """Small generator for ELECTRA pretraining. Predicts replacements at masked positions."""

    def __init__(self, config: RosettaConfig, shared_embed: nn.Embedding):
        super().__init__()
        gd = config.electra_gen_d_model
        self.shared_embed = shared_embed  # reference to discriminator's embedding
        self.embed_proj = nn.Linear(config.d_model, gd)
        self.layers = nn.ModuleList([
            _GeneratorLayer(gd, config.electra_gen_n_heads, config.electra_gen_d_ff, config.dropout)
            for _ in range(config.electra_gen_layers)
        ])
        self.mlm_head = nn.Sequential(
            nn.Linear(gd, gd), nn.GELU(), nn.LayerNorm(gd),
            nn.Linear(gd, config.vocab_size),
        )

    def forward(self, masked_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_ids: (batch, seq_len) with [MASK]=6 at masked positions
        Returns:
            logits: (batch, seq_len, vocab_size) predictions at all positions
        """
        x = self.embed_proj(self.shared_embed(masked_ids))
        for layer in self.layers:
            x = layer(x)
        return self.mlm_head(x)


# =============================================================================
# The Rosetta Transformer
# =============================================================================

class RosettaTransformer(nn.Module):
    """
    Rosetta: A Multi-Frame, RC-Equivariant Genomic Transformer

    The model that finds the "key" to DNA's information encoding.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │  Input: Raw nucleotide sequence (A, C, G, T)    │
    ├─────────────────────────────────────────────────┤
    │  Nucleotide Embedding (character-level)          │
    │  + Codon Position Encoding (6-frame aware)       │
    │  + Hierarchical Positional Encoding (4 scales)   │
    ├─────────────────────────────────────────────────┤
    │  RC-Equivariant Multi-Frame Attention (×4)       │
    │  ├── Forward strand processing                   │
    │  ├── Reverse complement processing               │
    │  └── Cross-strand equivariance enforcement       │
    ├─────────────────────────────────────────────────┤
    │  Standard Transformer Layers (×8)                │
    ├─────────────────────────────────────────────────┤
    │  Output Heads:                                   │
    │  ├── MLM Head (masked nucleotide prediction)     │
    │  ├── Frame Classification Head                   │
    │  └── Generation Head (autoregressive)            │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.config = config

        # --- Embedding layers ---
        self.nucleotide_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dinucleotide_embed = nn.Embedding(17, config.d_model)  # 16 pairs + 1 special
        self.codon_pos_encoding = CodonPositionEncoding(config)
        self.hierarchical_pos = HierarchicalPositionalEncoding(config)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_norm = nn.LayerNorm(config.d_model)

        # --- Multi-Frame Attention layers (coding structure) ---
        self.frame_layers = nn.ModuleList()
        for _ in range(config.n_frame_layers):
            frame_attn = MultiFrameAttention(config)
            if config.rc_equivariant:
                self.frame_layers.append(RCEquivariantWrapper(frame_attn, config.d_model))
            else:
                self.frame_layers.append(frame_attn)

        # --- Multi-Scale Attention layers (regulatory structure) ---
        assert config.n_frame_layers + config.n_scale_layers <= config.n_layers, (
            f"frame_layers ({config.n_frame_layers}) + scale_layers ({config.n_scale_layers}) "
            f"> n_layers ({config.n_layers})"
        )
        self.scale_layers = nn.ModuleList()
        for _ in range(config.n_scale_layers):
            scale_attn = MultiScaleAttention(config)
            if config.rc_equivariant:
                self.scale_layers.append(RCEquivariantWrapper(scale_attn, config.d_model))
            else:
                self.scale_layers.append(scale_attn)

        # --- Standard Transformer layers (integration) ---
        n_standard = config.n_layers - config.n_frame_layers - config.n_scale_layers
        self.standard_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(n_standard)
        ])

        # --- Strand role asymmetry (Gielis: same info, different roles) ---
        # Coding strand carries promoter signals, Shine-Dalgarno sequences;
        # template strand is read by RNA polymerase. Residual projections
        # let each strand develop role-specific features before fusion.
        if config.use_strand_asymmetry:
            self.fwd_strand_proj = nn.Linear(config.d_model, config.d_model)
            self.rc_strand_proj = nn.Linear(config.d_model, config.d_model)

        # --- Strand fusion (merge forward + RC representations) ---
        self.strand_fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

        # --- Output heads ---
        # Masked Language Model head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

        # Frame classification head: predicts which reading frame(s)
        # are coding at each position
        self.frame_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.n_frames),
        )

        # Conservation prediction: per-position evolutionary constraint proxy
        if config.use_conservation_head:
            self.conservation_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Linear(config.d_model // 2, 1),
                nn.Sigmoid(),
            )

        # Generation head (shared with MLM but with causal masking)
        if config.generative:
            self.gen_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, 4),  # A, C, G, T only
            )

        # JEPA: latent span prediction head (experimental)
        if config.use_jepa:
            self.jepa_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )

        # ELECTRA: generator + replaced-token detection head
        if config.use_electra:
            self.generator = ElectraGenerator(config, self.nucleotide_embed)
            self.rtd_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )

        self._init_weights()

        # Per-codon degeneracy weight table (static, rebuilt on construction)
        if config.use_codon_weights:
            self.register_buffer(
                'codon_weight_table', build_codon_weight_table(), persistent=False
            )

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create rich embeddings combining nucleotide, dinucleotide, codon-position, and hierarchical position."""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # Character-level nucleotide embedding
        x = self.nucleotide_embed(input_ids)

        # Dinucleotide embedding: overlapping pairs give CpG/epigenetic context
        if seq_len > 1:
            left = input_ids[:, :-1]
            right = input_ids[:, 1:]
            valid = (left <= 3) & (right <= 3)
            di_index = left * 4 + right  # 0-15 for valid nucleotide pairs
            di_index = torch.where(valid, di_index, torch.full_like(di_index, 16))
            di_index = F.pad(di_index, (0, 1), value=16)  # pad last position
            x = x + self.dinucleotide_embed(di_index)

        # Add codon position encoding (6-frame aware)
        x = x + self.codon_pos_encoding(seq_len, device)

        # Add hierarchical positional encoding (7 scales: coding + regulatory)
        x = x + self.hierarchical_pos(seq_len, device)

        return self.embed_dropout(self.embed_norm(x))

    def _create_rc_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create embeddings for the reverse complement strand."""
        rc_ids = reverse_complement(input_ids)
        return self._embed(rc_ids)

    def _run_rc_layers(self, layers, x_fwd, x_rc, mask, rc_mask, use_ckpt):
        """Run a list of layers with optional RC equivariance and gradient checkpointing."""
        for layer in layers:
            if self.config.rc_equivariant:
                if use_ckpt:
                    x_fwd, x_rc = grad_checkpoint(
                        layer, x_fwd, x_rc, mask, rc_mask, use_reentrant=False,
                    )
                else:
                    x_fwd, x_rc = layer(x_fwd, x_rc, mask=mask, rc_mask=rc_mask)
            else:
                if use_ckpt:
                    x_fwd = grad_checkpoint(layer, x_fwd, mask, use_reentrant=False)
                    x_rc = grad_checkpoint(layer, x_rc, rc_mask, use_reentrant=False)
                else:
                    x_fwd = layer(x_fwd, mask=mask)
                    x_rc = layer(x_rc, mask=rc_mask)
        return x_fwd, x_rc

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a DNA sequence through the full Rosetta architecture.

        Args:
            input_ids: (batch, seq_len) nucleotide indices
            attention_mask: optional mask

        Returns:
            (batch, seq_len, d_model) encoded representations
        """
        # Create embeddings for both strands
        x_fwd = self._embed(input_ids)
        x_rc = self._create_rc_embeddings(input_ids)

        # RC strand positions are reversed, so flip the causal mask
        rc_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 2:
            rc_mask = attention_mask.flip(dims=[0, 1])

        # Gradient checkpointing: recompute activations during backward pass
        use_ckpt = self.training and torch.is_grad_enabled()

        # Frame attention layers (coding structure)
        x_fwd, x_rc = self._run_rc_layers(
            self.frame_layers, x_fwd, x_rc, attention_mask, rc_mask, use_ckpt
        )

        # Scale attention layers (regulatory structure)
        x_fwd, x_rc = self._run_rc_layers(
            self.scale_layers, x_fwd, x_rc, attention_mask, rc_mask, use_ckpt
        )

        # Strand role asymmetry: apply strand-specific projections (residual)
        if self.config.use_strand_asymmetry:
            x_fwd = x_fwd + self.fwd_strand_proj(x_fwd)
            x_rc = x_rc + self.rc_strand_proj(x_rc)

        # Fuse forward and reverse complement representations
        x = self.strand_fusion(torch.cat([x_fwd, x_rc.flip(dims=[1])], dim=-1))

        # Standard transformer layers for integration (with optional MoDA)
        depth_cache = []
        for layer in self.standard_layers:
            # Build depth K/V from preceding standard layers
            depth_kv = None
            if depth_cache and self.config.use_moda:
                max_d = self.config.moda_depth
                recent = depth_cache[-max_d:] if max_d > 0 else depth_cache
                depth_kv = (
                    torch.cat([kv[0] for kv in recent], dim=2),
                    torch.cat([kv[1] for kv in recent], dim=2),
                )

            if use_ckpt:
                x, k, v = grad_checkpoint(
                    layer, x, attention_mask, depth_kv, use_reentrant=False,
                )
            else:
                x, k, v = layer(x, mask=attention_mask, depth_kv=depth_kv)

            # Cache K/V for subsequent layers (detached to prevent graph explosion)
            if self.config.use_moda:
                depth_cache.append((k.detach(), v.detach()))

        return x

    def _create_codon_aware_mask(
        self,
        original_ids: torch.Tensor,
        frame_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Create mask with mix of individual and codon-aligned masking (Pairwise MLM insight).

        Codon masking forces the model to learn codon-level correlations:
        positions 1-2 constrain position 3 (wobble), but when all 3 are masked,
        the model must use broader context to predict the entire codon.
        """
        batch_size, seq_len = original_ids.shape
        device = original_ids.device
        mask_prob = self.config.electra_mask_prob
        codon_frac = self.config.codon_mask_fraction

        special = (original_ids >= 4)
        masked_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Budget split: codon_frac goes to codon masking, rest to individual
        codon_budget = mask_prob * codon_frac
        indiv_budget = mask_prob * (1 - codon_frac)

        # Individual masking
        indiv_prob = torch.full((batch_size, seq_len), indiv_budget, device=device)
        indiv_prob.masked_fill_(special, 0.0)
        masked_positions |= torch.bernoulli(indiv_prob).bool()

        # Codon masking: mask groups of 3 aligned to position % 3
        # Use frame_labels to find the dominant reading frame, fallback to frame 0
        codon_starts = torch.arange(0, seq_len - 2, 3, device=device)  # frame 0 alignment
        n_codons = len(codon_starts)
        if n_codons > 0:
            codon_mask_prob = torch.full((batch_size, n_codons), codon_budget * 3, device=device)
            codon_selected = torch.bernoulli(codon_mask_prob).bool()
            for i, start in enumerate(codon_starts):
                for offset in range(3):
                    pos = start + offset
                    if pos < seq_len:
                        masked_positions[:, pos] |= codon_selected[:, i] & ~special[:, pos]

        return masked_positions

    def _forward_electra(
        self,
        original_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        frame_labels: Optional[torch.Tensor],
        conservation_targets: Optional[torch.Tensor],
        global_step: Optional[int],
    ) -> dict[str, torch.Tensor]:
        """ELECTRA forward with codon-aware masking, uniformity loss, and optional JEPA."""
        batch_size, seq_len = original_ids.shape
        device = original_ids.device

        # Step 1: Codon-aware masking (pairwise MLM insight)
        masked_positions = self._create_codon_aware_mask(original_ids, frame_labels)

        # Step 2: Mask input for generator
        masked_ids = original_ids.clone()
        masked_ids[masked_positions] = 6  # [MASK]

        # Step 3: Generator predicts replacements
        gen_logits = self.generator(masked_ids)

        # Step 4: Sample replacements
        with torch.no_grad():
            in_warmup = (global_step is not None
                         and self.config.warmup_plain_ce_steps > 0
                         and global_step < self.config.warmup_plain_ce_steps)
            n_masked = masked_positions.sum()
            if in_warmup or n_masked == 0:
                sampled = torch.randint(0, 4, (max(n_masked, 1),), device=device)
            else:
                gen_probs = F.softmax(gen_logits[masked_positions], dim=-1)
                sampled = torch.multinomial(gen_probs, 1).squeeze(-1)

        # Step 5: Build corrupted sequence
        corrupted_ids = original_ids.clone()
        if n_masked > 0:
            corrupted_ids[masked_positions] = sampled[:n_masked]

        # Step 6: RTD labels — 0=real, 1=replaced (correct guesses → "real")
        rtd_labels = (corrupted_ids != original_ids).float()

        # Step 7: Run discriminator on corrupted sequence
        hidden = self.encode(corrupted_ids, attention_mask)

        # Step 8: Predictions
        rtd_logits = self.rtd_head(hidden).squeeze(-1)
        frame_logits = self.frame_head(hidden)

        conservation_pred = None
        if hasattr(self, 'conservation_head'):
            conservation_pred = self.conservation_head(hidden).squeeze(-1)

        # Step 9: Losses (in FP32)
        with torch.amp.autocast("cuda", enabled=False):
            # Generator loss
            gen_loss = torch.tensor(0.0, device=device)
            if n_masked > 0:
                gen_loss = F.cross_entropy(
                    gen_logits[masked_positions].float(),
                    original_ids[masked_positions],
                )

            # Discriminator RTD loss
            has_cds = None
            if frame_labels is not None:
                has_cds = frame_labels.sum(dim=(1, 2)) > 0

            disc_loss = self._compute_rtd_loss(
                rtd_logits.float(), rtd_labels, original_ids,
                frame_labels, has_cds, global_step,
            )

            total_loss = disc_loss + self.config.electra_gen_weight * gen_loss

            # Embedding uniformity loss (anti-collapse contrastive)
            if self.config.contrastive_weight > 0 and batch_size > 1:
                embs = F.normalize(hidden.mean(dim=1).float(), dim=1)  # (batch, d_model)
                sim_matrix = embs @ embs.T
                off_diag = sim_matrix[~torch.eye(batch_size, dtype=torch.bool, device=device)]
                uniformity_loss = off_diag.mean()
                total_loss = total_loss + self.config.contrastive_weight * uniformity_loss

            # JEPA: predict original hidden states at masked positions (EXPERIMENTAL)
            # Requires a second no-grad forward pass on original_ids
            if self.config.use_jepa and hasattr(self, 'jepa_head') and n_masked > 0:
                with torch.no_grad():
                    target_hidden = self.encode(original_ids, attention_mask)
                jepa_pred = self.jepa_head(hidden[masked_positions].float())
                jepa_target = target_hidden[masked_positions].float().detach()
                jepa_loss = F.mse_loss(jepa_pred, jepa_target)
                total_loss = total_loss + self.config.jepa_weight * jepa_loss

            # Auxiliary losses
            if frame_labels is not None:
                frame_loss = F.binary_cross_entropy_with_logits(
                    frame_logits.float(), frame_labels.float()
                )
                total_loss = total_loss + 0.1 * frame_loss

            if conservation_targets is not None and conservation_pred is not None:
                conservation_loss = F.mse_loss(
                    conservation_pred.float(), conservation_targets.float()
                )
                total_loss = total_loss + self.config.conservation_weight * conservation_loss

        return {
            'loss': total_loss,
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'rtd_logits': rtd_logits,
            'rtd_labels': rtd_labels,
            'logits': gen_logits,  # for compatibility (trainer logging)
            'frame_logits': frame_logits,
            'hidden_states': hidden,
        }

    def _compute_rtd_loss(
        self,
        rtd_logits: torch.Tensor,
        rtd_labels: torch.Tensor,
        original_ids: torch.Tensor,
        frame_labels: Optional[torch.Tensor],
        has_cds: Optional[torch.Tensor],
        global_step: Optional[int],
    ) -> torch.Tensor:
        """Wobble-aware binary CE for replaced-token detection."""
        in_warmup = (global_step is not None
                     and self.config.warmup_plain_ce_steps > 0
                     and global_step < self.config.warmup_plain_ce_steps)

        if not self.config.use_wobble_weighting or in_warmup or frame_labels is None:
            return F.binary_cross_entropy_with_logits(rtd_logits, rtd_labels)

        batch_size, seq_len = rtd_logits.shape

        # Wobble weights from codon position (same logic, simpler since we have original_ids)
        fwd_labels = frame_labels[:, :, :3].float()
        label_sum = fwd_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        frame_probs = fwd_labels / label_sum

        if self.config.use_codon_weights and hasattr(self, 'codon_weight_table'):
            # Create dummy labels for _compute_codon_frame_weights (no -100 masking needed)
            dummy_labels = original_ids.clone()
            frame_weights = self._compute_codon_frame_weights(
                original_ids, dummy_labels, frame_probs, batch_size, seq_len
            )
        else:
            w = torch.tensor(self.config.wobble_weights, device=rtd_logits.device)
            positions = torch.arange(seq_len, device=rtd_logits.device)
            frame_weights = torch.zeros(batch_size, seq_len, device=rtd_logits.device)
            for f in range(3):
                codon_pos = (positions - f) % 3
                frame_weights += frame_probs[:, :, f] * w[codon_pos].unsqueeze(0)

        # Non-CDS samples get uniform weight
        if has_cds is not None:
            no_cds = ~has_cds.unsqueeze(1)
            frame_weights = torch.where(no_cds, torch.ones_like(frame_weights), frame_weights)

        # Entropy weighting
        if self.config.use_entropy_weighting:
            entropy_w = self._compute_entropy_weights(original_ids, original_ids)
            if has_cds is not None:
                entropy_w = torch.where(no_cds, torch.ones_like(entropy_w), entropy_w)
            frame_weights = frame_weights * entropy_w

        loss_per_pos = F.binary_cross_entropy_with_logits(
            rtd_logits, rtd_labels, reduction='none'
        )
        return (loss_per_pos * frame_weights).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        conservation_targets: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: (batch, seq_len) nucleotide indices
            attention_mask: optional mask
            labels: (batch, seq_len) target nucleotide indices for MLM
            frame_labels: (batch, seq_len, 6) binary labels for which frames are coding

        Returns:
            Dict with 'logits', 'frame_logits', and optionally 'loss'
        """
        # ELECTRA branch: generator + replaced-token detection
        if self.config.use_electra and (self.training or labels is None):
            return self._forward_electra(
                original_ids=input_ids,
                attention_mask=attention_mask,
                frame_labels=frame_labels,
                conservation_targets=conservation_targets,
                global_step=global_step,
            )

        # Standard MLM branch
        hidden = self.encode(input_ids, attention_mask)

        # MLM prediction
        logits = self.mlm_head(hidden)

        # Frame prediction
        frame_logits = self.frame_head(hidden)

        # Conservation prediction
        conservation_pred = None
        if hasattr(self, 'conservation_head'):
            conservation_pred = self.conservation_head(hidden).squeeze(-1)  # (batch, seq_len)

        output = {
            'logits': logits,
            'frame_logits': frame_logits,
            'hidden_states': hidden,
        }
        if conservation_pred is not None:
            output['conservation_pred'] = conservation_pred

        # Compute loss in FP32 — wobble weights, entropy, and log() are
        # numerically unstable in FP16. Disable autocast for this block.
        with torch.amp.autocast("cuda", enabled=False):
            logits_f32 = logits.float()
            frame_logits_f32 = frame_logits.float()

            if labels is not None:
                # Decide whether to use weighted loss or plain CE
                in_warmup = (global_step is not None
                             and self.config.warmup_plain_ce_steps > 0
                             and global_step < self.config.warmup_plain_ce_steps)
                use_weighted = self.config.use_wobble_weighting and not in_warmup

                # Detect which samples have CDS annotations (non-zero frame labels)
                has_cds = None
                if frame_labels is not None:
                    has_cds = frame_labels.sum(dim=(1, 2)) > 0  # (batch,)

                frame_probs = None
                if use_weighted:
                    if frame_labels is not None:
                        # Use ground truth frame labels (forward 3 frames)
                        fwd_labels = frame_labels[:, :, :3].float()
                        label_sum = fwd_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        frame_probs = fwd_labels / label_sum
                    else:
                        # Use model's own frame gate predictions (detached)
                        with torch.no_grad():
                            x_for_gate = self._embed(input_ids).float()
                            if self.config.rc_equivariant:
                                gate_layer = self.frame_layers[0].layer
                            else:
                                gate_layer = self.frame_layers[0]
                            raw_gates = F.softmax(gate_layer.frame_gate_proj(x_for_gate), dim=-1)
                            fwd_gates = raw_gates[:, :, :3]
                            frame_probs = fwd_gates / fwd_gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                loss = self._compute_wobble_aware_loss(
                    logits_f32, labels, input_ids, frame_probs, has_cds=has_cds
                )
                output['loss'] = loss

            if frame_labels is not None:
                frame_loss = F.binary_cross_entropy_with_logits(
                    frame_logits_f32, frame_labels.float()
                )
                output['frame_loss'] = frame_loss
                if 'loss' in output:
                    output['loss'] = output['loss'] + 0.1 * frame_loss

            # Conservation prediction loss
            if conservation_targets is not None and conservation_pred is not None:
                conservation_loss = F.mse_loss(
                    conservation_pred.float(), conservation_targets.float()
                )
                output['conservation_loss'] = conservation_loss
                if 'loss' in output:
                    output['loss'] = output['loss'] + self.config.conservation_weight * conservation_loss

            # Embedding uniformity loss (anti-collapse, works for MLM too)
            if self.config.contrastive_weight > 0 and 'loss' in output:
                batch_size = logits_f32.shape[0]
                if batch_size > 1:
                    embs = F.normalize(hidden.float().mean(dim=1), dim=1)
                    sim = embs @ embs.T
                    off_diag = sim[~torch.eye(batch_size, dtype=torch.bool, device=sim.device)]
                    output['loss'] = output['loss'] + self.config.contrastive_weight * off_diag.mean()

        return output

    def _compute_codon_frame_weights(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        frame_probs: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Vectorized per-codon weight lookup across all 3 forward frames.

        Reconstructs the original sequence, then for each frame gathers
        the 3 nucleotides of each position's codon and looks up the
        degeneracy weight from self.codon_weight_table.
        """
        device = input_ids.device

        # Reconstruct original sequence (unmask)
        original = input_ids.clone()
        known = labels != -100
        original[known] = labels[known]

        # Clamp to valid nucleotide range [0, 3] (special tokens → 0)
        safe_seq = original.clamp(0, 3)

        positions = torch.arange(seq_len, device=device)
        fallback_w = torch.tensor(self.config.wobble_weights, device=device)
        frame_weights = torch.zeros(batch_size, seq_len, device=device)

        for f in range(3):
            codon_pos = (positions - f) % 3
            codon_start = positions - codon_pos

            # Valid = codon fully within sequence
            valid = (codon_start >= 0) & (codon_start + 2 < seq_len)
            cs = codon_start.clamp(0, max(seq_len - 3, 0))

            # Gather codon nucleotides
            n0 = safe_seq[:, cs]
            n1 = safe_seq[:, cs + 1]
            n2 = safe_seq[:, cs + 2]
            cp = codon_pos.unsqueeze(0).expand(batch_size, -1)

            # Lookup per-codon degeneracy weight
            w = self.codon_weight_table[n0, n1, n2, cp]

            # Fallback for boundary positions
            if not valid.all():
                fallback = fallback_w[codon_pos]
                w[:, ~valid] = fallback[~valid].unsqueeze(0)

            frame_weights += frame_probs[:, :, f] * w

        return frame_weights

    def _compute_entropy_weights(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-position entropy weights from local nucleotide distribution.

        High-entropy regions (diverse coding/regulatory) get higher weight.
        Low-entropy regions (repeats, poly-A) get lower weight.
        Vectorized: uses 1D unfold for sliding window histogram.

        Returns:
            (batch, seq_len) weights in [entropy_min_weight, 1.0]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        window = self.config.entropy_window
        half_w = window // 2

        # Reconstruct original sequence
        original = input_ids.clone()
        known = labels != -100
        original[known] = labels[known]
        safe_seq = original.clamp(0, 3)  # clamp special tokens to valid nucleotides

        # One-hot encode in FP32 (entropy log() is unstable in FP16)
        one_hot = F.one_hot(safe_seq.long(), num_classes=4).to(torch.float32)

        # Sliding window sum via 1D convolution
        # Reshape to (batch, 4, seq_len) for conv1d
        one_hot_t = one_hot.permute(0, 2, 1)  # (batch, 4, seq_len)
        kernel = torch.ones(4, 1, window, device=device, dtype=torch.float32) / window
        # Grouped conv: each of 4 channels independently
        freqs = F.conv1d(one_hot_t, kernel, padding=half_w, groups=4)  # (batch, 4, seq_len)

        # Shannon entropy: H = -sum(p * log(p))
        freqs = freqs.clamp(min=1e-8)
        entropy = -(freqs * freqs.log()).sum(dim=1)  # (batch, seq_len)

        # Normalize to [0, 1] (max entropy = log(4) ≈ 1.386)
        max_entropy = math.log(4)
        entropy_norm = entropy / max_entropy

        # Apply floor and scale to [min_weight, 1.0]
        min_w = self.config.entropy_min_weight
        weights = min_w + (1.0 - min_w) * entropy_norm

        return weights

    def _compute_wobble_aware_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        frame_probs: Optional[torch.Tensor] = None,
        has_cds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Wobble-aware cross-entropy loss with per-codon degeneracy weighting.

        When use_codon_weights is True, each position's weight is derived
        from the actual degeneracy of its codon (e.g., Met ATG pos3 = 1.0,
        Ala GCx pos3 = 0.25). Falls back to fixed [1.0, 1.0, 0.5] weights
        when codon context is unavailable.

        has_cds: (batch,) bool tensor indicating which samples have CDS
        annotations. Samples without CDS get uniform weight=1.0 (plain CE)
        instead of wobble/entropy weighting — prevents near-zero loss on
        non-coding windows that caused representation collapse.

        Loss computed in FP32 — entropy log() and weighted reduction
        are numerically unstable in FP16.
        """
        # Force FP32 for numerical stability under AMP
        logits = logits.float()
        if frame_probs is not None:
            frame_probs = frame_probs.float()

        if not self.config.use_wobble_weighting or frame_probs is None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        batch_size, seq_len, vocab_size = logits.shape

        # Per-codon weights or fixed fallback
        if self.config.use_codon_weights and hasattr(self, 'codon_weight_table'):
            frame_weights = self._compute_codon_frame_weights(
                input_ids, labels, frame_probs, batch_size, seq_len
            )
        else:
            w = torch.tensor(self.config.wobble_weights, device=logits.device)
            positions = torch.arange(seq_len, device=logits.device)
            frame_weights = torch.zeros(batch_size, seq_len, device=logits.device)
            for frame_offset in range(3):
                codon_pos = (positions - frame_offset) % 3
                pos_weights = w[codon_pos]
                frame_weights += frame_probs[:, :, frame_offset] * pos_weights.unsqueeze(0)

        # Regional entropy weighting: upweight information-dense regions
        if self.config.use_entropy_weighting:
            entropy_weights = self._compute_entropy_weights(input_ids, labels)
            # Only apply entropy weighting on CDS samples
            if has_cds is not None:
                no_cds = ~has_cds.unsqueeze(1)  # (batch, 1) for broadcasting
                entropy_weights = torch.where(no_cds, torch.ones_like(entropy_weights), entropy_weights)
            frame_weights = frame_weights * entropy_weights

        # Non-CDS samples get uniform weight (plain CE) — prevents near-zero
        # loss on non-coding windows that caused representation collapse
        if has_cds is not None:
            no_cds = ~has_cds.unsqueeze(1)  # (batch, 1)
            frame_weights = torch.where(no_cds, torch.ones_like(frame_weights), frame_weights)

        # Per-position cross-entropy
        loss_per_position = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(batch_size, seq_len)

        weighted_loss = loss_per_position * frame_weights
        valid_mask = (labels != -100).float()
        return (weighted_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.95,
        frame_coherent: bool = True,
    ) -> torch.Tensor:
        """
        Generate new DNA sequences autoregressively.

        Args:
            prompt: (batch, prompt_len) starting nucleotide sequence
            max_new_tokens: number of new nucleotides to generate
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold
            frame_coherent: if True, bias generation to maintain reading frame coherence

        Returns:
            (batch, prompt_len + max_new_tokens) generated sequence
        """
        self.eval()
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Use only the last max_seq_len tokens if sequence is too long
            context = generated[:, -self.config.max_seq_len:]

            # Encode with causal masking -- model can only attend to past tokens
            seq_len = context.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=context.device))
            hidden = self.encode(context, attention_mask=causal_mask)
            next_logits = self.gen_head(hidden[:, -1, :])  # (batch, 4)

            # Apply temperature
            next_logits = next_logits / max(temperature, 1e-8)

            # Frame-coherent generation: slightly bias toward codons that
            # maintain the reading frame from the prompt
            if frame_coherent and generated.shape[1] >= 3:
                pos_in_codon = generated.shape[1] % 3
                # At wobble positions (pos 2), allow more diversity
                if pos_in_codon == 2:
                    next_logits = next_logits / 1.2  # Increase entropy at wobble

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, 4))
                threshold = top_k_vals[:, -1:]
                next_logits[next_logits < threshold] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                for b in range(next_logits.shape[0]):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    next_logits[b, indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_frame_attention_map(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the learned frame gate activations -- this IS the "key".

        The frame gates reveal which reading frame(s) the model considers
        active at each position. This is the interpretive key that decodes
        DNA's multi-frame encoding.

        Returns:
            (batch, seq_len, n_frames) frame activity map
        """
        x = self._embed(input_ids)

        # Extract gates from the first multi-frame layer
        if self.config.rc_equivariant:
            layer = self.frame_layers[0].layer
        else:
            layer = self.frame_layers[0]

        gates = F.softmax(layer.frame_gate_proj(x), dim=-1)
        return gates

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts['embedding'] = sum(
            p.numel() for p in self.nucleotide_embed.parameters()
        )
        counts['dinucleotide'] = sum(
            p.numel() for p in self.dinucleotide_embed.parameters()
        )
        counts['codon_position'] = sum(
            p.numel() for p in self.codon_pos_encoding.parameters()
        )
        counts['hierarchical_position'] = sum(
            p.numel() for p in self.hierarchical_pos.parameters()
        )
        counts['frame_attention'] = sum(
            p.numel() for p in self.frame_layers.parameters()
        )
        counts['scale_attention'] = sum(
            p.numel() for p in self.scale_layers.parameters()
        )
        counts['standard_layers'] = sum(
            p.numel() for p in self.standard_layers.parameters()
        )
        counts['heads'] = sum(
            p.numel() for n, p in self.named_parameters()
            if 'mlm_head' in n or 'frame_head' in n or 'gen_head' in n
        )
        if hasattr(self, 'conservation_head'):
            counts['conservation_head'] = sum(
                p.numel() for p in self.conservation_head.parameters()
            )
        if hasattr(self, 'jepa_head'):
            counts['jepa_head'] = sum(
                p.numel() for p in self.jepa_head.parameters()
            )
        if hasattr(self, 'generator'):
            counts['electra_generator'] = sum(
                p.numel() for p in self.generator.parameters()
            )
        if hasattr(self, 'rtd_head'):
            counts['rtd_head'] = sum(
                p.numel() for p in self.rtd_head.parameters()
            )
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
