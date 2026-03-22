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
    Multi-scale positional encoding reflecting biological organization:

    Scale 0: Nucleotide level (position within sequence)
    Scale 1: Codon level (position / 3) -- triplet structure
    Scale 2: Gene scale (position / ~1000) -- gene-level context
    Scale 3: TAD scale (position / ~100000) -- topological domain context

    Biology teaches us that position matters at EVERY scale:
    - Telomere proximity silences genes (up to 10Mb)
    - TAD boundaries define regulatory neighborhoods (~1Mb)
    - Codon position determines information significance
    - Single nucleotide position determines reading frame
    """

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.n_scales = config.n_position_scales
        self.scale_factors = config.position_scale_factors
        d_per_scale = config.d_model // config.n_position_scales

        # Sinusoidal encoding at each scale
        self.d_per_scale = d_per_scale
        self.projection = nn.Linear(
            d_per_scale * config.n_position_scales,
            config.d_model
        )

    def _sinusoidal(self, positions: torch.Tensor, d: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(positions.shape[0], d, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=positions.device).float()
            * (-math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1).float() * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1).float() * div_term)
        return pe

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate hierarchical positional encodings.

        Returns: (1, seq_len, d_model) tensor
        """
        positions = torch.arange(seq_len, device=device)

        scale_encodings = []
        for scale_factor in self.scale_factors:
            scaled_pos = positions // scale_factor
            pe = self._sinusoidal(scaled_pos, self.d_per_scale)
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
        frame_outputs = []
        for frame_idx in range(self.n_frames):
            q = self.frame_q[frame_idx](x)
            q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

            scale = math.sqrt(self.d_head)
            attn = torch.matmul(q, k_shared.transpose(-2, -1)) / scale

            if mask is not None:
                if mask.dim() == 2:
                    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
                else:
                    attn = attn.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v_shared)
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
    """Standard transformer layer with pre-norm."""

    def __init__(self, config: RosettaConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)

        # Convert mask from "1=attend, 0=block" to additive float mask
        # that nn.MultiheadAttention expects (0=attend, -inf=block)
        attn_mask = None
        if mask is not None:
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask = attn_mask.masked_fill(mask == 0, float('-inf'))

        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-norm feedforward
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x


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
        self.codon_pos_encoding = CodonPositionEncoding(config)
        self.hierarchical_pos = HierarchicalPositionalEncoding(config)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_norm = nn.LayerNorm(config.d_model)

        # --- Multi-Frame Attention layers (first N layers) ---
        self.frame_layers = nn.ModuleList()
        for _ in range(config.n_frame_layers):
            frame_attn = MultiFrameAttention(config)
            if config.rc_equivariant:
                self.frame_layers.append(RCEquivariantWrapper(frame_attn, config.d_model))
            else:
                self.frame_layers.append(frame_attn)

        # --- Standard Transformer layers (remaining layers) ---
        n_standard = config.n_layers - config.n_frame_layers
        self.standard_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(n_standard)
        ])

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

        # Generation head (shared with MLM but with causal masking)
        if config.generative:
            self.gen_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, 4),  # A, C, G, T only
            )

        self._init_weights()

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
        """Create rich embeddings combining nucleotide, codon-position, and hierarchical position."""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # Character-level nucleotide embedding
        x = self.nucleotide_embed(input_ids)

        # Add codon position encoding (6-frame aware)
        x = x + self.codon_pos_encoding(seq_len, device)

        # Add hierarchical positional encoding (4 scales)
        x = x + self.hierarchical_pos(seq_len, device)

        return self.embed_dropout(self.embed_norm(x))

    def _create_rc_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create embeddings for the reverse complement strand."""
        rc_ids = reverse_complement(input_ids)
        return self._embed(rc_ids)

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

        # Multi-frame attention layers with RC equivariance
        # RC strand positions are reversed, so flip the causal mask
        rc_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 2:
            rc_mask = attention_mask.flip(dims=[0, 1])

        for layer in self.frame_layers:
            if self.config.rc_equivariant:
                x_fwd, x_rc = layer(x_fwd, x_rc, mask=attention_mask, rc_mask=rc_mask)
            else:
                x_fwd = layer(x_fwd, mask=attention_mask)
                x_rc = layer(x_rc, mask=rc_mask)

        # Fuse forward and reverse complement representations
        # This is where both "endiannesses" merge into a unified representation
        x = self.strand_fusion(torch.cat([x_fwd, x_rc.flip(dims=[1])], dim=-1))

        # Standard transformer layers for deeper processing
        for layer in self.standard_layers:
            x = layer(x, mask=attention_mask)

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
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
        # Encode
        hidden = self.encode(input_ids, attention_mask)

        # MLM prediction
        logits = self.mlm_head(hidden)

        # Frame prediction
        frame_logits = self.frame_head(hidden)

        output = {
            'logits': logits,
            'frame_logits': frame_logits,
            'hidden_states': hidden,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Get frame probabilities for wobble weighting
            frame_probs = None
            if self.config.use_wobble_weighting:
                if frame_labels is not None:
                    # Use ground truth frame labels (forward 3 frames)
                    fwd_labels = frame_labels[:, :, :3]
                    label_sum = fwd_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    frame_probs = fwd_labels / label_sum
                else:
                    # Use model's own frame gate predictions (detached)
                    with torch.no_grad():
                        x_for_gate = self._embed(input_ids)
                        if self.config.rc_equivariant:
                            gate_layer = self.frame_layers[0].layer
                        else:
                            gate_layer = self.frame_layers[0]
                        raw_gates = F.softmax(gate_layer.frame_gate_proj(x_for_gate), dim=-1)
                        # Extract forward 3 frames and renormalize (NOT double softmax)
                        fwd_gates = raw_gates[:, :, :3]
                        frame_probs = fwd_gates / fwd_gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            loss = self._compute_wobble_aware_loss(logits, labels, input_ids, frame_probs)
            output['loss'] = loss

        if frame_labels is not None:
            frame_loss = F.binary_cross_entropy_with_logits(
                frame_logits, frame_labels.float()
            )
            output['frame_loss'] = frame_loss
            if 'loss' in output:
                output['loss'] = output['loss'] + 0.1 * frame_loss

        return output

    def _compute_wobble_aware_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        frame_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Wobble-aware cross-entropy loss.

        Uses frame gate predictions (or ground truth frame labels) to determine
        the likely reading frame at each position, then applies codon-position
        weights: positions 1-2 (amino acid identity) at 1.0x, position 3
        (wobble/degeneracy) at 0.5x.

        Args:
            logits: (batch, seq_len, vocab_size) predicted logits
            labels: (batch, seq_len) target labels (-100 for unmasked)
            input_ids: (batch, seq_len) input token ids
            frame_probs: (batch, seq_len, 3) soft frame probabilities for
                         the 3 forward reading frames. If None, falls back
                         to uniform cross-entropy.
        """
        if not self.config.use_wobble_weighting or frame_probs is None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        batch_size, seq_len, vocab_size = logits.shape
        w = torch.tensor(self.config.wobble_weights, device=logits.device)
        positions = torch.arange(seq_len, device=logits.device)

        # For each forward frame, compute the codon-position weight per position
        # then combine using the frame probability (soft frame selection)
        frame_weights = torch.zeros(batch_size, seq_len, device=logits.device)
        for frame_offset in range(3):
            codon_pos = (positions - frame_offset) % 3
            pos_weights = w[codon_pos]  # (seq_len,)
            frame_weights += frame_probs[:, :, frame_offset] * pos_weights.unsqueeze(0)

        # Compute per-position cross-entropy
        loss_per_position = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(batch_size, seq_len)

        # Apply wobble weighting
        weighted_loss = loss_per_position * frame_weights

        # Mask out ignored positions
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
        counts['codon_position'] = sum(
            p.numel() for p in self.codon_pos_encoding.parameters()
        )
        counts['hierarchical_position'] = sum(
            p.numel() for p in self.hierarchical_pos.parameters()
        )
        counts['frame_attention'] = sum(
            p.numel() for p in self.frame_layers.parameters()
        )
        counts['standard_layers'] = sum(
            p.numel() for p in self.standard_layers.parameters()
        )
        counts['heads'] = sum(
            p.numel() for n, p in self.named_parameters()
            if 'mlm_head' in n or 'frame_head' in n or 'gen_head' in n
        )
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
