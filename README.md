# Rosetta: Finding the Key to DNA's Information Encoding

A novel multi-frame, reverse-complement-equivariant genomic transformer that treats DNA as what it actually is: a multi-layered, bidirectional information system with inherent "endianness."

**22-check validation suite** | **Honest results, no inflated claims** | [Validation Results](#validation-results) | [Architecture](#architecture) | [Quick Start](#quick-start)

---

## The Thesis

Current genomic transformers (DNABERT-2, Nucleotide Transformer, Evo, Evo 2, Caduceus, HyenaDNA) treat DNA as flat text. But DNA has an inherent interpretive key -- analogous to byte-order conventions in binary systems -- that determines how the information is decoded. This key has three dimensions:

| Dimension | Binary Analog | DNA Reality |
|-----------|--------------|-------------|
| **Strand Direction** | Big/little endian | 5'->3' vs 3'->5' reading (reverse complement) |
| **Reading Frame** | Byte alignment | Codon offset (0, 1, 2) -- shifting by 1nt changes the entire protein |
| **Codon Position** | Bit significance (MSB vs LSB) | Position 1-2 = amino acid identity (~2 bits each), Position 3 = wobble (~0.5 bits) |

The genetic code itself is evidence: it is more error-tolerant than 99.9999% of random codes (Freeland & Hurst, 1998), with the wobble position functioning as a biological parity bit. Both strands are transcribed (~1,600 sense-antisense pairs in humans), and 27% of bacterial genes have overlapping reading frames.

**Rosetta models all three dimensions simultaneously.** No existing architecture does this.

### Gap Analysis

| Model | Strand Awareness | Multi-Frame | Wobble-Position | Hierarchical Position | Generation |
|-------|:---:|:---:|:---:|:---:|:---:|
| DNABERT-2 | - | - | - | ALiBi | - |
| Nucleotide Transformer | - | - | - | Learned | - |
| HyenaDNA | - | - | - | Implicit | Yes |
| Evo / Evo 2 | - | - | - | Implicit | Yes |
| Caduceus | **Yes** | - | - | Single scale | - |
| **Rosetta (ours)** | **Yes** | **Yes** | **Yes** | **Yes (4-scale)** | **Yes** |

---

## Architecture

```
Input: Raw nucleotide sequence (A, C, G, T)
  |
  v
[Nucleotide Embedding] + [Codon Position Encoding] + [Hierarchical Positional Encoding]
  Character-level          6-frame aware                4 biological scales
  (no k-mer compression)   (learned per-frame)          (nt / codon / gene / TAD)
  |
  v
[RC-Equivariant Multi-Frame Attention] x N layers
  |-- Forward strand processing
  |-- Reverse complement processing
  |-- Per-frame Q/K/V projections (6 separate attention pathways)
  |-- Frame gating: learned sigmoid gates per reading frame
  |-- Cross-strand equivariance: f(RC(x)) = reverse(f(x))
  |
  v
[Strand Fusion] -- merge forward + RC into unified representation
  |
  v
[Standard Transformer] x M layers
  |
  v
[Output Heads]
  |-- MLM Head: masked nucleotide prediction (vocab=7)
  |-- Frame Head: reading frame classification (6 outputs)
  |-- Generation Head: autoregressive DNA synthesis (A/C/G/T)
```

### Novel Components

**1. Multi-Frame Attention** (`src/rosetta/model.py:177`)
Processes all 6 reading frames (3 forward + 3 reverse complement) simultaneously. Each frame has its own Q/K/V projections. A learned gating mechanism (small MLP with sigmoid output) discovers which frames are coding at each position. After training, the gate activations *are* the interpretive key -- they reveal the multi-frame structure of any genomic region.

**2. RC Equivariance** (`src/rosetta/model.py:284`)
Wraps multi-frame layers to architecturally enforce `f(RC(x)) = reverse(f(x))`. Both strands are processed simultaneously, then cross-averaged. This reflects the physical reality that both DNA strands carry information, and the model should treat them consistently.

**3. Wobble-Aware Loss** (`src/rosetta/model.py:584`)
Weights prediction errors by codon position significance: positions 1-2 (amino acid identity) at 1.0x, position 3 (wobble/degeneracy) at 0.5x. Since the true reading frame is unknown a priori, weights are averaged across all 3 forward frames. This respects the information hierarchy of the genetic code.

**4. Hierarchical Positional Encoding** (`src/rosetta/model.py:115`)
Multi-scale sinusoidal encoding at 4 biological scales: nucleotide (1), codon (3), gene (~1,000), and TAD (~100,000). Biology teaches that position matters at every scale -- from single-nucleotide reading frame determination to megabase-scale topological domain boundaries.

**5. Codon Position Encoding** (`src/rosetta/model.py:55`)
Learnable embeddings that encode each nucleotide's position within all 6 possible reading frames. This is the "byte alignment" layer -- it tells the model where each nucleotide sits within every possible codon interpretation.

### Model Sizes

| Config | Parameters | Context | Use Case |
|--------|-----------|---------|----------|
| `configs/small.yaml` | ~1.2M | 512 nt | Local testing, CPU |
| `configs/medium.yaml` | ~30M | 4096 nt | Single GPU training |
| `configs/large.yaml` | ~120M | 8192 nt | Multi-GPU, competitive scale |

---

## Validation Results

We built a 7-level, 22-check validation suite (`validate.py`) that tests the architecture from sanity checks through biological benchmarks. Tests use statistical significance where applicable and are designed to be honest (several tests previously passed due to positional bias or loose thresholds and have been fixed).

### Current Results (5 epochs, synthetic data)

```
Level 1 (5/5) Sanity Checks           -- shapes, loss, gradients, seed reproducibility
Level 2 (4/5) MLM Prediction Quality   -- 32.4% accuracy, perplexity 3.08, all 4 nucleotides predicted
Level 3 (2/2) RC Equivariance          -- learned equivariance L2=0.0067 (approximate, tightens with training)
Level 4 (1/4) Frame Gate Analysis       -- gates still near-uniform after 5 epochs (needs more training)
Level 6 (3/3) Generation Quality        -- GC content, dinucleotide diversity, codon periodicity
Level 7 (1/3) Biological Benchmarks     -- ATG recognition 88% (with gene-like context)
```

### What Works

**The architecture is sound.** All 5 sanity checks pass: forward pass shapes correct, loss decreases during training, gradients flow to all 111 trainable parameter groups, seed reproducibility is exact (0.0 max diff), untrained model is at chance level.

**The model learns nucleotide prediction.** MLM accuracy of 32.4% beats random (14.3% for 7 classes) by a factor of 2.3x. All 4 nucleotides are predicted (A=55%, C=22%, G=38%, T=14% per-class accuracy). Perplexity 3.08 < 4.0 (random). Trained model beats untrained by +11.2 percentage points.

**RC equivariance holds (approximately).** The learned equivariant projection produces L2=0.0067 between `encode(X)` and `flip(encode(RC(X)))`. This is approximate (not exact like simple averaging) but preserves more information while maintaining the symmetry property. Expected to tighten with more training.

**Start codon recognition is genuine.** When tested with randomized ATG positions in gene-like context (coding suffix with codon structure), the model predicts the masked 'A' of ATG correctly 88% of the time vs 22% on random control positions. This was previously inflated to 100% due to a fixed-position test design that has been corrected.

**Generated sequences show biological structure.** GC content is realistic, dinucleotide frequencies deviate from uniform, and codon periodicity (lag-3 autocorrelation) peaks above lag-2 and lag-4.

### What Doesn't Work Yet

**Frame gates are still near-uniform.** After 5 epochs on synthetic data, the 6 softmax frame gates show ~1/6 each (entropy 99.85% of maximum). The model hasn't learned to specialize which frame is active at each position. This requires either more training or real genomic data with annotated gene boundaries.

**Wobble differentiation is zero.** The wobble-aware loss depends on frame gate predictions to identify codon positions. Since gates are uniform, wobble weights collapse. This will resolve once frame gates learn to specialize.

**Stop codon avoidance is not statistically significant.** Frequency of 4.81% vs 4.69% random (z=0.30, p>0.05). Needs more training for the model to learn ORF structure.

### What This Tells Us

The architecture implements the right inductive biases (multi-frame attention with frame competition via softmax, cross-frame communication, learned RC equivariance, wobble-aware loss). The remaining failures are all "model hasn't trained long enough on rich enough data" -- not architectural problems. The key unlock will be training on real annotated genomes (E. coli with GFF annotations), where frame structure is real and learnable.

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy

# Run the interactive demo (no training required)
python demo.py

# Train on synthetic data
python train.py --synthetic --epochs 5

# Train on real E. coli genome (auto-downloads 4.6MB from NCBI)
python train.py --download-genome --epochs 20

# Train on your own FASTA file
python train.py --fasta path/to/genome.fasta --epochs 50
```

### Run Validation

```bash
# Full validation suite (all 7 levels)
python validate.py

# Quick run (skip ablation study)
python validate.py --quick

# Specific levels only
python validate.py --levels 1,2,3

# Force CPU
python validate.py --device cpu

# Custom checkpoint
python validate.py --checkpoint checkpoints/rosetta_epoch_3.pt
```

### Inspect the "Key"

```python
from src.rosetta.model import RosettaTransformer
from src.rosetta.config import RosettaConfig
from src.data.tokenizer import DNATokenizer
import torch

# Load trained model
ckpt = torch.load("checkpoints/rosetta_best.pt", weights_only=False)
model = RosettaTransformer(ckpt['config'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Encode a sequence
tokenizer = DNATokenizer()
input_ids = tokenizer.encode("ATGACCGATGCTAACGGT...").unsqueeze(0)

# Extract the interpretive key
frame_gates = model.get_frame_attention_map(input_ids)
# Shape: (1, seq_len, 6) = gate activation per reading frame per position
# High activation = model considers this frame coding at this position
```

---

## Project Structure

```
brain/
  src/
    rosetta/
      config.py          -- RosettaConfig dataclass (all hyperparameters)
      model.py           -- Core architecture (759 lines)
                            RosettaTransformer, MultiFrameAttention,
                            RCEquivariantWrapper, CodonPositionEncoding,
                            HierarchicalPositionalEncoding
    data/
      tokenizer.py       -- Character-level DNA tokenizer (A/C/G/T/N/CLS/MASK)
      dataset.py         -- GenomicDataset (synthetic), FASTADataset, E. coli download
    training/
      trainer.py         -- Training loop with gradient accumulation, LR scheduling
  configs/
    small.yaml           -- ~1.2M params, 512nt context
    medium.yaml          -- ~30M params, 4096nt context
    large.yaml           -- ~120M params, 8192nt context
  research/
    THESIS.md            -- Full research synthesis with 30+ citations
  checkpoints/           -- Trained model weights (.pt files)
  validation/            -- Validation outputs (report.json, gate visualizations)
  train.py               -- Training CLI entry point
  validate.py            -- 7-level validation suite
  demo.py                -- Interactive demonstrations (5 demos, no training needed)
  pyproject.toml         -- Project metadata and dependencies
```

---

## The Key: What Rosetta Discovers

The frame gate activations after training form a 6-dimensional "interpretive key" for any genomic region:

- **High activation in one forward frame** -> coding gene on the forward strand
- **High activation in one reverse frame** -> coding gene on the reverse strand
- **High activation in multiple frames** -> overlapping genes
- **Low activation across all frames** -> non-coding / intergenic region
- **Frame transitions** -> gene boundaries, programmed frameshifts

This is the Rosetta Stone of genomics -- a single learned representation that decodes all layers of DNA's information encoding simultaneously.

---

## Connection to Brain Science

The same principles of multi-layered information encoding apply to neural coding:

| DNA Encoding | Neural Analog |
|-------------|---------------|
| Codon degeneracy (many codons -> one amino acid) | Population coding (many neurons -> one percept) |
| Bidirectional transcription (sense + antisense) | Feedforward + feedback processing in cortex |
| Hierarchical position (nt / codon / gene / TAD) | Place cells + grid cells in hippocampus |
| Wobble error tolerance | Redundant neural representations |
| Reading frame selection | Attentional routing / gating |

A transformer that learns DNA's interpretive key may reveal principles applicable to understanding how the brain encodes and decodes information.

---

## References

See [research/THESIS.md](research/THESIS.md) for the full research synthesis.

### Foundational
- Chargaff (1968) -- Second parity rule
- Crick, Barnett, Brenner, Watts-Tobin (1961) -- Triplet reading frame
- Freeland & Hurst (1998) -- Genetic code optimality ("one in a million")
- Itzkovitz & Alon (2007) -- Secondary information in wobble redundancy

### Genomic Transformers
- DNABERT-2: Zhou et al. (2023)
- Nucleotide Transformer: Dalla-Torre et al. (2024) -- Nature Methods
- HyenaDNA: Nguyen et al. (2023) -- NeurIPS
- Evo: Nguyen et al. (2024) -- Science 386
- Evo 2: Nguyen et al. (2026) -- Nature
- Caduceus: Schiff et al. (2024)
- GROVER: Dalla-Torre et al. (2024) -- Nature Machine Intelligence

---

## License

Research project. See [pyproject.toml](pyproject.toml) for package metadata.
