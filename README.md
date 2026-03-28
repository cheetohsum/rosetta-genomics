# Rosetta: Finding the Key to DNA's Information Encoding

A multi-frame, reverse-complement-equivariant genomic transformer that models DNA's inherent "endianness" — strand direction, reading frame, codon position, and regulatory-scale structure.

**72M parameters** | **Multi-species pretraining** | **Coding + regulatory architecture** | [Architecture](#architecture) | [Benchmarks](#benchmark-results) | [Quick Start](#quick-start)

---

## The Thesis

Current genomic transformers (DNABERT-2, Nucleotide Transformer, Evo, Caduceus) treat DNA as flat text. But DNA encodes information across multiple overlapping systems:

**Coding endianness** (1.5% of genome):

| Dimension | Binary Analog | DNA Reality |
|-----------|--------------|-------------|
| **Strand Direction** | Big/little endian | 5'→3' vs 3'→5' (reverse complement) |
| **Reading Frame** | Byte alignment | Codon offset (0, 1, 2) — shifting by 1nt changes the entire protein |
| **Codon Position** | Bit significance | Position 1-2 = amino acid identity (~2 bits), Position 3 = wobble (~0.5 bits) |

**Regulatory endianness** (98.5% of genome):

| Dimension | Scale | DNA Reality |
|-----------|-------|-------------|
| **CpG dinucleotides** | 2bp | Primary epigenetic switch (methylation), marks ~70% of promoters |
| **Helical face** | ~10.5bp | TF binding accessibility — same motif on exposed vs hidden face has different function |
| **Nucleosome positioning** | 147bp | Chromatin wrapping periodicity, determines regulatory accessibility |

**Rosetta models both systems simultaneously.** No existing architecture does this.

### Gap Analysis

| Model | RC Equivariant | Multi-Frame | Wobble Loss | Multi-Scale Regulatory | Positional Scales | Conservation Head |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| DNABERT-2 | - | - | - | - | ALiBi | - |
| Nucleotide Transformer | - | - | - | - | Learned | - |
| HyenaDNA | - | - | - | - | Implicit | - |
| Caduceus | **Yes** | - | - | - | Single | - |
| **Rosetta** | **Yes** | **Yes** | **Yes** | **Yes** | **7-scale** | **Yes** |

---

## Architecture

```
Input: Raw nucleotide sequence (A, C, G, T)
  |
  v
[Nucleotide Embedding] + [Dinucleotide Embedding] + [Codon Position] + [Hierarchical Position]
  Character-level        16 pairs + 1 special       6-frame aware      7 biological scales
  (no k-mer loss)        (CpG-aware)                (learned)          (1/2/3/10/147/1K/100K bp)
  |
  v
[RC-Equivariant Multi-Frame Attention] × 4 layers       ← coding structure
  |-- Forward + reverse complement strand processing
  |-- 6 per-frame Q projections, shared K/V
  |-- Competitive softmax gating (which frame is active?)
  |-- Flash Attention + gradient checkpointing
  |
  v
[RC-Equivariant Multi-Scale Attention] × 2 layers       ← regulatory structure
  |-- 3 per-scale Q projections (motif/nucleosome/enhancer)
  |-- Competitive softmax gating (which scale is active?)
  |-- Same Flash Attention + checkpointing
  |
  v
[Strand Fusion] → merge forward + RC representations
  |
  v
[Standard Transformer] × 6 layers                       ← integration
  |
  v
[Output Heads]
  |-- MLM Head: masked nucleotide prediction (vocab=7)
  |-- Frame Head: reading frame classification (6 outputs)
  |-- Conservation Head: per-position evolutionary constraint (sigmoid, 0-1)
  |-- Generation Head: autoregressive DNA synthesis (A/C/G/T)
```

### Key Components

| Component | Description | Innovation |
|-----------|-------------|------------|
| **Multi-Frame Attention** | 6 parallel attention pathways (3 fwd + 3 RC reading frames) with competitive softmax gating | First model to attend across all reading frames simultaneously |
| **Multi-Scale Attention** | 3 regulatory-scale attention pathways (motif ~10bp, nucleosome ~150bp, enhancer ~500bp) | Captures regulatory structure at biologically meaningful scales |
| **RC Equivariance** | Architectural guarantee: `f(RC(x)) = reverse(f(x))` | Both strands produce consistent representations |
| **Wobble-Aware Loss** | Per-codon degeneracy weighting — identity positions weighted 1.0, wobble positions by actual degeneracy (0.25-1.0) | Respects the genetic code's error tolerance structure |
| **Conditional Wobble** | Wobble/entropy weighting only on CDS-annotated windows; plain CE on non-coding | Prevents representation collapse on human data |
| **7-Scale Positional Encoding** | Sinusoidal at 1bp, 2bp, 3bp, 10bp, 147bp, 1Kbp, 100Kbp | Spans nucleotide to topological domain scales |
| **Dinucleotide Embedding** | 17-class embedding for overlapping nucleotide pairs | Direct CpG/epigenetic sensitivity |
| **Conservation Head** | Predicts per-position evolutionary constraint from local entropy proxy | Teaches coding vs regulatory conservation signatures |

---

## Training Pipeline

### Multi-Species Pretraining

Trains on 5 genomes spanning prokaryotes to human:

| Genome | Size | Coding % | Role |
|--------|------|----------|------|
| E. coli K-12 | 4.6 MB | 87% | Dense coding signal |
| B. subtilis 168 | 4.2 MB | 87% | Gram-positive diversity |
| S. cerevisiae S288C | 12 MB | 70% | Eukaryotic introns |
| Human chr22 | 51 MB | ~1.5% | Regulatory structure |
| Human chr1 | 249 MB | ~1.5% | Large-scale genome context |

### Training Strategy

- **Gradual curriculum**: Human data weight ramps from 0% → equal over first 2 epochs
- **CDS-guided sampling**: Windows overlapping coding annotations get 3× sampling weight (most impactful for human, where 98.5% is non-coding)
- **Conditional loss**: Wobble/entropy weighting only on CDS windows; plain cross-entropy on non-coding (prevents near-zero gradient signal that caused representation collapse)
- **Plain CE warmup**: First 500 optimizer steps use unweighted cross-entropy to establish basic MLM before complex loss
- **Progressive context**: Optional 512bp → 2048bp context growth mid-training for regulatory-scale patterns
- **Anti-collapse monitoring**: Per-epoch embedding cosine similarity check + per-step A/C/G/T prediction distribution

### Performance (RTX 3090, 24GB)

| Configuration | Batch | VRAM | Tok/s |
|---|---|---|---|
| 512bp, AMP, Flash Attn + grad ckpt | 256 | 18.8 GB (78%) | ~24,500 |
| 2048bp progressive phase | 64 | ~18.8 GB | ~6,000-8,000 |

```bash
# Standard training run
python pretrain_multispecies.py --amp

# With progressive context growth (512bp → 2048bp at epoch 3)
python pretrain_multispecies.py --amp --progressive-context

# Resume from checkpoint with different batch size
python pretrain_multispecies.py --amp --batch-size 256 --resume checkpoints/rosetta_epoch_1.pt
```

---

## Benchmark Results

Probing protocol: freeze pretrained weights, extract mean-pooled embeddings, train logistic regression (5-fold CV). Reports Matthews Correlation Coefficient (MCC × 100).

### Current (partial training, ~1.5 epochs with anti-collapse fix)

| Task | Rosetta | Random Init | NT-2500M | DNABERT-2 |
|------|---------|-------------|----------|-----------|
| enhancers | 30.4 | 32.4 | 60.6 | 55.0 |
| promoter_all | 65.1 | 66.5 | 91.0 | 86.8 |
| promoter_tata | 50.5 | 66.9 | 94.3 | 94.3 |

Pretrained representations are no longer collapsed (MCC was ~1 before anti-collapse fix). Not yet beating random init — model needs more training. Published baselines use 500M-2.5B parameters vs our 72M.

### Validation Suite

7-level, 22-check validation (`validate.py`):

- **Level 1**: Sanity checks — shapes, loss decrease, gradients, reproducibility (all pass)
- **Level 2**: MLM quality — accuracy, per-nucleotide breakdown, perplexity
- **Level 3**: RC equivariance — exact L2=0.000000 (architecturally enforced)
- **Level 4**: Frame gate analysis — activation patterns, coding/non-coding contrast
- **Level 5**: Ablation studies — full vs no-RC vs no-wobble vs no-multiframe
- **Level 6**: Generation quality — GC content, dinucleotides, codon periodicity
- **Level 7**: Biological benchmarks — Chargaff's rule, start codons, stop codon avoidance

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy scikit-learn

# Run multi-species pretraining (downloads genomes automatically)
python pretrain_multispecies.py --amp

# Run validation suite
python validate.py --quick

# Run benchmarks against published baselines
python benchmark.py

# Train on synthetic data (for testing)
python train.py --synthetic --epochs 5

# Train on real E. coli genome (auto-downloads 4.6MB from NCBI)
python train.py --download-genome --epochs 20
```

---

## Project Structure

```
brain/
  src/
    rosetta/
      config.py          -- RosettaConfig (all hyperparameters)
      model.py           -- RosettaTransformer, MultiFrameAttention,
                            MultiScaleAttention, RCEquivariantWrapper,
                            CodonPositionEncoding, HierarchicalPositionalEncoding
    data/
      tokenizer.py       -- Character-level DNA tokenizer (A/C/G/T/N/CLS/MASK)
      dataset.py         -- GenomicDataset (synthetic), FASTADataset,
                            conservation target computation, CDS-guided sampling
    training/
      trainer.py         -- Training loop with AMP, gradient checkpointing,
                            collapse monitoring, NaN recovery
  research/
    THESIS.md            -- Full research synthesis with 30+ citations
    GENOMIC_BENCHMARKS.md -- Survey of published benchmark suites
  pretrain_multispecies.py -- Multi-species curriculum pretraining pipeline
  benchmark.py           -- Probing benchmark against NT/DNABERT-2 baselines
  validate.py            -- 7-level validation suite
  train.py               -- Single-species training CLI
  demo.py                -- Interactive demonstrations
```

---

## References

See [research/THESIS.md](research/THESIS.md) for the full research synthesis.

### Foundational
- Chargaff (1968) — Second parity rule
- Crick, Barnett, Brenner, Watts-Tobin (1961) — Triplet reading frame
- Freeland & Hurst (1998) — Genetic code optimality ("one in a million")
- Itzkovitz & Alon (2007) — Secondary information in wobble redundancy

### Genomic Transformers
- DNABERT-2: Zhou et al. (2023)
- Nucleotide Transformer: Dalla-Torre et al. (2024) — Nature Methods
- HyenaDNA: Nguyen et al. (2023) — NeurIPS
- Evo: Nguyen et al. (2024) — Science 386
- Evo 2: Nguyen et al. (2026) — Nature
- Caduceus: Schiff et al. (2024)

---

## License

Research project. See [pyproject.toml](pyproject.toml) for package metadata.
