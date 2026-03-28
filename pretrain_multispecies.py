"""
Multi-species pretraining pipeline for Rosetta.

Downloads and trains on genomes from multiple species to build
cross-species representations for downstream benchmarks.

Genomes:
  - E. coli K-12 (4.6MB, bacterial, 87% coding)
  - S. cerevisiae (12MB, yeast, histone benchmark organism)
  - Human chr22 + chr1 (~300MB, covers promoter/enhancer benchmarks)
  - B. subtilis (4.2MB, gram-positive bacterial diversity)

Training strategy:
  - Curriculum: bacteria/yeast first (clean codon signal), then all species
  - Species-balanced sampling: equal weight per species regardless of genome size
  - Dynamic LR schedule: warmup + cosine decay tuned to actual step count

Usage:
    # Overnight run (recommended):
    python pretrain_multispecies.py --amp

    # Quick test on bacteria only:
    python pretrain_multispecies.py --amp --skip-human --epochs 2

    # Full control:
    python pretrain_multispecies.py --amp --d-model 512 --n-layers 12 --lr 6e-4
"""

import argparse
import gzip
import math
import os
import shutil
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, random_split

from src.rosetta.config import RosettaConfig
from src.rosetta.model import RosettaTransformer
from src.data.tokenizer import DNATokenizer
from src.data.dataset import FASTADataset
from src.training.trainer import RosettaTrainer


# =============================================================================
# Genome registry
# =============================================================================

GENOMES = {
    'ecoli': {
        'name': 'E. coli K-12 MG1655',
        'fasta_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz',
        'gff_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.gff.gz',
        'group': 'prokaryote',
    },
    'yeast': {
        'name': 'S. cerevisiae S288C',
        'fasta_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.fna.gz',
        'gff_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.gff.gz',
        'group': 'prokaryote',  # grouped with bacteria for curriculum phase 1
    },
    'bsubtilis': {
        'name': 'B. subtilis 168',
        'fasta_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.fna.gz',
        'gff_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.gff.gz',
        'group': 'prokaryote',
    },
    'human_chr22': {
        'name': 'Human chr22 (GRCh38)',
        'fasta_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/chr22.fna.gz',
        'gff_url': None,
        'group': 'human',
    },
    'human_chr1': {
        'name': 'Human chr1 (GRCh38)',
        'fasta_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_structure/Primary_Assembly/assembled_chromosomes/FASTA/chr1.fna.gz',
        'gff_url': None,
        'group': 'human',
    },
    'human_gff': {
        'name': 'Human GRCh38 annotations',
        'fasta_url': None,
        'gff_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.gff.gz',
        'group': 'human',
    },
}


def download_genome(genome_id: str, output_dir: Path) -> tuple[str, str]:
    """Download a genome's FASTA and GFF files. Returns (fasta_path, gff_path)."""
    info = GENOMES[genome_id]
    fasta_path = None
    gff_path = None

    if info['fasta_url']:
        fasta_path = output_dir / f"{genome_id}.fasta"
        if not fasta_path.exists():
            print(f"  Downloading {info['name']} FASTA...")
            gz_path = output_dir / f"{genome_id}.fasta.gz"
            urllib.request.urlretrieve(info['fasta_url'], gz_path)
            with gzip.open(gz_path, 'rt') as f_in:
                with open(fasta_path, 'w') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            size_mb = fasta_path.stat().st_size / 1e6
            print(f"    -> {fasta_path} ({size_mb:.1f} MB)")
        else:
            print(f"  Already have: {fasta_path}")
        fasta_path = str(fasta_path)

    if info['gff_url']:
        gff_path = output_dir / f"{genome_id}.gff"
        if not gff_path.exists():
            print(f"  Downloading {info['name']} GFF...")
            gz_path = output_dir / f"{genome_id}.gff.gz"
            urllib.request.urlretrieve(info['gff_url'], gz_path)
            with gzip.open(gz_path, 'rt') as f_in:
                with open(gff_path, 'w') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            size_mb = gff_path.stat().st_size / 1e6
            print(f"    -> {gff_path} ({size_mb:.1f} MB)")
        else:
            print(f"  Already have: {gff_path}")
        gff_path = str(gff_path)

    return fasta_path, gff_path


def split_and_combine(species_datasets: dict, val_fraction: float = 0.1):
    """Split each species into train/val, then combine. Avoids Subset indexing issues."""
    train_parts = []
    val_parts = []
    for gid, ds in species_datasets.items():
        val_n = max(1, int(len(ds) * val_fraction))
        train_n = len(ds) - val_n
        tr, va = random_split(ds, [train_n, val_n])
        train_parts.append(tr)
        val_parts.append(va)
    return train_parts, ConcatDataset(train_parts), ConcatDataset(val_parts)


def build_balanced_sampler(datasets: list, num_samples: int) -> WeightedRandomSampler:
    """Build a sampler that gives equal weight to each species per epoch."""
    weights = []
    n_species = len(datasets)
    for ds in datasets:
        w = 1.0 / (n_species * len(ds))
        weights.extend([w] * len(ds))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def _get_cds_weight(ds, idx: int, cds_boost: float = 3.0) -> float:
    """Get sampling weight boost for a window: cds_boost if it overlaps CDS, else 1.0."""
    # Reach through Subset to the underlying FASTADataset
    underlying = ds.dataset if hasattr(ds, 'dataset') else ds
    if hasattr(underlying, 'has_cds_annotation'):
        # Map Subset index to original dataset index
        orig_idx = ds.indices[idx] if hasattr(ds, 'indices') else idx
        return cds_boost if underlying.has_cds_annotation(orig_idx) else 1.0
    return 1.0


def build_blended_sampler(
    datasets: list,
    species_groups: list[str],
    human_weight: float,
    num_samples: int,
    cds_boost: float = 3.0,
) -> WeightedRandomSampler:
    """Build a sampler with specified human vs prokaryote balance.

    human_weight: fraction of sampling budget for human species (0.0 to 1.0).
    Prokaryote species share the remaining (1 - human_weight) equally.
    cds_boost: within each species, windows overlapping CDS get this multiplier
               (default 3x). Makes human training focus on gene-dense regions.
    """
    n_prok = sum(1 for g in species_groups if g == 'prokaryote')
    n_human = sum(1 for g in species_groups if g == 'human')

    weights = []
    for ds, group in zip(datasets, species_groups):
        if group == 'prokaryote' and n_prok > 0:
            species_budget = (1.0 - human_weight) / n_prok
        elif group == 'human' and n_human > 0 and human_weight > 0:
            species_budget = human_weight / n_human
        else:
            species_budget = 0.0

        # Per-window CDS boost (most impactful for human where ~98.5% is non-coding)
        window_weights = []
        for i in range(len(ds)):
            w = species_budget * _get_cds_weight(ds, i, cds_boost)
            window_weights.append(w)
        weights.extend(window_weights)

    # If all weights are 0, fall back to uniform
    if sum(weights) == 0:
        weights = [1.0 / max(len(weights), 1)] * len(weights)

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def load_species_datasets(genome_files, seq_length, stride, tokenizer, use_electra=False):
    """Load FASTADatasets per species. Returns {genome_id: dataset}."""
    species_datasets = {}
    for gid, (fasta_path, gff_path) in genome_files.items():
        print(f"  Loading {gid}: {fasta_path}")
        try:
            ds = FASTADataset(
                fasta_path,
                seq_length=seq_length,
                stride=stride,
                tokenizer=tokenizer,
                gff_path=gff_path,
                use_electra=use_electra,
            )
            print(f"    -> {len(ds)} windows")
            species_datasets[gid] = ds
        except Exception as e:
            print(f"    FAILED: {e}")
    return species_datasets


def main():
    parser = argparse.ArgumentParser(description="Multi-species pretraining for Rosetta")
    parser.add_argument("--data-dir", type=str, default="D:/sourcecode/brain_data/genomes",
                        help="Directory for genome downloads")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")

    # Training schedule
    parser.add_argument("--epochs", type=int, default=5,
                        help="Total training epochs (across all phases)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate")
    parser.add_argument("--accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (FP16)")
    parser.add_argument("--electra", action="store_true",
                        help="Use ELECTRA (replaced-token detection) instead of MLM")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker processes (0 on Windows)")

    # Data
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None,
                        help="Window stride (default: seq-length, i.e. no overlap)")
    parser.add_argument("--skip-human", action="store_true",
                        help="Skip human chromosomes (faster, less disk)")
    parser.add_argument("--progressive-context", action="store_true",
                        help="Start at 512bp, switch to 2048bp at epoch 3")
    parser.add_argument("--long-seq-length", type=int, default=2048,
                        help="Sequence length for progressive context phase 2")
    parser.add_argument("--long-batch-size", type=int, default=64,
                        help="Batch size for progressive context phase 2")

    # Model
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)

    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning (train all species from start)")
    parser.add_argument("--curriculum-epochs", type=int, default=2,
                        help="Epochs before human data reaches equal weight")
    parser.add_argument("--warmup-ce-steps", type=int, default=500,
                        help="Steps using plain CE before wobble/entropy weighting")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (e.g. checkpoints/rosetta_epoch_1.pt)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stride = args.stride if args.stride is not None else args.seq_length

    print("=" * 60)
    print("ROSETTA MULTI-SPECIES PRETRAINING")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Download genomes
    # ---------------------------------------------------------------
    print("\n--- Step 1: Downloading genomes ---\n")

    genome_list = ['ecoli', 'yeast', 'bsubtilis']
    if not args.skip_human:
        genome_list.extend(['human_gff', 'human_chr22', 'human_chr1'])

    genome_files = {}
    for gid in genome_list:
        fasta, gff = download_genome(gid, data_dir)
        if fasta:
            genome_files[gid] = (fasta, gff)

    # Human chromosomes share the whole-genome GFF
    human_gff_path = str(data_dir / "human_gff.gff") if (data_dir / "human_gff.gff").exists() else None
    for hg_key in ['human_chr22', 'human_chr1']:
        if hg_key in genome_files:
            fasta, _ = genome_files[hg_key]
            genome_files[hg_key] = (fasta, human_gff_path)

    # ---------------------------------------------------------------
    # Step 2: Load all datasets
    # ---------------------------------------------------------------
    print("\n--- Step 2: Creating datasets ---\n")

    total_fasta_mb = sum(
        os.path.getsize(f) / 1e6 for f, _ in genome_files.values() if os.path.isfile(f)
    )
    print(f"  Total FASTA on disk: {total_fasta_mb:.0f} MB")
    if total_fasta_mb > 500:
        print(f"  Frame labels for seqs > 10M bp use sparse storage to save RAM.")

    tokenizer = DNATokenizer(max_length=args.seq_length)
    species_datasets = load_species_datasets(genome_files, args.seq_length, stride, tokenizer, args.electra)

    # Split into curriculum groups
    prokaryote_datasets = {k: v for k, v in species_datasets.items()
                           if GENOMES[k]['group'] == 'prokaryote'}
    human_datasets = {k: v for k, v in species_datasets.items()
                      if GENOMES[k]['group'] == 'human'}

    for group_name, group in [("Prokaryote/yeast", prokaryote_datasets), ("Human", human_datasets)]:
        total = sum(len(ds) for ds in group.values())
        print(f"  {group_name}: {total:,} windows ({len(group)} genomes)")

    # ---------------------------------------------------------------
    # Step 3: Build model
    # ---------------------------------------------------------------
    print("\n--- Step 3: Model ---\n")

    effective_batch = args.batch_size * args.accumulation

    config = RosettaConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_frame_layers=min(4, args.n_layers),
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        use_amp=args.amp,
        num_workers=args.workers,
        warmup_plain_ce_steps=args.warmup_ce_steps,
        use_electra=args.electra,
    )

    model = RosettaTransformer(config)
    params = model.count_parameters()
    print(f"  Model parameters: {params['total']:,}")
    print(f"  Context window: {args.seq_length} nt")
    print(f"  Effective batch: {effective_batch} ({args.batch_size} x {args.accumulation})")
    print(f"  Peak LR: {args.lr:.1e}")
    device_name = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device_name}")
    if args.amp:
        print(f"  Mixed precision: FP16")
    print()

    # ---------------------------------------------------------------
    # Step 4: Train
    # ---------------------------------------------------------------
    use_curriculum = (not args.no_curriculum
                      and human_datasets
                      and args.curriculum_epochs < args.epochs)

    # All species split into train/val
    all_train_parts, all_train, all_val = split_and_combine(species_datasets)
    train_size = len(all_train)

    # Species groups for blended sampling
    species_groups = [GENOMES[gid]['group'] for gid in species_datasets]

    # Dynamic LR schedule
    steps_per_epoch = math.ceil(train_size / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    config.warmup_steps = max(500, min(1000, steps_per_epoch // 2))
    config.max_steps = total_steps

    print(f"  Windows: {train_size:,} train, {len(all_val):,} val")
    print(f"  Steps/epoch: ~{steps_per_epoch:,}")
    print(f"  Warmup: {config.warmup_steps} steps (LR) + {config.warmup_plain_ce_steps} steps (plain CE)")
    print(f"  Total: {total_steps:,} steps across {args.epochs} epochs")

    if use_curriculum:
        # === Gradual blend: linearly increase human weight over curriculum_epochs ===
        blend_epochs = args.curriculum_epochs
        print(f"\n--- Gradual curriculum: human weight 0% -> equal over {blend_epochs} epochs ---\n")

        # Initial sampler: 0% human
        sampler = build_blended_sampler(all_train_parts, species_groups, 0.0, train_size)
        trainer = RosettaTrainer(
            model=model, config=config,
            train_dataset=all_train, val_dataset=all_val,
            output_dir=args.output_dir, train_sampler=sampler,
        )

        # Resume from checkpoint if requested
        start_epoch = 0
        if args.resume:
            print(f"  Resuming from {args.resume}", flush=True)
            trainer.load_checkpoint(args.resume)
            # Infer which epoch we're on from global_step
            start_epoch = trainer.global_step // steps_per_epoch
            print(f"  Restored global_step={trainer.global_step}, starting at epoch {start_epoch+1}", flush=True)

        n_species = len(species_datasets)
        equal_weight = 1.0 / n_species  # target human weight at full blend

        current_seq_length = args.seq_length
        current_batch_size = config.batch_size

        for epoch in range(start_epoch, args.epochs):
            # Progressive context: switch to longer sequences at epoch 3
            if (args.progressive_context and epoch >= blend_epochs
                    and current_seq_length < args.long_seq_length):
                current_seq_length = args.long_seq_length
                current_batch_size = args.long_batch_size
                config.batch_size = current_batch_size
                config.gradient_accumulation_steps = max(1, 256 // current_batch_size)
                print(f"\n  === Context transition: {args.seq_length}bp -> {current_seq_length}bp, "
                      f"batch={current_batch_size}, accum={config.gradient_accumulation_steps} ===\n",
                      flush=True)

                # Rebuild datasets at new window size
                new_stride = current_seq_length
                species_datasets = load_species_datasets(
                    genome_files, current_seq_length, new_stride, tokenizer, args.electra
                )
                species_groups = [GENOMES[gid]['group'] for gid in species_datasets]
                all_train_parts, all_train, all_val = split_and_combine(species_datasets)
                train_size = len(all_train)
                steps_per_epoch = math.ceil(train_size / (current_batch_size * config.gradient_accumulation_steps))
                print(f"  Rebuilt: {train_size:,} windows, ~{steps_per_epoch:,} steps/epoch", flush=True)

                # Update val loader on trainer
                nw = config.num_workers
                trainer.val_loader = DataLoader(
                    all_val, batch_size=current_batch_size, shuffle=False,
                    num_workers=nw, pin_memory=(trainer.device.type == "cuda" and nw > 0),
                )

            # Ramp human weight: 0 at epoch 0, equal_weight at blend_epochs
            if epoch < blend_epochs:
                human_w = equal_weight * (epoch / max(blend_epochs - 1, 1))
            else:
                human_w = equal_weight

            prok_w = (1.0 - human_w * sum(1 for g in species_groups if g == 'human'))
            print(f"  Epoch {epoch+1}: seq={current_seq_length}bp, human={human_w:.3f}/species, "
                  f"prokaryote={prok_w:.3f} total", flush=True)

            # Rebuild sampler with updated blend
            new_sampler = build_blended_sampler(
                all_train_parts, species_groups, human_w, train_size
            )
            nw = config.num_workers
            trainer.train_loader = DataLoader(
                all_train,
                batch_size=current_batch_size,
                sampler=new_sampler,
                num_workers=nw,
                pin_memory=(trainer.device.type == "cuda" and nw > 0),
                persistent_workers=nw > 0,
            )
            trainer.train(num_epochs=1, log_interval=10)

    else:
        # === No curriculum: train on everything with balanced sampling ===
        print(f"\n--- Training: all species ({args.epochs} epochs) ---\n")

        sampler = build_balanced_sampler(all_train_parts, num_samples=train_size)
        trainer = RosettaTrainer(
            model=model, config=config,
            train_dataset=all_train, val_dataset=all_val,
            output_dir=args.output_dir, train_sampler=sampler,
        )
        if args.resume:
            print(f"  Resuming from {args.resume}", flush=True)
            trainer.load_checkpoint(args.resume)
            print(f"  Restored global_step={trainer.global_step}", flush=True)
        trainer.train(num_epochs=args.epochs, log_interval=10)

    print("\nMulti-species pretraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
