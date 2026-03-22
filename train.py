"""
Train the Rosetta transformer on genomic data.

Usage:
    # Train on synthetic data (for testing)
    python train.py --synthetic --epochs 5

    # Train on real E. coli genome
    python train.py --download-genome --epochs 20

    # Train on custom FASTA file
    python train.py --fasta path/to/genome.fasta --epochs 50
"""

import argparse
import torch

from src.rosetta.config import RosettaConfig
from src.rosetta.model import RosettaTransformer
from src.data.tokenizer import DNATokenizer
from src.data.dataset import GenomicDataset, FASTADataset, download_sample_genome
from src.training.trainer import RosettaTrainer


def main():
    parser = argparse.ArgumentParser(description="Train the Rosetta genomic transformer")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--download-genome", action="store_true", help="Download E. coli genome")
    parser.add_argument("--fasta", type=str, help="Path to FASTA file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Configuration
    config = RosettaConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_frame_layers=min(2, args.n_layers),
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Model
    model = RosettaTransformer(config)
    params = model.count_parameters()
    print("=" * 60)
    print("ROSETTA: Multi-Frame RC-Equivariant Genomic Transformer")
    print("=" * 60)
    print(f"\nParameter breakdown:")
    for name, count in params.items():
        print(f"  {name:30s}: {count:>12,}")
    print()

    # Dataset
    tokenizer = DNATokenizer(max_length=args.seq_length)

    if args.fasta:
        print(f"Loading FASTA: {args.fasta}")
        dataset = FASTADataset(args.fasta, seq_length=args.seq_length, tokenizer=tokenizer)
    elif args.download_genome:
        fasta_path = download_sample_genome()
        dataset = FASTADataset(fasta_path, seq_length=args.seq_length, tokenizer=tokenizer)
    else:
        print("Using synthetic genomic data")
        dataset = GenomicDataset(
            num_samples=5000,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
        )

    # Split into train/val
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Device:        {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()

    # Train
    trainer = RosettaTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    trainer.train(num_epochs=args.epochs)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
