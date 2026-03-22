"""
Genomic datasets for training Rosetta.

Supports:
1. FASTA files (standard genomic format)
2. Synthetic data generation for testing
3. Public genome downloads (NCBI/Ensembl)
"""

import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from .tokenizer import DNATokenizer


class GenomicDataset(Dataset):
    """
    Dataset that generates random genomic windows for pre-training.

    For real training, replace with FASTADataset loading actual genome data.
    This synthetic dataset generates sequences with realistic properties:
    - ~40-60% GC content
    - Coding regions with codon structure
    - Approximate Chargaff's second parity rule compliance
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 1024,
        coding_fraction: float = 0.3,
        tokenizer: Optional[DNATokenizer] = None,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.coding_fraction = coding_fraction
        self.tokenizer = tokenizer or DNATokenizer(max_length=seq_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a synthetic genomic sequence with realistic properties."""
        seq, coding_regions = self._generate_realistic_sequence()

        input_ids = self.tokenizer.encode(seq)
        masked_ids, labels = self.tokenizer.mask_tokens(input_ids)
        frame_labels = self._generate_frame_labels(len(seq), coding_regions)

        return {
            'input_ids': masked_ids,
            'labels': labels,
            'original_ids': input_ids,
            'frame_labels': frame_labels,
        }

    def _generate_realistic_sequence(self) -> tuple[str, list[tuple[int, int, int]]]:
        """
        Generate a sequence with realistic genomic properties.

        Returns:
            (sequence, coding_regions) where coding_regions is a list of
            (start, end, frame) tuples tracking where ORFs were placed.
        """
        seq = []
        coding_regions = []

        pos = 0
        while pos < self.seq_length:
            if random.random() < self.coding_fraction:
                # Coding region with optional frame offset
                frame = random.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0]
                # Add offset nucleotides to shift the reading frame
                if frame > 0:
                    offset_nts = random.choices(['A', 'C', 'G', 'T'], k=frame)
                    seq.extend(offset_nts)
                    pos += frame

                region_len = random.randint(90, 600)
                region_len = min(region_len, self.seq_length - pos)
                region = self._generate_coding_region(region_len)
                coding_regions.append((pos, pos + len(region), frame))
            else:
                region_len = random.randint(100, 500)
                region_len = min(region_len, self.seq_length - pos)
                gc_content = random.uniform(0.35, 0.65)
                region = self._generate_noncoding_region(region_len, gc_content)

            seq.extend(region)
            pos += len(region)

        return ''.join(seq[:self.seq_length]), coding_regions

    def _generate_coding_region(self, length: int) -> list[str]:
        """Generate a coding region with codon structure."""
        # Codon usage frequencies (simplified, roughly E. coli-like)
        common_codons = [
            'ATG', 'GCT', 'GAA', 'AAA', 'CTG', 'GTG', 'ACC',
            'GAT', 'GGC', 'TTC', 'CAG', 'AGC', 'CCG', 'TAT',
            'CGC', 'TGG', 'AAC', 'CAC', 'ATC', 'GTA',
        ]

        region = list('ATG')  # Start codon
        while len(region) < length - 3:
            codon = random.choice(common_codons)
            region.extend(list(codon))

        # Add stop codon
        stop = random.choice(['TAA', 'TAG', 'TGA'])
        region.extend(list(stop))

        return region[:length]

    def _generate_noncoding_region(self, length: int, gc_content: float) -> list[str]:
        """Generate a non-coding region with specified GC content."""
        gc_prob = gc_content / 2
        at_prob = (1 - gc_content) / 2
        weights = [at_prob, gc_prob, gc_prob, at_prob]  # A, C, G, T
        return random.choices(['A', 'C', 'G', 'T'], weights=weights, k=length)

    def _generate_frame_labels(
        self, seq_len: int, coding_regions: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        """Generate binary labels based on actual coding region positions."""
        labels = torch.zeros(seq_len, 6)
        for start, end, frame in coding_regions:
            end = min(end, seq_len)
            if start < seq_len:
                labels[start:end, frame] = 1.0
        return labels


class FASTADataset(Dataset):
    """
    Dataset loading real genomic sequences from FASTA files.

    Optionally loads GFF annotations to generate frame labels for
    supervised frame prediction and wobble-aware loss.
    """

    def __init__(
        self,
        fasta_path: str,
        seq_length: int = 1024,
        stride: int = 512,
        tokenizer: Optional[DNATokenizer] = None,
        gff_path: Optional[str] = None,
    ):
        self.seq_length = seq_length
        self.stride = stride
        self.tokenizer = tokenizer or DNATokenizer(max_length=seq_length)

        # Load FASTA file
        self.sequences = self._load_fasta(fasta_path)

        # Load gene annotations if available
        self.gene_annotations = []
        if gff_path is not None:
            self.gene_annotations = self._load_gff(gff_path)

        # Create windows
        self.windows = []
        for seq_idx, seq in enumerate(self.sequences):
            for start in range(0, len(seq) - seq_length + 1, stride):
                self.windows.append((seq_idx, start))

    def _load_fasta(self, path: str) -> list[str]:
        """Load sequences from a FASTA file."""
        sequences = []
        current_seq = []

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                elif line:
                    filtered = ''.join(c for c in line.upper() if c in 'ACGTN')
                    current_seq.append(filtered)

        if current_seq:
            sequences.append(''.join(current_seq))

        return sequences

    def _load_gff(self, path: str) -> list[list[tuple[int, int, int, int]]]:
        """
        Load CDS annotations from a GFF3 file.

        Returns list (per sequence) of (start, end, strand, frame) tuples.
        strand: 0=forward, 1=reverse. frame: 0, 1, or 2.
        """
        annotations: list[list[tuple[int, int, int, int]]] = [[] for _ in self.sequences]

        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                feature_type = parts[2]
                if feature_type != 'CDS':
                    continue

                start = int(parts[3]) - 1  # GFF is 1-based
                end = int(parts[4])
                strand = 0 if parts[6] == '+' else 1
                frame = int(parts[7]) if parts[7].isdigit() else 0

                # Assign to first sequence (single-chromosome genomes)
                if annotations:
                    annotations[0].append((start, end, strand, frame))

        return annotations

    def _get_frame_labels(self, seq_idx: int, start: int) -> Optional[torch.Tensor]:
        """Generate frame labels for a window from gene annotations."""
        if not self.gene_annotations or seq_idx >= len(self.gene_annotations):
            return None

        labels = torch.zeros(self.seq_length, 6)
        end = start + self.seq_length

        for gene_start, gene_end, strand, frame in self.gene_annotations[seq_idx]:
            # Check overlap with window
            overlap_start = max(gene_start, start) - start
            overlap_end = min(gene_end, end) - start

            if overlap_start < overlap_end:
                if strand == 0:  # forward strand
                    frame_idx = frame  # 0, 1, or 2
                else:  # reverse strand
                    frame_idx = frame + 3  # 3, 4, or 5
                labels[overlap_start:overlap_end, frame_idx] = 1.0

        return labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start = self.windows[idx]
        seq_window = self.sequences[seq_idx][start:start + self.seq_length]

        input_ids = self.tokenizer.encode(seq_window)
        masked_ids, labels = self.tokenizer.mask_tokens(input_ids)

        result = {
            'input_ids': masked_ids,
            'labels': labels,
            'original_ids': input_ids,
        }

        # Add frame labels from annotations if available
        frame_labels = self._get_frame_labels(seq_idx, start)
        if frame_labels is not None:
            result['frame_labels'] = frame_labels

        return result


def download_sample_genome(output_dir: str = "data/genomes") -> tuple[str, Optional[str]]:
    """
    Download E. coli K-12 MG1655 genome and annotations.

    Returns (fasta_path, gff_path). gff_path may be None if download fails.
    """
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
        "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2"
    )

    # Download FASTA
    fasta_path = output_path / "ecoli_k12.fasta"
    if not fasta_path.exists():
        import gzip
        print("Downloading E. coli K-12 genome from NCBI...")
        gz_path = output_path / "ecoli_k12.fasta.gz"
        urllib.request.urlretrieve(f"{base_url}_genomic.fna.gz", gz_path)
        with gzip.open(gz_path, 'rt') as f_in:
            with open(fasta_path, 'w') as f_out:
                f_out.write(f_in.read())
        gz_path.unlink()
        print(f"  FASTA: {fasta_path}")
    else:
        print(f"Genome already downloaded: {fasta_path}")

    # Download GFF annotations
    gff_path = output_path / "ecoli_k12.gff"
    if not gff_path.exists():
        try:
            import gzip
            print("Downloading E. coli K-12 annotations...")
            gz_path = output_path / "ecoli_k12.gff.gz"
            urllib.request.urlretrieve(f"{base_url}_genomic.gff.gz", gz_path)
            with gzip.open(gz_path, 'rt') as f_in:
                with open(gff_path, 'w') as f_out:
                    f_out.write(f_in.read())
            gz_path.unlink()
            print(f"  GFF:   {gff_path}")
        except Exception as e:
            print(f"  GFF download failed: {e}")
            gff_path = None
    else:
        print(f"Annotations already downloaded: {gff_path}")

    return str(fasta_path), str(gff_path) if gff_path and gff_path.exists() else None
