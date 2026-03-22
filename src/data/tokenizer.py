"""
DNA Tokenizer for the Rosetta transformer.

We use character-level tokenization (each nucleotide = 1 token) because:
1. No information loss from k-mer or BPE compression
2. The model learns codon structure through the CodonPositionEncoding
3. Reading frame boundaries are preserved exactly
4. Reverse complement is a simple per-character operation

This is deliberate: we want the model to discover the "byte alignment"
(reading frame) and "endianness" (strand direction) from data, not
impose it through tokenization.
"""

import torch
from typing import Optional


class DNATokenizer:
    """Character-level DNA tokenizer."""

    # Vocabulary: each nucleotide is one token
    VOCAB = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3,
        'N': 4,       # unknown/ambiguous
        '[CLS]': 5,   # classification token
        '[MASK]': 6,   # masking token for MLM
    }
    INV_VOCAB = {v: k for k, v in VOCAB.items()}

    # Complement mapping for RC
    COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

    def __init__(self, max_length: int = 8192):
        self.max_length = max_length
        self.vocab_size = len(self.VOCAB)
        self.pad_token_id = self.VOCAB['N']  # Use N as padding
        self.mask_token_id = self.VOCAB['[MASK]']
        self.cls_token_id = self.VOCAB['[CLS]']

    def encode(
        self,
        sequence: str,
        add_cls: bool = False,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a DNA sequence string to token indices.

        Args:
            sequence: DNA string (e.g., "ATCGATCG")
            add_cls: prepend [CLS] token
            max_length: override max length

        Returns:
            (seq_len,) tensor of token indices
        """
        max_len = max_length or self.max_length
        sequence = sequence.upper().replace('U', 'T')  # Handle RNA input

        tokens = []
        if add_cls:
            tokens.append(self.cls_token_id)

        for char in sequence[:max_len - (1 if add_cls else 0)]:
            tokens.append(self.VOCAB.get(char, self.VOCAB['N']))

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token indices back to a DNA string."""
        return ''.join(
            self.INV_VOCAB.get(idx.item(), 'N')
            for idx in token_ids
            if idx.item() not in (self.cls_token_id, self.mask_token_id)
        )

    def reverse_complement(self, sequence: str) -> str:
        """Compute reverse complement of a DNA string."""
        return ''.join(
            self.COMPLEMENT.get(c, 'N') for c in reversed(sequence.upper())
        )

    def mask_tokens(
        self,
        input_ids: torch.Tensor,
        mlm_probability: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for MLM pre-training.

        Following BERT: 80% [MASK], 10% random, 10% unchanged.

        Returns:
            (masked_input_ids, labels) where labels=-100 for unmasked positions
        """
        labels = input_ids.clone()
        masked_input = input_ids.clone()

        # Create masking probability tensor
        prob_matrix = torch.full(input_ids.shape, mlm_probability)

        # Don't mask special tokens
        special_mask = (input_ids == self.cls_token_id) | (input_ids == self.pad_token_id)
        prob_matrix.masked_fill_(special_mask, 0.0)

        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & masked_indices
        masked_input[indices_replaced] = self.mask_token_id

        # 10% -> random nucleotide
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(0, 4, input_ids.shape)  # Only A, C, G, T
        masked_input[indices_random] = random_tokens[indices_random]

        # 10% -> unchanged (already handled by not modifying)

        return masked_input, labels

    def translate_codon(self, codon: str) -> str:
        """Translate a 3-nucleotide codon to amino acid (standard genetic code)."""
        codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        return codon_table.get(codon.upper(), 'X')

    def translate_sequence(self, sequence: str, frame: int = 0) -> str:
        """
        Translate a DNA sequence to protein in a given reading frame.

        This demonstrates the "reading frame as byte alignment" concept:
        the same sequence produces completely different proteins in
        different frames.
        """
        seq = sequence.upper()[frame:]
        protein = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            aa = self.translate_codon(codon)
            if aa == '*':
                break
            protein.append(aa)
        return ''.join(protein)
