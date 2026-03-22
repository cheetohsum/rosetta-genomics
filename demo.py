"""
Rosetta Demonstration: The Key to DNA's Information Encoding

This demo shows how DNA sequences encode information at multiple
simultaneous levels, and how Rosetta discovers the interpretive key.

Run:
    python demo.py
"""

import torch
from src.rosetta.config import RosettaConfig
from src.rosetta.model import RosettaTransformer
from src.data.tokenizer import DNATokenizer


def demo_endianness():
    """
    Demonstrate DNA's "endianness" -- the same physical DNA
    encodes completely different information depending on
    reading direction and frame.
    """
    print("=" * 70)
    print("DEMO 1: DNA Endianness -- Same Sequence, Different Information")
    print("=" * 70)

    tokenizer = DNATokenizer()

    # A real coding sequence (first 30nt of E. coli lacZ gene)
    seq = "ATGACCATGATTACGCCAAGCTTTGCTGAT"

    print(f"\nForward strand (5'→3'): {seq}")
    print(f"Reverse complement:     {tokenizer.reverse_complement(seq)}")
    print()

    # Translation in all 3 forward frames
    print("Forward reading frames (the 'byte alignment' of DNA):")
    for frame in range(3):
        protein = tokenizer.translate_sequence(seq, frame=frame)
        print(f"  Frame +{frame}: {seq[frame:frame+21]}... → {protein}")

    # Translation in all 3 reverse frames
    rc = tokenizer.reverse_complement(seq)
    print("\nReverse complement reading frames:")
    for frame in range(3):
        protein = tokenizer.translate_sequence(rc, frame=frame)
        print(f"  Frame -{frame}: {rc[frame:frame+21]}... → {protein}")

    print(f"\n→ The SAME 30 nucleotides encode 6 completely different proteins!")
    print(f"→ This is DNA's 'endianness' -- you need the KEY (strand + frame)")
    print(f"   to correctly interpret the information.")


def demo_wobble_position():
    """
    Demonstrate wobble position degeneracy -- the "error correcting code"
    of the genetic code.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Wobble Position -- DNA's Error-Correcting Code")
    print("=" * 70)

    tokenizer = DNATokenizer()

    # Show all codons for Leucine (6 codons!)
    leu_codons = ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG']

    print(f"\nLeucine (L) is encoded by {len(leu_codons)} different codons:")
    for codon in leu_codons:
        print(f"  {codon} → {tokenizer.translate_codon(codon)} (Leucine)")

    print(f"\nInformation per codon position:")
    print(f"  Position 1 (C/T): Determines amino acid CLASS  (~2 bits)")
    print(f"  Position 2 (T):   Determines amino acid IDENTITY (~2 bits)")
    print(f"  Position 3 (X):   'Wobble' -- tolerates mutation (~0.5 bits)")
    print(f"\n→ Position 3 is the 'parity bit' -- mutations here are usually SILENT")
    print(f"→ This is why Rosetta weights position 3 at 0.5x in the loss function")


def demo_model_architecture():
    """
    Demonstrate the Rosetta model architecture and frame key extraction.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Rosetta Architecture -- Finding the Key")
    print("=" * 70)

    # Create a small model for demo
    config = RosettaConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        n_frame_layers=2,
        d_ff=512,
        max_seq_len=256,
    )
    model = RosettaTransformer(config)
    tokenizer = DNATokenizer()

    # Print architecture
    params = model.count_parameters()
    print(f"\nRosetta Architecture:")
    for name, count in params.items():
        bar = "█" * (count // 50000)
        print(f"  {name:30s}: {count:>10,}  {bar}")

    # Encode a sequence
    seq = "ATGACCATGATTACGCCAAGCTTTGCTGATCAGGCGTCCTGCAACTGGTGG"
    input_ids = tokenizer.encode(seq).unsqueeze(0)

    print(f"\nInput sequence ({len(seq)} nt): {seq[:30]}...")
    print(f"Token shape: {input_ids.shape}")

    # Get frame attention map -- THIS IS THE KEY
    with torch.no_grad():
        frame_gates = model.get_frame_attention_map(input_ids)

    print(f"\nFrame gate activations (the 'interpretive key'):")
    print(f"  Shape: {frame_gates.shape} = (batch, seq_len, n_frames)")
    print(f"  Frames: [+0, +1, +2, -0, -1, -2]")
    print()

    # Show gate values for first 10 positions
    gates = frame_gates[0, :10, :].numpy()
    print(f"  Position  | Frame+0 | Frame+1 | Frame+2 | Frame-0 | Frame-1 | Frame-2")
    print(f"  {'-'*9} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")
    for i in range(10):
        values = " | ".join(f"  {v:.3f}" for v in gates[i])
        print(f"  nt {i:5d} | {values}")

    print(f"\n→ After training, the frame gates learn to activate on coding regions")
    print(f"→ The pattern of gate activations across all 6 frames IS the key")
    print(f"→ It reveals: which frames are coding, where genes overlap,")
    print(f"   and how the forward/reverse strands relate")


def demo_generation():
    """
    Demonstrate DNA sequence generation.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: DNA Sequence Generation")
    print("=" * 70)

    config = RosettaConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        n_frame_layers=2,
        d_ff=512,
        max_seq_len=256,
    )
    model = RosettaTransformer(config)
    tokenizer = DNATokenizer()

    # Start with a promoter-like sequence
    prompt = "ATGAAAGCGATTATTGGTCTG"
    prompt_ids = tokenizer.encode(prompt).unsqueeze(0)

    print(f"\nPrompt: {prompt}")
    print(f"Generating 60 new nucleotides...")

    generated = model.generate(
        prompt_ids,
        max_new_tokens=60,
        temperature=0.8,
        top_p=0.95,
        frame_coherent=True,
    )

    full_seq = tokenizer.decode(generated[0])
    new_part = full_seq[len(prompt):]

    print(f"\nGenerated: {prompt}|{new_part}")
    print(f"           {'.' * len(prompt)}^--- new nucleotides start here")

    # Translate the generated sequence
    protein = tokenizer.translate_sequence(full_seq, frame=0)
    print(f"\nTranslation (frame 0): {protein}")
    print(f"\n→ Note: this is an UNTRAINED model, so output is random")
    print(f"→ After training on real genomes, generation follows learned codon")
    print(f"   usage, respects reading frames, and produces realistic sequences")


def demo_chargaff_parity():
    """
    Demonstrate Chargaff's second parity rule -- evidence that DNA
    evolves toward endianness-invariant encoding.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Chargaff's Second Parity Rule")
    print("=" * 70)

    import random
    random.seed(42)

    # Generate a "genome-like" sequence (with Chargaff compliance)
    seq_len = 10000
    seq = []
    for _ in range(seq_len):
        # Biased toward Chargaff compliance
        r = random.random()
        if r < 0.29:
            seq.append('A')
        elif r < 0.50:
            seq.append('C')
        elif r < 0.71:
            seq.append('G')
        else:
            seq.append('T')

    seq_str = ''.join(seq)

    # Count single-strand frequencies
    counts = {nt: seq_str.count(nt) for nt in 'ACGT'}
    total = sum(counts.values())

    print(f"\nSingle-strand nucleotide frequencies ({seq_len} nt):")
    print(f"  A: {counts['A']/total:.3f}  T: {counts['T']/total:.3f}  "
          f"(A≈T: {abs(counts['A']-counts['T'])/total:.4f} difference)")
    print(f"  C: {counts['C']/total:.3f}  G: {counts['G']/total:.3f}  "
          f"(C≈G: {abs(counts['C']-counts['G'])/total:.4f} difference)")

    print(f"\n→ Chargaff's SECOND rule: even on a SINGLE strand, A≈T and C≈G")
    print(f"→ This means the genome is approximately 'endianness-invariant':")
    print(f"   reading the sequence forward or as reverse-complement gives")
    print(f"   statistically similar information content.")
    print(f"→ Evolution pushes genomes toward this symmetry -- it's not random!")


if __name__ == "__main__":
    demo_endianness()
    demo_wobble_position()
    demo_model_architecture()
    demo_generation()
    demo_chargaff_parity()

    print("\n" + "=" * 70)
    print("SYNTHESIS: THE KEY TO DNA")
    print("=" * 70)
    print("""
The "key" to DNA's information encoding is a 3-dimensional interpretive
framework:

  1. STRAND DIRECTION (Endianness)
     → Which of the 2 antiparallel strands to read
     → Determines sense vs antisense transcription
     → Analogous to big-endian vs little-endian byte order

  2. READING FRAME (Byte Alignment)
     → Which of 3 possible codon phases to use
     → Shifting by 1 nucleotide = completely different protein
     → Analogous to memory address alignment

  3. CODON POSITION (Bit Significance)
     → Positions 1-2: amino acid identity (high bits)
     → Position 3: wobble/error tolerance (low bit / parity)
     → Analogous to MSB vs LSB in binary

Rosetta learns all three layers simultaneously through:
  - Multi-frame attention (discovers active reading frames)
  - RC equivariance (enforces strand symmetry)
  - Wobble-aware loss (respects information hierarchy)
  - Hierarchical positional encoding (captures multi-scale structure)

The result: a model that doesn't just process DNA as text, but
understands its STRUCTURE as an information encoding system.
""")
