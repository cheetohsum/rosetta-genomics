"""
Rosetta: A Multi-Frame, RC-Equivariant Genomic Transformer

The "key" to DNA is simultaneous awareness of:
  1. Strand direction (5'->3' vs 3'->5')
  2. Reading frame (offset 0, 1, 2)
  3. Codon position significance (identity vs wobble)

Rosetta learns all three layers in a unified architecture.
"""

from .model import RosettaTransformer
from .config import RosettaConfig

__all__ = ["RosettaTransformer", "RosettaConfig"]
