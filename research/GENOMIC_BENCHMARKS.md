# Genomic Foundation Model Benchmarks - Comprehensive Research

## 1. Standard Benchmark Suites

### 1.1 GUE (Genome Understanding Evaluation)
- **Paper:** DNABERT-2 (ICLR 2024) - arXiv:2306.15006
- **Composition:** 36 datasets across 9 tasks, 4 species
- **Download:** https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view
- **HuggingFace mirror:** https://huggingface.co/datasets/leannmlindsey/GUE
- **GitHub:** https://github.com/MAGICS-LAB/DNABERT_2

#### GUE Task Summary

| Task | Species | # Datasets | Seq Length | Classes | Metric |
|------|---------|-----------|------------|---------|--------|
| Core Promoter Detection (CPD) | Human | 3 (tata, notata, all) | 70 bp | 2 | MCC |
| Promoter Detection (PD) | Human | 3 (tata, notata, all) | 300 bp | 2 | MCC |
| Transcription Factor Prediction (TF-H) | Human | 5 (TF0-TF4) | 100 bp | 2 | MCC |
| Transcription Factor Prediction (TF-M) | Mouse | 5 (TF0-TF4) | 100 bp | 2 | MCC |
| Splice Site Prediction (SSP) | Human | 1 | 400 bp | 3 | MCC |
| Epigenetic Marks Prediction (EMP) | Yeast | 10 (H3, H4, H3K9ac, etc.) | 500 bp | 2 | MCC |
| COVID Variant Classification (CVC) | Virus | 1 | 1000 bp | 9 | MCC |
| **GUE+ Extensions:** | | | | | |
| Species Classification | Fungi | 1 | 5000 bp | multi | Accuracy |
| Species Classification | Virus | 1 | 5000 bp | multi | Accuracy |
| Enhancer-Promoter Interaction | Human | 1 | 10000 bp | 2 | Accuracy |

**Primary metric:** Matthews Correlation Coefficient (MCC); also F1-Score
**Evaluation protocol:** Full fine-tuning with linear classification head on [CLS] token

---

### 1.2 Nucleotide Transformer (NT) Benchmark - 18 Tasks
- **Paper:** Nature Methods (2024) - doi:10.1038/s41592-024-02523-z
- **Download:** https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
- **Revised version:** https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
- **GitHub:** https://github.com/instadeepai/nucleotide-transformer

#### NT 18 Task Details

| Task | Train | Test | Classes | Seq Len | Source |
|------|-------|------|---------|---------|--------|
| promoter_all | 53,276 | 5,920 | 2 | 300 bp | DeePromoter |
| promoter_tata | 5,509 | 621 | 2 | 300 bp | DeePromoter |
| promoter_no_tata | 47,767 | 5,299 | 2 | 300 bp | DeePromoter |
| enhancers | 14,968 | 400 | 2 | 200 bp | Enhancer Prediction |
| enhancers_types | 14,968 | 400 | 3 | 200 bp | Enhancer Prediction |
| splice_sites_all | 27,000 | 3,000 | 3 | 400 bp | SpliceFinder |
| splice_sites_acceptor | 19,961 | 2,218 | 2 | 600 bp | Spliceator |
| splice_sites_donor | 19,775 | 2,198 | 2 | 600 bp | Spliceator |
| H3 | 13,468 | 1,497 | 2 | 500 bp | Histone Modification |
| H4 | 13,140 | 1,461 | 2 | 500 bp | Histone Modification |
| H3K9ac | 25,003 | 2,779 | 2 | 500 bp | Histone Modification |
| H3K14ac | 29,743 | 3,305 | 2 | 500 bp | Histone Modification |
| H4ac | 30,685 | 3,410 | 2 | 500 bp | Histone Modification |
| H3K4me1 | 28,509 | 3,168 | 2 | 500 bp | Histone Modification |
| H3K4me2 | 27,614 | 3,069 | 2 | 500 bp | Histone Modification |
| H3K4me3 | 33,119 | 3,680 | 2 | 500 bp | Histone Modification |
| H3K36me3 | 31,392 | 3,488 | 2 | 500 bp | Histone Modification |
| H3K79me3 | 25,953 | 2,884 | 2 | 500 bp | Histone Modification |

**Primary metric:** MCC (Matthews Correlation Coefficient)
**Evaluation protocol:** Both probing (frozen embeddings + logistic regression/MLP) and full fine-tuning
**Key result:** NT-2.5B-multispecies compound MCC = 0.709 (fine-tuning), beating baseline 0.674

---

### 1.3 Genomic Benchmarks (Grevsova et al., BMC Genomic Data 2023)
- **Paper:** doi:10.1186/s12863-023-01123-8
- **Download:** `pip install genomic-benchmarks`
- **GitHub:** https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
- **HuggingFace:** https://huggingface.co/katarinagresova

8 datasets focusing on regulatory elements from human, mouse, roundworm:
- demo_coding_vs_intergenomic_seqs
- demo_human_or_worm
- dummy_mouse_enhancers_ensembl
- human_enhancers_cohn
- human_enhancers_ensembl
- human_ensembl_regulatory
- human_nontata_promoters
- human_ocr_ensembl

---

### 1.4 GenBench (NeurIPS 2024)
- **Paper:** arXiv:2406.01627
- **GitHub:** https://github.com/jimmylihui/GenBench

10 models, 43 datasets across short-range and long-range tasks.

**Short-range datasets (38, <1000 bp):**
Mouse Enhancers, Coding vs Intergenomic, Human vs Worm, Human Enhancers Cohn/Ensembl,
Human Ensembl Regulatory, Human Nontata Promoters, Human OCR Ensembl,
Drosophila Enhancers (regression), Human Core Promoter Detection,
Human TF Prediction, Promoter Detection, Splice Site Detection,
Mouse TF Prediction, Yeast Epigenetic Marks, COVID Variant Classification

**Long-range datasets (5, >1000 bp):**
Splice Site Prediction (15K), Species Classification (80M), Promoters (8K),
Genomic Structure Prediction (256M), Bulk RNA Prediction (196K)

**Metrics:** Accuracy, AUC-ROC, Pearson/Spearman correlation, MSE
**Protocol:** Full fine-tuning with AdamW, 100 epochs, cosine LR decay

---

### 1.5 DNALongBench (Nature Communications 2025)
- **Paper:** doi:10.1038/s41467-025-65077-4
- 5 long-range tasks up to 1M bp:
  1. Enhancer-target gene interaction
  2. Expression quantitative trait loci (eQTL)
  3. 3D genome organization
  4. Regulatory sequence activity
  5. Transcription initiation signals

Key finding: Foundation models still lag behind expert models for long-range dependencies.

---

### 1.6 GUANinE (MLCB 2023, updated v1.1 Dec 2025)
- **GitHub:** https://github.com/ni-lab/guanine
- Focus: Sequence-to-function tasks
- Tasks: DNase accessibility, cCRE histone modifications, sequence conservation
- Finding: Supervised S2F models excel at chromatin/histone tasks; LMs excel at conservation

---

### 1.7 OmniGenBench (May 2025)
- **Paper:** arXiv:2505.14402
- **Website:** https://omnigenbench.com/
- **GitHub:** https://github.com/yangheng95/OmniGenBench
- 123+ datasets, 58+ metrics, 31+ models (DNA and RNA)
- Automated one-command evaluation pipeline

---

## 2. Comprehensive Benchmark Results (Feng et al., Nature Communications 2025)

This is the most comprehensive head-to-head comparison published to date.
**Paper:** doi:10.1038/s41467-025-65823-8
**Protocol:** Zero-shot embeddings with frozen weights + random forest classifier (mean token pooling)
**57 total datasets**

### 2.1 Human Genome Sequence Classification (AUC)

| Task | DNABERT-2 | NT-v2 | HyenaDNA | Caduceus-Ph | GROVER |
|------|-----------|-------|----------|-------------|--------|
| DNase I Hypersensitive | 0.867 | 0.852 | 0.830 | **0.880** | 0.857 |
| Human TFBS 1 | 0.838 | 0.832 | 0.830 | **0.880** | 0.862 |
| Human TFBS 2 | 0.821 | 0.809 | 0.821 | **0.869** | 0.850 |
| Human TFBS 3 | 0.790 | 0.797 | 0.788 | **0.825** | 0.816 |
| Human TFBS 4 | 0.726 | 0.710 | 0.715 | **0.773** | 0.763 |
| Human TFBS 5 | 0.920 | 0.915 | 0.916 | 0.929 | **0.931** |
| Promoter GM12878 | 0.986 | 0.984 | 0.976 | **0.987** | 0.984 |
| Promoter HUVEC | **0.990** | 0.987 | 0.982 | 0.990 | 0.989 |
| Promoter Hela-S3 | **0.989** | 0.984 | 0.981 | 0.987 | 0.986 |
| Promoter NHEK | 0.950 | 0.932 | 0.927 | **0.957** | 0.951 |
| Splice Acceptor | **0.897** | 0.793 | 0.795 | 0.845 | 0.804 |
| Coding Region | 0.944 | 0.929 | 0.941 | **0.974** | 0.959 |
| Splice Donor | **0.906** | 0.820 | 0.813 | 0.854 | 0.819 |
| Enhancer | **0.872** | 0.867 | 0.834 | 0.838 | 0.855 |
| Enhancer Cohn | **0.822** | 0.789 | 0.775 | 0.821 | 0.816 |
| Enhancer Ensembl | 0.937 | 0.939 | 0.936 | **0.943** | 0.938 |
| Open Chromatin Region | 0.725 | 0.718 | 0.719 | **0.765** | 0.746 |
| Promoter All 300 bps | 0.943 | 0.945 | 0.939 | **0.952** | 0.940 |
| Promoter All 70 bps | 0.831 | 0.853 | 0.832 | **0.875** | 0.851 |
| Promoter NonTATA 251 | 0.930 | 0.891 | 0.928 | **0.943** | 0.940 |
| Promoter NonTATA 300 | 0.977 | 0.976 | 0.966 | **0.983** | 0.973 |
| Promoter NonTATA 70 | 0.853 | 0.873 | 0.852 | **0.896** | 0.870 |
| Promoter TATA 300 | 0.765 | 0.779 | **0.808** | 0.760 | 0.780 |
| Promoter TATA 70 | 0.778 | 0.795 | 0.783 | **0.810** | 0.796 |

### 2.2 Multi-Species Genome Classification (AUC)

| Task | DNABERT-2 | NT-v2 | HyenaDNA | Caduceus-Ph | GROVER |
|------|-----------|-------|----------|-------------|--------|
| Arabidopsis NonTATA | 0.946 | 0.940 | **0.955** | 0.944 | 0.949 |
| Arabidopsis TATA | 0.951 | 0.950 | **0.961** | 0.937 | 0.949 |
| B.Amyloliquefaciens | 0.852 | 0.823 | 0.864 | **0.869** | 0.862 |
| R.Capsulatus | 0.686 | 0.675 | 0.712 | 0.670 | **0.715** |
| Human vs Worm | 0.980 | 0.979 | 0.950 | **0.992** | 0.984 |
| Mouse TFBS 1 | **0.711** | 0.704 | 0.590 | 0.684 | 0.695 |
| Mouse TFBS 2 | 0.907 | 0.901 | 0.900 | **0.947** | 0.909 |
| Mouse TFBS 3 | 0.931 | 0.927 | 0.894 | **0.935** | 0.933 |
| Mouse TFBS 4 | **0.762** | 0.694 | 0.588 | 0.705 | 0.682 |
| Mouse TFBS 5 | 0.678 | 0.708 | 0.627 | **0.715** | 0.682 |

### 2.3 Epigenetic Modification Detection (AUC)

| Task | DNABERT-2 | NT-v2 | HyenaDNA | Caduceus-Ph | GROVER |
|------|-----------|-------|----------|-------------|--------|
| Human 5mC | 0.685 | 0.738 | 0.684 | **0.783** | 0.744 |
| Human 6mA | 0.735 | 0.751 | 0.738 | **0.773** | 0.767 |
| Yeast H3 | 0.914 | 0.895 | 0.900 | **0.929** | 0.906 |
| Yeast H3K14ac | **0.760** | 0.741 | 0.707 | 0.730 | 0.730 |
| Yeast H3K36me3 | **0.799** | 0.785 | 0.740 | 0.766 | 0.753 |
| Yeast H3K4me1 | **0.731** | 0.712 | 0.699 | 0.707 | 0.696 |
| Yeast H3K4me2 | **0.708** | 0.685 | 0.685 | 0.690 | 0.694 |
| Yeast H3K4me3 | **0.681** | 0.660 | 0.649 | 0.660 | 0.668 |
| Yeast H3K79me3 | **0.857** | 0.844 | 0.822 | 0.845 | 0.843 |
| Yeast H3K9ac | **0.792** | 0.769 | 0.756 | 0.778 | 0.769 |
| Yeast H4 | **0.931** | 0.910 | 0.898 | 0.930 | 0.908 |
| Yeast H4ac | **0.747** | 0.726 | 0.698 | 0.724 | 0.718 |

### 2.4 Gene Expression Prediction (Pearson Correlation)

| Model | Input Length | Avg Correlation |
|-------|-------------|-----------------|
| DNABERT-2 | 6000 bp | 0.121 |
| NT-v2 | 6000 bp | 0.122 |
| HyenaDNA | 6000 bp | 0.122 |
| Caduceus-Ph | 6000 bp | 0.123 |
| GROVER | 2048 bp | 0.114 |
| HyenaDNA-450K | 196K bp | **0.137** |
| Enformer | 196K bp | 0.129 |

### 2.5 Pathogenic Variant Classification (AUC)

| Model | AUC |
|-------|-----|
| NT-v2 | **0.732** |
| Caduceus-Ph | 0.696 |
| Enformer hidden | 0.688 |
| Sei output | 0.664 |
| Sei hidden | 0.660 |
| HyenaDNA-450K | 0.626 |
| HyenaDNA | 0.612 |
| GROVER | 0.603 |
| DNABERT-2 | 0.538 |

### 2.6 QTL Prediction (AUC)

| Model | eQTL | sQTL | paQTL | ipaQTL |
|-------|------|------|-------|--------|
| AlphaGenome | **0.803** | **0.715** | **0.754** | **0.864** |
| Enformer hidden | 0.774 | 0.666 | 0.674 | 0.692 |
| Caduceus-Ph | 0.649 | 0.567 | 0.508 | 0.568 |
| HyenaDNA | 0.612 | 0.553 | 0.470 | 0.448 |
| NT-v2 | 0.609 | 0.505 | 0.525 | 0.602 |
| GROVER | 0.590 | 0.474 | 0.449 | 0.476 |
| DNABERT-2 | 0.570 | 0.580 | 0.507 | 0.469 |

---

## 3. GenBench Results (Short-Range, Top-1 Accuracy)

| Dataset | HyenaDNA | DNABERT | DNABERT-2 | GENA-LM | NT | Caduceus |
|---------|----------|---------|-----------|---------|------|----------|
| Mouse Enhancers | 0.793 | 0.810 | 0.818 | 0.830 | **0.851** | 0.816 |
| Coding vs Intergenomic | 0.910 | 0.936 | 0.936 | 0.932 | **0.958** | 0.937 |
| Human vs Worm | 0.962 | 0.958 | 0.974 | 0.970 | **0.975** | 0.956 |
| Splice Site Detection | 0.566 | 0.872 | 0.881 | 0.918 | **0.948** | 0.567 |
| COVID Variant | 0.377 | 0.599 | **0.720** | 0.703 | 0.694 | 0.379 |

---

## 4. DNABERT-2 GUE Results Summary

| Task | DNABERT-2 | NT-2500M-multi | DNABERT (3-mer) |
|------|-----------|----------------|-----------------|
| Epigenetic Marks (Yeast) | 55.98 | 58.06 | - |
| TF Prediction (Mouse) | 67.99 | 67.01 | - |
| COVID Variant | 71.02 | 73.04 | 62.23 |
| TF Prediction (Human) | 70.10 | 63.32 | - |
| Promoter Detection | 84.21 | 88.14 | - |
| Core Promoter Detection | 70.52 | 71.62 | - |
| Splice Site Prediction | 84.99 | 89.36 | - |
| **Overall Average** | **66.80** | **66.93** | **61.62** |

DNABERT-2 (117M params) achieves comparable to NT-2500M (2537M params) with 21x fewer params.

---

## 5. Metrics Summary

| Metric | Used By | Task Types |
|--------|---------|------------|
| **MCC** (Matthews Correlation Coefficient) | GUE, NT tasks | Binary/multi-class classification |
| **AUROC** (Area Under ROC) | Feng et al. 2025, GenBench | Binary classification, variant effect |
| **Accuracy** | GenBench, GUE species | Multi-class classification |
| **F1-Score** | GUE, various | Binary/multi-class classification |
| **Pearson Correlation** | Gene expression tasks | Regression |
| **Spearman Correlation** | Expression, VEP | Regression/ranking |
| **MSE** | Expression tasks | Regression |
| **Cohen's d** | Variant effect | Effect size |

---

## 6. Evaluation Protocols

### Protocol A: Full Fine-Tuning (GUE / DNABERT-2 style)
- Unfreeze all model weights
- Add linear classification head on [CLS] token embedding
- Train end-to-end with AdamW, global batch size 32
- Use train/test splits provided by benchmark
- Report MCC or F1

### Protocol B: Probing / Frozen Embeddings (NT style)
- Freeze all model weights
- Extract embeddings (probing with logistic regression or small MLP)
- Cross-validation (10-fold in NT)
- Report MCC

### Protocol C: Zero-Shot Embeddings (Feng et al. 2025 style)
- Freeze all model weights
- Extract mean token embeddings
- Train downstream random forest classifier
- 70:30 train/test split, 5-fold CV for hyperparameters
- Report AUROC
- Statistical significance via DeLong's test (p < 0.01)

### Key Finding on Protocols
- Fine-tuning generally outperforms probing
- Mean token pooling consistently outperforms [CLS] token pooling by 1.4-8.7% AUC
- "Genomic Foundationless Models" (2024 critique) found randomly initialized models can match pretrained ones on many tasks, questioning pretraining value

---

## 7. Models to Benchmark Against (Comprehensive List)

### Established Models (2023-2024)
| Model | Params | Architecture | Context | Tokenization | Training |
|-------|--------|-------------|---------|--------------|----------|
| DNABERT-2 | 117M | Transformer (BERT) | ~3 kb | BPE | MLM, multi-species |
| NT v2 (500M-multi) | 500M | Transformer (BERT) | 12 kb | 6-mer | MLM, multi-species |
| NT v2 (2.5B-multi) | 2.5B | Transformer (BERT) | 12 kb | 6-mer | MLM, multi-species |
| HyenaDNA | 1.6-6.6M | Hyena (conv) | 1kb-1M | Single nucleotide | Autoregressive |
| Caduceus-Ph | ~7M | BiMamba (SSM) | 131K | Single nucleotide | Bidirectional MLM |
| GROVER | ~100M | Transformer (BERT) | 3 kb | BPE | Next-k-mer prediction |
| Evo | 7B | StripedHyena | 131K | Single nucleotide | Autoregressive |

### Newer Models (2025-2026)
| Model | Params | Architecture | Context | Key Innovation |
|-------|--------|-------------|---------|----------------|
| **Evo 2** | 40B | StripedHyena | 1M bp | 9T bp training, zero-shot VEP |
| **AIDO.DNA** | 300M / 7B | Encoder Transformer | 4K | Single-nt tokenization, 796 species |
| **GENERator** | 1.2B | Transformer | 98K | Eukaryotic DNA, generative |
| **NT v3 (NTv3)** | Large | U-Net-like | 1M bp | Joint seq-function, 16K functional tracks |
| **JEPA-DNA** | ~117M | JEPA + BERT | varies | Latent grounding, zero-shot gains |
| **GenomeOcean** | 4B | Generative | varies | Metagenomic, #1 downloaded GFM |
| **AlphaGenome** | Large | DeepMind | 1M bp | 1000s of functional tracks, SOTA VEP |
| **DNABERT-S** | ~117M | Transformer | ~3 kb | Species-aware embeddings |

### Specialized Sequence-to-Function Models (for comparison)
| Model | Type | Context | Strength |
|-------|------|---------|----------|
| Enformer | CNN+Transformer | 196K bp | Gene expression, epigenomics |
| Borzoi | CNN+Transformer | 524K bp | RNA-seq coverage prediction |
| Sei | CNN | 4K bp | Regulatory variant prediction |
| Flashzoi | Borzoi-optimized | 524K bp | 3x faster Borzoi |

---

## 8. Key Takeaways for Rosetta Benchmarking

1. **Minimum benchmark suite:** Run on GUE (28 tasks) and NT-18 tasks for comparability with existing literature.

2. **Evaluation protocol matters enormously:** Report both frozen-embedding probing AND full fine-tuning results, since different models excel under different protocols.

3. **Caduceus-Ph wins on zero-shot embeddings:** In the Feng et al. 2025 study, Caduceus-Ph was the best general-purpose model across most sequence classification tasks when using frozen embeddings.

4. **DNABERT-2 wins on yeast epigenetics:** When using zero-shot embeddings, DNABERT-2 was strongest on yeast histone modification tasks.

5. **All foundation models underperform on long-range tasks:** Expert models (Enformer, Borzoi, AlphaGenome) still dominate gene expression and variant effect prediction.

6. **Critical gap in the field:** The "Genomic Foundationless Models" paper challenges whether pretraining provides real benefit over random initialization + fine-tuning. Rosetta should explicitly test this.

7. **Report mean token pooling results:** This was consistently shown to be superior to [CLS] token pooling across all models.

8. **New SOTA targets:** Evo 2 (40B) and NTv3 (1M context) are the current frontier models to compare against.

---

## Sources

- [DNABERT-2 Paper (ICLR 2024)](https://arxiv.org/abs/2306.15006)
- [Nucleotide Transformer (Nature Methods 2024)](https://www.nature.com/articles/s41592-024-02523-z)
- [Feng et al. Comprehensive Benchmark (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-65823-8)
- [GenBench (NeurIPS 2024)](https://arxiv.org/abs/2406.01627)
- [DNALongBench (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-65077-4)
- [Caduceus (ICML 2024)](https://arxiv.org/abs/2403.03234)
- [Evo 2 (Nature 2026)](https://www.nature.com/articles/s41586-026-10176-5)
- [AIDO.DNA (CMU 2025)](https://www.cs.cmu.edu/~epxing/papers/2025/AIDO.DNA.pdf)
- [GENERator (2025)](https://arxiv.org/abs/2502.07272)
- [NTv3 (InstaDeep Dec 2025)](https://instadeep.com/research/paper/a-foundational-model-for-joint-sequence-function-multi-species-modeling-at-scale-for-long-range-genomic-prediction/)
- [JEPA-DNA (Feb 2026)](https://arxiv.org/abs/2602.17162)
- [GenomeOcean (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11838515/)
- [AlphaGenome (Nature 2025)](https://www.nature.com/articles/s41586-025-10014-0)
- [Genomic Foundationless Models (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.12.18.628606v2)
- [OmniGenBench (2025)](https://arxiv.org/abs/2505.14402)
- [GUANinE v1.1 (Dec 2025)](https://www.biorxiv.org/content/10.64898/2025.12.06.692772v1)
- [Genomic Benchmarks (BMC 2023)](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8)
- [NT Downstream Tasks (HuggingFace)](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
- [GUE Dataset (HuggingFace)](https://huggingface.co/datasets/leannmlindsey/GUE)
