"""
Benchmark Rosetta against published genomic foundation models.

Uses the probing protocol: freeze Rosetta weights, extract mean-pooled
embeddings, train a logistic regression classifier. Reports MCC.

Includes a random-initialization baseline to verify pretraining matters
(per "Genomic Foundationless Models" paper, bioRxiv 2024).

Benchmarks:
  1. NT downstream tasks (enhancers 200bp, promoters 300bp center-cropped)
  2. Human non-TATA promoters from genomic-benchmarks (251bp)

Usage:
    python benchmark.py                          # Run all benchmarks
    python benchmark.py --tasks enhancers        # Run specific task
    python benchmark.py --checkpoint path.pt     # Custom checkpoint
"""

import argparse
import time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.rosetta.config import RosettaConfig
from src.rosetta.model import RosettaTransformer
from src.data.tokenizer import DNATokenizer


# =============================================================================
# Published baselines (MCC × 100 where available, accuracy otherwise)
# =============================================================================

BASELINES = {
    'enhancers': {
        'metric': 'MCC',
        'NT-500M-human': 54.97,
        'NT-2500M-multi': 60.57,
        'DNABERT-2': 55.00,
        'HyenaDNA': None,
    },
    'enhancers_types': {
        'metric': 'MCC',
        'NT-500M-human': 41.97,
        'NT-2500M-multi': 46.25,
        'DNABERT-2': 44.00,
        'HyenaDNA': None,
    },
    'promoter_all': {
        'metric': 'MCC',
        'NT-500M-human': 87.71,
        'NT-2500M-multi': 91.01,
        'DNABERT-2': 86.77,
        'HyenaDNA': None,
    },
    'promoter_tata': {
        'metric': 'MCC',
        'NT-500M-human': 90.75,
        'NT-2500M-multi': 94.00,
        'DNABERT-2': 94.27,
        'HyenaDNA': None,
    },
    'promoter_no_tata': {
        'metric': 'MCC',
        'NT-500M-human': 78.07,
        'NT-2500M-multi': 79.43,
        'DNABERT-2': 71.59,
        'HyenaDNA': None,
    },
    'human_nontata_promoters': {
        'metric': 'accuracy',
        'CNN': 84.6,
        'DNABERT': 85.6,
        'HyenaDNA': 96.6,
    },
}

# Tasks that fit in 256nt or can be center-cropped
TASK_CONFIGS = {
    'enhancers': {'source': 'nt', 'max_len': 200, 'n_classes': 2},
    'enhancers_types': {'source': 'nt', 'max_len': 200, 'n_classes': 3},
    'promoter_all': {'source': 'nt', 'max_len': 300, 'n_classes': 2},
    'promoter_tata': {'source': 'nt', 'max_len': 300, 'n_classes': 2},
    'promoter_no_tata': {'source': 'nt', 'max_len': 300, 'n_classes': 2},
    'human_nontata_promoters': {'source': 'genomic_benchmarks', 'max_len': 251, 'n_classes': 2},
}


# =============================================================================
# Data loading
# =============================================================================

def load_nt_task(task_name: str) -> tuple[list[str], list[int]]:
    """Load a task from the NT downstream tasks dataset."""
    from datasets import load_dataset

    print(f"  Loading NT task '{task_name}' from HuggingFace...")
    ds = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        split="train",
    )
    # Filter to this task
    task_data = ds.filter(lambda x: x["task"] == task_name)

    sequences = task_data["sequence"]
    labels = task_data["label"]

    print(f"  Loaded {len(sequences)} samples")
    return sequences, labels


def load_genomic_benchmark(task_name: str) -> tuple[list[str], list[int]]:
    """Load a task from the genomic-benchmarks HuggingFace dataset."""
    from datasets import load_dataset

    print(f"  Loading genomic benchmark '{task_name}' from HuggingFace...")
    ds = load_dataset("InstaDeepAI/human_nontata_promoters")

    sequences = []
    labels = []
    for split in ds:
        for item in ds[split]:
            sequences.append(item["sequence"])
            labels.append(item["label"])

    print(f"  Loaded {len(sequences)} samples")
    return sequences, labels


def load_task_data(task_name: str) -> tuple[list[str], list[int]]:
    """Load data for a benchmark task."""
    config = TASK_CONFIGS[task_name]
    if config['source'] == 'nt':
        return load_nt_task(task_name)
    elif config['source'] == 'genomic_benchmarks':
        return load_genomic_benchmark(task_name)
    else:
        raise ValueError(f"Unknown source: {config['source']}")


# =============================================================================
# Embedding extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    model: RosettaTransformer,
    sequences: list[str],
    tokenizer: DNATokenizer,
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract mean-pooled embeddings from frozen Rosetta model.

    For sequences longer than max_len, center-crop.
    For shorter sequences, pad with N tokens.
    """
    model.eval()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize with center-cropping
        token_batch = []
        for seq in batch_seqs:
            seq = seq.upper().replace('U', 'T')
            # Center-crop if too long
            if len(seq) > max_len:
                start = (len(seq) - max_len) // 2
                seq = seq[start:start + max_len]
            tokens = tokenizer.encode(seq, max_length=max_len)
            # Pad if too short
            if len(tokens) < max_len:
                pad = torch.full((max_len - len(tokens),), tokenizer.pad_token_id)
                tokens = torch.cat([tokens, pad])
            token_batch.append(tokens)

        input_ids = torch.stack(token_batch).to(device)

        # Extract hidden states
        hidden = model.encode(input_ids)  # (batch, seq_len, d_model)

        # Mean pooling (exclude padding)
        # For simplicity, mean over all positions
        embeddings = hidden.mean(dim=1)  # (batch, d_model)
        all_embeddings.append(embeddings.cpu().numpy())

        if (i // batch_size) % 20 == 0 and i > 0:
            print(f"    {i}/{len(sequences)} sequences embedded...")

    return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_probing(
    embeddings: np.ndarray,
    labels: list[int],
    n_classes: int,
) -> dict:
    """
    Train logistic regression on embeddings, evaluate with 5-fold CV.
    Returns MCC and accuracy.
    """
    labels_arr = np.array(labels)

    # Pipeline: scale features, then logistic regression
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                   multi_class='multinomial' if n_classes > 2 else 'auto')),
    ])

    # 5-fold cross-validation predictions
    y_pred = cross_val_predict(clf, embeddings, labels_arr, cv=5)

    mcc = matthews_corrcoef(labels_arr, y_pred)
    acc = accuracy_score(labels_arr, y_pred)

    return {
        'mcc': mcc,
        'mcc_100': mcc * 100,
        'accuracy': acc,
        'accuracy_100': acc * 100,
        'n_samples': len(labels),
        'n_classes': n_classes,
    }


# =============================================================================
# Main benchmark runner
# =============================================================================

def run_benchmark(
    task_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Run a single benchmark task with pretrained and random baselines."""
    config = TASK_CONFIGS[task_name]
    max_len = min(config['max_len'], 256)

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {task_name}")
    print(f"{'=' * 60}")

    # Load data
    sequences, labels = load_task_data(task_name)

    # Load tokenizer
    tokenizer = DNATokenizer(max_length=max_len)

    # --- Pretrained model ---
    print(f"\n  Extracting embeddings (pretrained)...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['config']
    model = RosettaTransformer(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    t0 = time.time()
    embeddings = extract_embeddings(model, sequences, tokenizer, device, max_len)
    embed_time = time.time() - t0
    print(f"  Embedded {len(sequences)} sequences in {embed_time:.1f}s")

    results_pretrained = evaluate_probing(embeddings, labels, config['n_classes'])
    print(f"  Pretrained: MCC={results_pretrained['mcc_100']:.2f} Acc={results_pretrained['accuracy_100']:.2f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Random baseline (untrained model, same architecture) ---
    print(f"\n  Extracting embeddings (random init baseline)...")
    random_model = RosettaTransformer(model_config).to(device)
    random_model.eval()

    embeddings_random = extract_embeddings(random_model, sequences, tokenizer, device, max_len)
    results_random = evaluate_probing(embeddings_random, labels, config['n_classes'])
    print(f"  Random init: MCC={results_random['mcc_100']:.2f} Acc={results_random['accuracy_100']:.2f}%")

    del random_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Compare to published baselines ---
    print(f"\n  Comparison:")
    baselines = BASELINES.get(task_name, {})
    metric_name = baselines.get('metric', 'MCC')
    our_score = results_pretrained['mcc_100'] if metric_name == 'MCC' else results_pretrained['accuracy_100']
    random_score = results_random['mcc_100'] if metric_name == 'MCC' else results_random['accuracy_100']

    print(f"    {'Model':<30s} {metric_name:>8s}")
    print(f"    {'-'*40}")
    print(f"    {'Rosetta (pretrained)':<30s} {our_score:>8.2f}")
    print(f"    {'Rosetta (random init)':<30s} {random_score:>8.2f}")

    for model_name, score in baselines.items():
        if model_name == 'metric':
            continue
        if score is not None:
            print(f"    {model_name:<30s} {score:>8.2f}")

    return {
        'task': task_name,
        'pretrained': results_pretrained,
        'random_init': results_random,
        'baselines': baselines,
        'pretrained_beats_random': our_score > random_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Rosetta against published models")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task names or 'all'")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rosetta_best.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tasks == "all":
        tasks = list(TASK_CONFIGS.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    print("=" * 60)
    print("ROSETTA BENCHMARK SUITE")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Tasks:      {tasks}")

    all_results = []
    for task in tasks:
        if task not in TASK_CONFIGS:
            print(f"\n  Unknown task: {task}, skipping")
            continue
        try:
            result = run_benchmark(task, args.checkpoint, device)
            all_results.append(result)
        except Exception as e:
            print(f"\n  FAILED on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}\n")
    print(f"  {'Task':<30s} {'Rosetta':>8s} {'Random':>8s} {'Delta':>8s} {'Best Published':>15s}")
    print(f"  {'-' * 75}")

    for r in all_results:
        task = r['task']
        baselines = r['baselines']
        metric = baselines.get('metric', 'MCC')

        our = r['pretrained']['mcc_100'] if metric == 'MCC' else r['pretrained']['accuracy_100']
        rand = r['random_init']['mcc_100'] if metric == 'MCC' else r['random_init']['accuracy_100']
        delta = our - rand

        # Best published
        best_name = ""
        best_score = 0
        for k, v in baselines.items():
            if k == 'metric' or v is None:
                continue
            if v > best_score:
                best_score = v
                best_name = k

        marker = "+" if delta > 0 else ""
        print(f"  {task:<30s} {our:>8.2f} {rand:>8.2f} {marker}{delta:>7.2f} {best_score:>8.2f} ({best_name})")

    pretraining_helps = sum(1 for r in all_results if r['pretrained_beats_random'])
    print(f"\n  Pretraining beats random init: {pretraining_helps}/{len(all_results)} tasks")


if __name__ == "__main__":
    main()
