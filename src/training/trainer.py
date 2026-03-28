"""
Training loop for the Rosetta transformer.

Supports:
1. Masked Language Model (MLM) pre-training
2. Frame prediction auxiliary task
3. Wobble-aware loss weighting
4. Gradient accumulation
5. Learning rate warmup + cosine decay
6. Checkpointing
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from pathlib import Path
from typing import Optional

from ..rosetta.model import RosettaTransformer
from ..rosetta.config import RosettaConfig


class RosettaTrainer:
    """Training loop for Rosetta."""

    def __init__(
        self,
        model: RosettaTransformer,
        config: RosettaConfig,
        train_dataset,
        val_dataset=None,
        output_dir: str = "checkpoints",
        train_sampler: Optional[Sampler] = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # AMP (automatic mixed precision)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Data loaders (sampler and shuffle are mutually exclusive)
        nw = config.num_workers
        use_pin = (self.device.type == "cuda" and nw > 0)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=use_pin,
            persistent_workers=nw > 0,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=use_pin,
                persistent_workers=nw > 0,
            )

        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        param_groups = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        self.last_grad_norm = 0.0
        self._nan_streak = 0
        self._max_nan_streak = 50  # reset scaler after this many consecutive NaN batches

    def get_lr(self, step: int) -> float:
        """Learning rate with warmup + cosine decay."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self, num_epochs: int = 10, log_interval: int = 50):
        """Run the training loop."""
        print(f"Training Rosetta on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps} accumulation")
        if self.use_amp:
            print(f"Mixed precision: enabled (FP16)")
        print()

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_steps = 0
            t0 = time.time()

            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                frame_labels = batch.get('frame_labels')
                if frame_labels is not None:
                    frame_labels = frame_labels.to(self.device)
                conservation_targets = batch.get('conservation_targets')
                if conservation_targets is not None:
                    conservation_targets = conservation_targets.to(self.device)

                # Forward (with optional AMP)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        frame_labels=frame_labels,
                        conservation_targets=conservation_targets,
                        global_step=self.global_step,
                    )
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps

                # NaN detection with scaler recovery
                if torch.isnan(loss) or torch.isinf(loss):
                    self._nan_streak += 1
                    if self._nan_streak <= 3 or self._nan_streak % 50 == 0:
                        print(f"  WARNING: NaN/Inf loss at step {self.global_step} "
                              f"(streak: {self._nan_streak})", flush=True)
                    if self._nan_streak >= self._max_nan_streak:
                        # GradScaler is stuck — reset it and halve LR to recover
                        print(f"  RECOVERING: {self._nan_streak} consecutive NaN batches. "
                              f"Resetting GradScaler, halving LR.", flush=True)
                        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
                        for pg in self.optimizer.param_groups:
                            pg['lr'] *= 0.5
                        self._nan_streak = 0
                    self.optimizer.zero_grad()
                    continue

                self._nan_streak = 0  # reset on valid batch

                # Backward (scaler handles FP16 gradient scaling)
                self.scaler.scale(loss).backward()

                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Unscale before clipping so grad norms are in FP32 scale
                    self.scaler.unscale_(self.optimizer)
                    self.last_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()

                    # Update learning rate
                    lr = self.get_lr(self.global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += outputs['loss'].item()
                epoch_steps += 1

                if epoch_steps % log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    elapsed = time.time() - t0
                    tokens_per_sec = (
                        epoch_steps * self.config.batch_size * input_ids.shape[1]
                    ) / elapsed
                    print(
                        f"  Epoch {epoch+1} | Step {epoch_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {self.get_lr(self.global_step):.2e} | "
                        f"GradNorm: {self.last_grad_norm:.4f} | "
                        f"Tok/s: {tokens_per_sec:.0f}",
                        flush=True,
                    )

                    # Monitoring: prediction distribution (MLM) or RTD accuracy (ELECTRA)
                    with torch.no_grad():
                        if 'rtd_logits' in outputs:
                            # ELECTRA mode
                            rtd_preds = (outputs['rtd_logits'].sigmoid() > 0.5).float()
                            rtd_acc = (rtd_preds == outputs['rtd_labels']).float().mean() * 100
                            replaced_frac = outputs['rtd_labels'].mean() * 100
                            print(f"    RTD acc: {rtd_acc:.1f}% | replaced: {replaced_frac:.1f}% | "
                                  f"disc: {outputs['disc_loss'].item():.4f} | "
                                  f"gen: {outputs['gen_loss'].item():.4f}", flush=True)
                        else:
                            preds = outputs['logits'].argmax(dim=-1)
                            pred_mask = (labels != -100)
                            if pred_mask.any():
                                pred_dist = torch.bincount(preds[pred_mask].view(-1).clamp(0, 3), minlength=4)[:4]
                                pred_pct = pred_dist.float() / pred_dist.sum().clamp(min=1) * 100
                                print(f"    Pred dist: A:{pred_pct[0]:.0f}% C:{pred_pct[1]:.0f}% "
                                      f"G:{pred_pct[2]:.0f}% T:{pred_pct[3]:.0f}%", flush=True)

            # Epoch summary
            avg_loss = epoch_loss / max(epoch_steps, 1)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{num_epochs} complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Embedding diversity check (catches collapse early)
            embed_sim = self._check_embedding_diversity()
            sim_status = " [WARNING: possible collapse]" if embed_sim > 0.95 else ""
            print(f"  Embedding cosine sim: {embed_sim:.4f}{sim_status}", flush=True)

            # Validation
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"  Validation Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"  Early stopping: no improvement for {self.patience} epochs")
                        self.save_checkpoint(f"epoch_{epoch+1}")
                        break

            # Save periodic checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0
        total_steps = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            total_steps += 1

        self.model.train()
        return total_loss / max(total_steps, 1)

    @torch.no_grad()
    def _check_embedding_diversity(self, n_samples: int = 256) -> float:
        """Mean pairwise cosine similarity of embeddings. >0.95 = collapse."""
        self.model.eval()
        embeddings = []
        for i, batch in enumerate(self.train_loader):
            if i * self.config.batch_size >= n_samples:
                break
            input_ids = batch['input_ids'].to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                hidden = self.model.encode(input_ids)
            embeddings.append(hidden.mean(dim=1))  # mean pool -> (batch, d_model)
        embeddings = F.normalize(torch.cat(embeddings, dim=0)[:n_samples], dim=1)
        sim = embeddings @ embeddings.T
        n = sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        self.model.train()
        return sim[mask].mean().item()

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.output_dir / f"rosetta_{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
