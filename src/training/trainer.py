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
from torch.utils.data import DataLoader
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

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
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

                # Forward
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    frame_labels=frame_labels,
                )
                loss = outputs['loss'] / self.config.gradient_accumulation_steps

                # Backward
                loss.backward()

                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Update learning rate
                    lr = self.get_lr(self.global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                    self.optimizer.step()
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
                        f"Tok/s: {tokens_per_sec:.0f}"
                    )

            # Epoch summary
            avg_loss = epoch_loss / max(epoch_steps, 1)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{num_epochs} complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Validation
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"  Validation Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")

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

            outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            total_steps += 1

        self.model.train()
        return total_loss / max(total_steps, 1)

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
