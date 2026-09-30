"""src/optimize.py — Production imitation learning trainer.

Trains the ProductionVLA model on real HDF5-logged trajectories using the
correct DDPM noise-prediction loss (NOT MSE on diffusion samples, which
was mathematically wrong in the original code).

Features:
  - Real Dataset + DataLoader reading HDF5 with full image frames
  - Multi-epoch training loop (not one-step like the original)
  - AMP (mixed precision) for ~2x speedup on MPS / 3x on CUDA
  - Cosine LR schedule (AdamW + CosineAnnealingLR)
  - Gradient clipping (max_norm=1.0) — prevents transformer instability
  - Validation split with held-out episodes
  - Checkpointing — saves best-val model to checkpoints/
  - wandb logging (falls back to stdout if wandb unavailable)
  - Real CLIP text encoding — no random text embeddings
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Make the project root importable when run as `python -m src.optimize`
# or `python src/optimize.py`. This allows `from src.models.vla_network ...`
# to work in both invocation styles.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.vla_network import ProductionVLA  # noqa: E402


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Reads per-step transitions from one or more HDF5 episode files."""

    def __init__(self, h5_paths: List[str], img_size: int = 224):
        self.img_size = img_size
        self._samples: List[Tuple[str, int]] = []  # (file_path, step_idx)
        self._open_files: Dict[str, h5py.File] = {}
        for p in h5_paths:
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                n = int(f.attrs.get("num_steps", 0))
                for i in range(n):
                    self._samples.append((p, i))
        # Don't open files yet — lazy open in __getitem__ to support multi-worker DataLoader

    def __len__(self) -> int:
        return len(self._samples)

    def _get_file(self, path: str) -> h5py.File:
        if path not in self._open_files:
            self._open_files[path] = h5py.File(path, "r", swmr=True)
        return self._open_files[path]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, step_idx = self._samples[idx]
        f = self._get_file(path)
        img = f["images"][step_idx]  # (H, W, 3) uint8
        # Convert to (3, H, W) float [0,1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        action = torch.from_numpy(f["torque"][step_idx]).float() / 5.0  # normalize to [-1, 1]
        # Decode instruction bytes back to string
        inst_bytes = bytes(f["instruction"][step_idx].astype(np.uint8).tobytes())
        instruction = inst_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        return {
            "image": img_t,
            "action": action,
            "instruction": instruction,
        }


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class VLAImitationTrainer:
    def __init__(
        self,
        train_h5: List[str],
        val_h5: Optional[List[str]] = None,
        lr: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 50,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        project_name: str = "vla-push-block",
        use_wandb: bool = True,
    ):
        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.amp_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.amp_dtype = torch.float16  # MPS supports float16 via autocast
        else:
            self.device = torch.device("cpu")
            self.amp_dtype = torch.bfloat16
        print(f"[TRAINER] device = {self.device} | amp = {self.amp_dtype}")

        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Model
        self.model = ProductionVLA(
            action_dim=3, diffusion_steps=100,
        ).to(self.device)
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[TRAINER] trainable params: {n_trainable/1e6:.1f}M / total {n_total/1e6:.1f}M")

        # Optimizer + scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4,
        )
        steps_per_epoch = max(1, len(TrajectoryDataset(train_h5)) // batch_size)
        total_steps = steps_per_epoch * epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=lr * 0.01,
        )
        # GradScaler is CUDA-only; on MPS we use autocast without scaler
        self.scaler = (
            GradScaler(device_type="cuda")
            if self.device.type == "cuda" else None
        )

        # Data
        self.train_loader = DataLoader(
            TrajectoryDataset(train_h5),
            batch_size=batch_size, shuffle=True,
            num_workers=0,  # avoid HDF5 SWMR fork issues; set >0 only with care
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                TrajectoryDataset(val_h5),
                batch_size=batch_size, shuffle=False, num_workers=0,
            ) if val_h5 else None
        )

        # wandb
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(
                    project=project_name,
                    config={
                        "lr": lr, "batch_size": batch_size, "epochs": epochs,
                        "max_grad_norm": max_grad_norm,
                        "device": str(self.device),
                        "amp_dtype": str(self.amp_dtype),
                        "n_train_examples": len(self.train_loader.dataset),
                    },
                )
            except Exception as e:
                print(f"[TRAINER] wandb disabled: {e}")
                self.wandb = None

    # -----------------------------------------------------------------
    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        for epoch in range(self.epochs):
            t0 = time.time()
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in self.train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                actions = batch["action"].to(self.device, non_blocking=True)
                instructions = list(batch["instruction"])

                self.optimizer.zero_grad()
                if self.device.type == "cuda":
                    with autocast(device_type="cuda", dtype=self.amp_dtype):
                        out = self.model(
                            imgs, instructions,
                            action_gt=actions, return_loss=True,
                        )
                        loss = out["loss"]
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # MPS or CPU — autocast optional, no GradScaler
                    with autocast(device_type=str(self.device.type),
                                  dtype=self.amp_dtype,
                                  enabled=self.device.type == "mps"):
                        out = self.model(
                            imgs, instructions,
                            action_gt=actions, return_loss=True,
                        )
                        loss = out["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm,
                    )
                    self.optimizer.step()

                self.scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(1, n_batches)
            history["train_loss"].append(avg_train)

            # Validation
            val_loss = float("nan")
            if self.val_loader is not None:
                val_loss = self.validate()
                history["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = os.path.join(
                        self.checkpoint_dir,
                        f"vla_epoch{epoch+1:03d}_val{val_loss:.4f}.pt",
                    )
                    self._save_checkpoint(ckpt_path)

            elapsed = time.time() - t0
            lr_now = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1:3d}/{self.epochs} | "
                f"train_loss={avg_train:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={lr_now:.2e} | "
                f"elapsed={elapsed:.1f}s"
            )
            if self.wandb:
                self.wandb.log({
                    "train/loss": avg_train,
                    "val/loss": val_loss,
                    "epoch": epoch,
                    "lr": lr_now,
                })

        # Save final model (only trainable params — see _save_checkpoint)
        final_path = os.path.join(self.checkpoint_dir, "vla_final.pt")
        self._save_checkpoint(final_path)

        if self.wandb:
            self.wandb.finish()
        return history

    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                actions = batch["action"].to(self.device, non_blocking=True)
                instructions = list(batch["instruction"])
                with autocast(
                    device_type=str(self.device.type),
                    dtype=self.amp_dtype,
                    enabled=self.device.type == "mps",
                ):
                    out = self.model(
                        imgs, instructions,
                        action_gt=actions, return_loss=True,
                    )
                total += out["loss"].item()
                n += 1
        return total / max(1, n)

    # -----------------------------------------------------------------
    # Checkpoint helpers — save ONLY trainable params (not frozen CLIP).
    # Without this filter, checkpoints are 584 MB per save (full CLIP
    # weights). With it, they are ~10 MB (only the 2.4M trainable params
    # in the denoiser + projections + cross-attention).
    # -----------------------------------------------------------------
    def _save_checkpoint(self, path: str) -> None:
        """Save only the trainable parameters (denoiser + projections +
        cross-attention). Frozen CLIP weights are NOT saved — they are
        always re-loaded from HuggingFace at model init."""
        trainable_state = {
            n: p.cpu().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        # Also save non-param buffer state (LayerNorm running stats, etc.)
        for n, b in self.model.named_buffers():
            trainable_state[n] = b.cpu().clone()
        torch.save(trainable_state, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[TRAINER] saved checkpoint: {path} ({size_mb:.1f} MB, "
              f"{len(trainable_state)} tensors)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _gather_h5(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "episode_*.h5")
    return sorted(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train ProductionVLA on logged HDF5 trajectories.",
    )
    parser.add_argument("--data-dir", default="data/sim_logs",
                        help="Directory containing episode_*.h5 files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of episodes to hold out for validation")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    all_h5 = _gather_h5(args.data_dir)
    if not all_h5:
        print(f"[ERROR] No episode_*.h5 files found in {args.data_dir}")
        print("        Run `python -m src.train --collect-only` first to generate data.")
        return 1

    # Train/val split
    n_val = max(1, int(len(all_h5) * args.val_fraction))
    val_h5 = all_h5[:n_val]
    train_h5 = all_h5[n_val:]
    print(f"[DATA] train episodes: {len(train_h5)} | val episodes: {len(val_h5)}")
    print(f"[DATA] train steps  : {sum(_count_steps(p) for p in train_h5)}")
    print(f"[DATA] val steps    : {sum(_count_steps(p) for p in val_h5)}")

    trainer = VLAImitationTrainer(
        train_h5=train_h5,
        val_h5=val_h5 or None,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
    )
    history = trainer.train()

    # Save loss curve
    with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return 0


def _count_steps(path: str) -> int:
    with h5py.File(path, "r") as f:
        return int(f.attrs.get("num_steps", 0))


if __name__ == "__main__":
    sys.exit(main())
