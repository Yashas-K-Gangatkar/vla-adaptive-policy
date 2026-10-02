"""src/compare_modes.py — A/B test: MSE vs DDPM on the SAME dataset.

This is the script that actually answers the reviewer's question:
"Does DDPM outperform MSE on multimodal data, or do they tie?"

Previous runs compared MSE-on-unimodal vs DDPM-on-multimodal — apples-to-
oranges. This script trains BOTH modes on the SAME multimodal data and
evaluates BOTH on the SAME 20 evaluation seeds, then prints a clean
comparison table.

Usage:
    # Assumes data/multimodal_logs/ already has 100+ episodes collected via:
    #   python -m src.train --collect-only --multimodal \\
    #       --episodes 100 --max-steps 50 --data-dir data/multimodal_logs

    python -m src.compare_modes \\
        --data-dir data/multimodal_logs \\
        --epochs 100 --batch-size 4 --eval-episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from typing import Dict, List

import numpy as np
import torch

# Make project importable when run as `python -m src.compare_modes`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.vla_network import ProductionVLA  # noqa: E402
from src.simulation.mujoco_env_multimodal import MultiModalVLAEnv  # noqa: E402
from src.optimize import VLAImitationTrainer, _gather_h5, _count_steps  # noqa: E402


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_one_mode(
    train_h5: List[str],
    val_h5: List[str],
    mode: str,
    epochs: int,
    batch_size: int,
    ckpt_dir: str,
) -> str:
    """Train a model with a FORCED policy mode, return the checkpoint path."""
    print(f"\n{'='*70}")
    print(f"  TRAINING WITH FORCED MODE = {mode.upper()}")
    print(f"{'='*70}")

    # Clean checkpoint dir so we don't load a stale vla_final.pt
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = VLAImitationTrainer(
        train_h5=train_h5,
        val_h5=val_h5 or None,
        lr=1e-4,
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        use_wandb=False,
        force_mode=mode,  # <-- force this mode regardless of selector
    )
    trainer.train()
    final_ckpt = os.path.join(ckpt_dir, "vla_final.pt")
    assert os.path.exists(final_ckpt), f"Training did not produce {final_ckpt}"
    return final_ckpt


def _evaluate(
    checkpoint: str,
    n_episodes: int,
    max_steps: int,
    seed: int = 42,
) -> Dict:
    """Evaluate a checkpoint on MultiModalVLAEnv with deterministic seeds.

    Returns dict with success_rate, distances, per_episode details.
    """
    device = _device()
    print(f"\n[EVAL] device = {device}")

    # Read training metadata to get policy_mode + diffusion_steps
    meta_path = os.path.join(os.path.dirname(checkpoint), "vla_final_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    diffusion_steps = int(meta.get("diffusion_steps", 100))
    policy_mode = meta.get("policy_mode", "auto")
    print(f"[EVAL] checkpoint: {checkpoint}")
    print(f"[EVAL] policy_mode={policy_mode} | diffusion_steps={diffusion_steps} | "
          f"best_val={meta.get('best_val_loss', '?')} (epoch {meta.get('best_epoch', '?')})")

    env = MultiModalVLAEnv(seed=seed, max_steps=max_steps)
    model = ProductionVLA(action_dim=3, diffusion_steps=diffusion_steps).to(device)
    model.policy_mode = policy_mode  # critical: use trained mode, not default MSE
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    successes = 0
    distances = []
    per_ep = []
    # IMPORTANT: use deterministic seeds 42..42+n-1 so both MSE and DDPM
    # see the EXACT same starting conditions. Otherwise comparing success
    # rates is meaningless (different seeds = different difficulty).
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_success = False
        final_dist = float("nan")
        for step in range(max_steps):
            img = (torch.from_numpy(obs["image"])
                   .permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0)
            with torch.no_grad():
                out = model(img, [info["instruction"]])
            action = out["action"].squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = env.step(action)
            if term:
                ep_success = True
                final_dist = info["distance_to_target"]
                break
            if trunc:
                final_dist = info["distance_to_target"]
                break
        if ep_success:
            successes += 1
        distances.append(final_dist)
        per_ep.append({
            "episode": ep + 1,
            "seed": seed + ep,
            "color": info.get("target_color", "?"),
            "success": ep_success,
            "final_distance": final_dist,
        })
        print(f"  ep {ep+1:2d}/{n_episodes} | success={ep_success} | "
              f"dist={final_dist:.3f} | {info.get('instruction', '')}")

    env.close()
    rate = successes / n_episodes
    result = {
        "checkpoint": checkpoint,
        "policy_mode": policy_mode,
        "diffusion_steps": diffusion_steps,
        "best_val_loss": meta.get("best_val_loss"),
        "best_epoch": meta.get("best_epoch"),
        "successes": successes,
        "n_episodes": n_episodes,
        "success_rate": rate,
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "per_episode": per_ep,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B test: train MSE and DDPM on the same data, "
                    "evaluate both on the same seeds.",
    )
    parser.add_argument("--data-dir", default="data/multimodal_logs",
                        help="Directory with episode_*.h5 files "
                             "(must be multi-modal data for the comparison "
                             "to be meaningful)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; eval uses seeds 42..42+N-1")
    parser.add_argument("--mse-ckpt-dir", default="checkpoints_mse")
    parser.add_argument("--ddpm-ckpt-dir", default="checkpoints_ddpm")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; evaluate existing checkpoints "
                             "at <mse_ckpt_dir>/vla_final.pt and "
                             "<ddpm_ckpt_dir>/vla_final.pt")
    args = parser.parse_args()

    all_h5 = _gather_h5(args.data_dir)
    if not all_h5:
        print(f"[ERROR] No episode_*.h5 files in {args.data_dir}")
        print("Collect multi-modal data first:")
        print("  python -m src.train --collect-only --multimodal \\")
        print("      --episodes 100 --max-steps 50 "
              "--data-dir data/multimodal_logs")
        return 1

    # Train/val split (deterministic — same for both MSE and DDPM)
    n_val = max(1, int(len(all_h5) * 0.2))
    val_h5 = all_h5[:n_val]
    train_h5 = all_h5[n_val:]
    print(f"[DATA] train episodes: {len(train_h5)} | val episodes: {len(val_h5)}")
    print(f"[DATA] train steps  : {sum(_count_steps(p) for p in train_h5)}")
    print(f"[DATA] val steps    : {sum(_count_steps(p) for p in val_h5)}")

    mse_ckpt = os.path.join(args.mse_ckpt_dir, "vla_final.pt")
    ddpm_ckpt = os.path.join(args.ddpm_ckpt_dir, "vla_final.pt")

    if not args.skip_train:
        _train_one_mode(
            train_h5, val_h5, mode="mse",
            epochs=args.epochs, batch_size=args.batch_size,
            ckpt_dir=args.mse_ckpt_dir,
        )
        _train_one_mode(
            train_h5, val_h5, mode="ddpm",
            epochs=args.epochs, batch_size=args.batch_size,
            ckpt_dir=args.ddpm_ckpt_dir,
        )
    else:
        print("\n[COMPARE] --skip-train: using existing checkpoints")
        assert os.path.exists(mse_ckpt), f"Missing {mse_ckpt}"
        assert os.path.exists(ddpm_ckpt), f"Missing {ddpm_ckpt}"

    # Evaluate both on the SAME seeds
    print("\n" + "=" * 70)
    print("  EVALUATING MSE")
    print("=" * 70)
    mse_result = _evaluate(
        mse_ckpt, args.eval_episodes, args.max_steps, seed=args.seed,
    )
    print(f"\n>>> MSE success: {mse_result['successes']}/{args.eval_episodes} "
          f"= {mse_result['success_rate']*100:.0f}% "
          f"(mean dist = {mse_result['mean_distance']:.3f})")

    print("\n" + "=" * 70)
    print("  EVALUATING DDPM")
    print("=" * 70)
    ddpm_result = _evaluate(
        ddpm_ckpt, args.eval_episodes, args.max_steps, seed=args.seed,
    )
    print(f"\n>>> DDPM success: {ddpm_result['successes']}/{args.eval_episodes} "
          f"= {ddpm_result['success_rate']*100:.0f}% "
          f"(mean dist = {ddpm_result['mean_distance']:.3f})")

    # Print comparison table
    print("\n" + "=" * 70)
    print("  A/B COMPARISON: MSE vs DDPM on SAME multi-modal data")
    print("=" * 70)
    print(f"  Dataset       : {args.data_dir}")
    print(f"  Train episodes: {len(train_h5)} | val episodes: {len(val_h5)}")
    print(f"  Eval episodes : {args.eval_episodes} (seeds {args.seed}..{args.seed+args.eval_episodes-1})")
    print()
    print(f"  {'Mode':<6} | {'Success':<10} | {'Rate':<6} | "
          f"{'Mean dist':<10} | {'Best val':<10} | {'Best epoch':<10}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
    for r, label in [(mse_result, "MSE"), (ddpm_result, "DDPM")]:
        print(f"  {label:<6} | {r['successes']}/{r['n_episodes']:<7} | "
              f"{r['success_rate']*100:>4.0f}% | "
              f"{r['mean_distance']:>10.3f} | "
              f"{r.get('best_val_loss', '?'):>10} | "
              f"epoch {r.get('best_epoch', '?')}")
    print()

    # Honest verdict
    diff = ddpm_result["success_rate"] - mse_result["success_rate"]
    if abs(diff) < 0.10:
        verdict = "TIE (within 10%) — DDPM does NOT meaningfully outperform MSE"
    elif diff > 0:
        verdict = f"DDPM WINS by {diff*100:.0f} percentage points"
    else:
        verdict = f"MSE WINS by {-diff*100:.0f} percentage points"
    print(f"  VERDICT: {verdict}")
    print()
    print("Note: a 'TIE' verdict means the paper's main claim (DDPM > MSE on")
    print("multimodal data) does NOT hold for this task. Either:")
    print("  (a) the task isn't actually hard enough to require DDPM, or")
    print("  (b) the DDPM training is unstable (look at val_loss curve), or")
    print("  (c) the multimodal data is too easy (single-step expert solves).")
    print("=" * 70)

    # Save results to JSON for the paper
    out_path = "compare_modes_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.data_dir,
            "train_episodes": len(train_h5),
            "val_episodes": len(val_h5),
            "eval_episodes": args.eval_episodes,
            "eval_seed_base": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "mse": mse_result,
            "ddpm": ddpm_result,
            "verdict": verdict,
            "diff_success_rate": diff,
        }, f, indent=2)
    print(f"\n[COMPARE] Full results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
