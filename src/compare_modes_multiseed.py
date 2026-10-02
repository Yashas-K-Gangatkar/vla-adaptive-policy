"""src/compare_modes_multiseed.py — Run the A/B comparison across multiple seeds.

This is the one-command reproduction script for the paper's 5-seed robustness
result. It uses ALREADY-TRAINED checkpoints in checkpoints_mse/ and
checkpoints_ddpm/ (run `python -m src.compare_modes` first to produce them)
and evaluates both on N different eval-seed bases, then prints a summary
table with mean ± std and the final verdict.

Usage:
    # Assumes checkpoints_mse/ and checkpoints_ddpm/ already exist (from
    # a previous `python -m src.compare_modes` run).
    python -m src.compare_modes_multiseed \\
        --eval-episodes 20 --seeds 42 142 242 342 442

Output:
    Per-seed success rates for MSE and DDPM
    Mean ± std across seeds
    Final verdict (DDPM WINS / TIE / MSE WINS) with delta

This is the script referenced in the paper's "Reproducing the 5-seed
robustness check" section. The numbers it produces are the ones reported
in Table 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.vla_network import ProductionVLA  # noqa: E402
from src.simulation.mujoco_env_multimodal import MultiModalVLAEnv  # noqa: E402


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_one_seed(
    checkpoint: str,
    seed_base: int,
    n_episodes: int,
    max_steps: int,
) -> Dict:
    """Evaluate one checkpoint on N episodes starting at seed_base."""
    device = _device()

    # Read metadata for policy_mode + diffusion_steps
    meta_path = os.path.join(os.path.dirname(checkpoint), "vla_final_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    diffusion_steps = int(meta.get("diffusion_steps", 100))
    policy_mode = meta.get("policy_mode", "auto")

    env = MultiModalVLAEnv(seed=seed_base, max_steps=max_steps)
    model = ProductionVLA(action_dim=3, diffusion_steps=diffusion_steps).to(device)
    model.policy_mode = policy_mode
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    successes = 0
    distances = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_base + ep)
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
    env.close()
    return {
        "seed_base": seed_base,
        "successes": successes,
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "mean_distance": float(np.mean(distances)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the A/B MSE-vs-DDPM comparison across multiple seeds "
                    "using ALREADY-TRAINED checkpoints.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Episodes per seed base")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 142, 242, 342, 442],
                        help="List of seed bases to evaluate (default: "
                             "42 142 242 342 442 — the 5-seed set used in the paper)")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--mse-ckpt", default="checkpoints_mse/vla_final.pt")
    parser.add_argument("--ddpm-ckpt", default="checkpoints_ddpm/vla_final.pt")
    parser.add_argument("--out", default="compare_modes_multiseed_results.json")
    args = parser.parse_args()

    assert os.path.exists(args.mse_ckpt), (
        f"Missing {args.mse_ckpt}. Run `python -m src.compare_modes` first "
        "to train both MSE and DDPM checkpoints."
    )
    assert os.path.exists(args.ddpm_ckpt), (
        f"Missing {args.ddpm_ckpt}. Run `python -m src.compare_modes` first."
    )

    print("=" * 70)
    print(f"  MULTI-SEED A/B COMPARISON — {len(args.seeds)} seeds × "
          f"{args.eval_episodes} episodes each = "
          f"{len(args.seeds) * args.eval_episodes} total eval episodes")
    print("=" * 70)

    mse_results = []
    ddpm_results = []
    for seed in args.seeds:
        print(f"\n--- Seed base {seed} ---")
        print(f"[MSE ] evaluating {args.mse_ckpt} on seeds {seed}..{seed+args.eval_episodes-1}...")
        mse_r = _evaluate_one_seed(
            args.mse_ckpt, seed, args.eval_episodes, args.max_steps,
        )
        print(f"  → MSE : {mse_r['successes']}/{mse_r['n_episodes']} = "
              f"{mse_r['success_rate']*100:.0f}% (mean dist {mse_r['mean_distance']:.3f})")
        mse_results.append(mse_r)

        print(f"[DDPM] evaluating {args.ddpm_ckpt} on seeds {seed}..{seed+args.eval_episodes-1}...")
        ddpm_r = _evaluate_one_seed(
            args.ddpm_ckpt, seed, args.eval_episodes, args.max_steps,
        )
        print(f"  → DDPM: {ddpm_r['successes']}/{ddpm_r['n_episodes']} = "
              f"{ddpm_r['success_rate']*100:.0f}% (mean dist {ddpm_r['mean_distance']:.3f})")
        ddpm_results.append(ddpm_r)

    # Aggregate
    mse_rates = [r["success_rate"] for r in mse_results]
    ddpm_rates = [r["success_rate"] for r in ddpm_results]
    mse_mean, mse_std = float(np.mean(mse_rates)), float(np.std(mse_rates))
    ddpm_mean, ddpm_std = float(np.mean(ddpm_rates)), float(np.std(ddpm_rates))

    # Verdict
    diff = ddpm_mean - mse_mean
    if abs(diff) < 0.10:
        verdict = "TIE (within 10%) — DDPM does NOT meaningfully outperform MSE"
    elif diff > 0:
        verdict = f"DDPM WINS by {diff*100:.0f} percentage points"
    else:
        verdict = f"MSE WINS by {-diff*100:.0f} percentage points"

    # Print summary
    print("\n" + "=" * 70)
    print(f"  MULTI-SEED SUMMARY ({len(args.seeds)} seeds × "
          f"{args.eval_episodes} episodes)")
    print("=" * 70)
    print(f"  {'Seed':<6} | {'MSE':<14} | {'DDPM':<14} | {'Delta':<10}")
    print(f"  {'-'*6} | {'-'*14} | {'-'*14} | {'-'*10}")
    for i, seed in enumerate(args.seeds):
        m = mse_rates[i] * 100
        d = ddpm_rates[i] * 100
        delta = d - m
        print(f"  {seed:<6} | {m:>5.0f}%          | {d:>5.0f}%          | "
              f"{delta:+5.0f} pts   ")
    print(f"  {'-'*6} | {'-'*14} | {'-'*14} | {'-'*10}")
    print(f"  {'MEAN':<6} | {mse_mean:>5.1f}% ± {mse_std:>4.1f} | "
          f"{ddpm_mean:>5.1f}% ± {ddpm_std:>4.1f} | "
          f"{(ddpm_mean - mse_mean)*100:+5.1f} pts")
    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # Save results JSON
    out = {
        "n_seeds": len(args.seeds),
        "n_episodes_per_seed": args.eval_episodes,
        "total_episodes": len(args.seeds) * args.eval_episodes,
        "seeds": args.seeds,
        "mse": {
            "per_seed": [{"seed_base": r["seed_base"],
                          "success_rate": r["success_rate"],
                          "successes": r["successes"],
                          "n_episodes": r["n_episodes"],
                          "mean_distance": r["mean_distance"]}
                         for r in mse_results],
            "mean_success_rate": mse_mean,
            "std_success_rate": mse_std,
        },
        "ddpm": {
            "per_seed": [{"seed_base": r["seed_base"],
                          "success_rate": r["success_rate"],
                          "successes": r["successes"],
                          "n_episodes": r["n_episodes"],
                          "mean_distance": r["mean_distance"]}
                         for r in ddpm_results],
            "mean_success_rate": ddpm_mean,
            "std_success_rate": ddpm_std,
        },
        "verdict": verdict,
        "delta_success_rate": diff,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Results saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
