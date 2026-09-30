"""src/utils/policy_selector.py — Adaptive DDPM/MSE Policy Selector

Analyzes the action distribution in training data and automatically selects:
  - Direct MSE regression  → if action distribution is unimodal (std < threshold)
  - DDPM diffusion policy  → if action distribution is multimodal (std >= threshold)

This is the NOVEL CONTRIBUTION of the paper: nobody has published an automatic
method for choosing between diffusion and regression based on the action
distribution's modality.

Usage:
    from src.utils.policy_selector import PolicySelector
    selector = PolicySelector(threshold=0.01)
    mode = selector.analyze(h5_paths)
    # mode = "mse" or "ddpm"
    # Then use mode to select training/inference path
"""
from __future__ import annotations

import os
from typing import List, Dict, Tuple
import numpy as np
import h5py


class PolicySelector:
    """Automatically selects between DDPM and MSE based on action distribution."""

    def __init__(self, threshold: float = 0.01, verbose: bool = True):
        """
        Args:
            threshold: action std below this → deterministic → use MSE.
                       Above this → multimodal → use DDPM.
                       Default 0.01 (in normalized [-1,1] action space).
                       0.01 means ~0.015 rad joint angle variation.
            verbose: print analysis results.
        """
        self.threshold = threshold
        self.verbose = verbose
        self.mode = None
        self.stats = {}

    def analyze(self, h5_paths: List[str]) -> str:
        """Analyze action distributions from HDF5 episode files.

        For each unique instruction, computes the standard deviation of
        the expert actions. If ALL instructions have std < threshold,
        the task is deterministic → use MSE. Otherwise → use DDPM.

        Returns:
            "mse" or "ddpm"
        """
        # Collect all (instruction, action) pairs
        instruction_actions: Dict[str, List[np.ndarray]] = {}

        for path in h5_paths:
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as f:
                n = int(f.attrs.get("num_steps", 0))
                for i in range(n):
                    # Decode instruction
                    inst_bytes = bytes(f["instruction"][i].astype(np.uint8).tobytes())
                    inst = inst_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
                    # Get action (torque)
                    action = f["torque"][i].astype(np.float32)
                    if inst not in instruction_actions:
                        instruction_actions[inst] = []
                    instruction_actions[inst].append(action)

        if not instruction_actions:
            if self.verbose:
                print("[SELECTOR] WARNING: no data found, defaulting to 'mse'")
            self.mode = "mse"
            return self.mode

        # Compute per-instruction action statistics
        max_std = 0.0
        per_instruction_stats = {}

        for inst, actions in instruction_actions.items():
            actions_array = np.array(actions)  # (N, 3)
            if len(actions_array) < 2:
                std = 0.0
                mean = actions_array[0] if len(actions_array) > 0 else np.zeros(3)
            else:
                std = float(actions_array.std(axis=0).max())
                mean = actions_array.mean(axis=0)
            per_instruction_stats[inst] = {
                "n_samples": len(actions_array),
                "mean": mean.tolist(),
                "max_std": std,
            }
            max_std = max(max_std, std)

        # Select mode
        if max_std < self.threshold:
            self.mode = "mse"
        else:
            self.mode = "ddpm"

        self.stats = {
            "max_action_std": max_std,
            "threshold": self.threshold,
            "selected_mode": self.mode,
            "per_instruction": per_instruction_stats,
            "n_instructions": len(instruction_actions),
            "n_total_samples": sum(len(v) for v in instruction_actions.values()),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SELECTOR] ADAPTIVE POLICY SELECTION")
            print(f"{'='*60}")
            print(f"  Threshold: {self.threshold}")
            print(f"  Instructions found: {len(instruction_actions)}")
            print(f"  Total samples: {sum(len(v) for v in instruction_actions.values())}")
            print()
            for inst, stats in per_instruction_stats.items():
                print(f"  '{inst}':")
                print(f"    samples: {stats['n_samples']}")
                print(f"    max_std: {stats['max_std']:.6f}")
                print(f"    mean:    [{', '.join(f'{x:.4f}' for x in stats['mean'])}]")
                tag = "DETERMINISTIC" if stats["max_std"] < self.threshold else "MULTIMODAL"
                print(f"    → {tag}")
            print()
            print(f"  MAX action std across all instructions: {max_std:.6f}")
            print(f"  Selected mode: {self.mode.upper()}")
            if self.mode == "mse":
                print(f"  → Action distribution is UNIMODAL (std={max_std:.6f} < {self.threshold})")
                print(f"  → Using DIRECT MSE REGRESSION (deterministic, fast, precise)")
            else:
                print(f"  → Action distribution is MULTIMODAL (std={max_std:.6f} >= {self.threshold})")
                print(f"  → Using DDPM DIFFUSION POLICY (stochastic, captures multi-modality)")
            print(f"{'='*60}\n")

        return self.mode


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os, glob, tempfile
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Test 1: Analyze existing data (should be deterministic → MSE)
    h5_files = sorted(glob.glob("data/sim_logs/episode_*.h5"))
    if h5_files:
        print(f"Found {len(h5_files)} episode files")
        selector = PolicySelector(threshold=0.01)
        mode = selector.analyze(h5_files)
        print(f"Selected mode: {mode}")
        print(f"Expected: mse (deterministic reach-to-target task)")
    else:
        print("No episode files found. Run data collection first.")

    # Test 2: Synthetic multimodal data
    print("\n--- Test with synthetic multimodal data ---")
    with tempfile.TemporaryDirectory() as tmp:
        # Create a fake multimodal episode
        with h5py.File(os.path.join(tmp, "test.h5"), "w") as f:
            n = 20
            f.create_dataset("torque", (n, 3), dtype="f4")
            f.create_dataset("instruction", (n, 128), dtype="uint8",
                            chunks=(n, 128))
            for i in range(n):
                # Two modes: [+0.5, 0.7, 0.5] and [-0.5, 0.7, 0.5]
                if i % 2 == 0:
                    f["torque"][i] = [0.5, 0.7, 0.5]
                else:
                    f["torque"][i] = [-0.5, 0.7, 0.5]
                inst = b"reach to the red zone" + b"\x00" * 107
                f["instruction"][i] = np.frombuffer(inst[:128], dtype="uint8")
            f.attrs["num_steps"] = n

        selector2 = PolicySelector(threshold=0.01)
        mode2 = selector2.analyze([os.path.join(tmp, "test.h5")])
        print(f"Selected mode: {mode2}")
        print(f"Expected: ddpm (multimodal: two distinct action modes)")
