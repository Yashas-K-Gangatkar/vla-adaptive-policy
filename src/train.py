"""src/train.py — Closed-loop VLA execution pipeline.

The orchestrator. Wires together mujoco_env + vla_network + data_logger +
optimize into a single pipeline that can:
  (1) Collect expert demonstrations (random-policy rollout for now)
  (2) Train the VLA via imitation learning
  (3) Roll out the trained policy and visualize the trajectory

This file replaces the original `train.py` which used random text
embeddings and only ran 5 simulation steps. Here:
  - Text is encoded by the real CLIP tokenizer inside ProductionVLA
  - Episodes have proper reset() + max_steps + success detection
  - The trained policy is loaded from checkpoint before inference
  - Trajectories are logged to HDF5 with full image frames
  - The plotter is called at the end to produce paper-quality figures

Usage:
  # Collect 20 episodes of random-policy data for offline training
  python -m src.train --collect-only --episodes 20

  # Train the VLA on collected data
  python -m src.optimize --epochs 50

  # Evaluate the trained policy and plot
  python -m src.train --evaluate --checkpoint checkpoints/vla_final.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

# Make the project root importable when run as `python -m src.train`
# or `python src/train.py`. This allows `from src.* import ...` to work
# in both invocation styles.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.vla_network import ProductionVLA  # noqa: E402
from src.simulation.mujoco_env import PushBlockVLAEnv  # noqa: E402
from src.simulation.mujoco_env_multimodal import MultiModalVLAEnv, multimodal_expert_action  # noqa: E402
from src.utils.data_logger import ProductionDataLogger  # noqa: E402
from src.utils.plotter import render_trajectory_analytics  # noqa: E402
from src.utils.scripted_expert import scripted_expert_action  # noqa: E402


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data collection (random policy for now; replace with teleop later)
# -----------------------------------------------------------------------------
def collect_data(
    n_episodes: int,
    max_steps_per_episode: int,
    out_dir: str,
    seed: int = 0,
    multimodal: bool = False,
) -> List[str]:
    """Collect n_episodes of expert demonstrations.

    Returns list of saved HDF5 file paths.

    If multimodal=True, uses MultiModalVLAEnv (each color has 2 valid zones,
    creating bimodal action distribution). Otherwise uses the unimodal
    PushBlockVLAEnv (one zone per color, deterministic expert).
    """
    if multimodal:
        env = MultiModalVLAEnv(render_mode="rgb_array", seed=seed,
                               max_steps=max_steps_per_episode)
        expert_fn = multimodal_expert_action
    else:
        env = PushBlockVLAEnv(render_mode="rgb_array", seed=seed,
                              max_steps=max_steps_per_episode)
        expert_fn = scripted_expert_action
    rng = np.random.default_rng(seed)
    saved_paths: List[str] = []
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1 << 30)))
        instruction = info["instruction"]
        logger = ProductionDataLogger(log_dir=out_dir)
        total_reward = 0.0
        success = False

        for step in range(max_steps_per_episode):
            # Use the SCRIPTED EXPERT to generate expert demonstrations.
            # For the unimodal env: IK solve to the single target zone.
            # For the multimodal env: IK solve to the randomly-selected zone.
            # In both cases the expert solves the task on step 1, so each
            # episode records one expert action that the VLA learns to imitate.
            action = expert_fn(env, obs, info)
            pre_image = obs["image"].copy()
            pre_ee = obs["end_effector"].copy()
            pre_blocks = obs["block_positions"].copy()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.record_step(
                step_idx=step,
                image_frame=pre_image,
                instruction=instruction,
                action_torque=action,
                end_effector_pos=pre_ee,
                reward=0.0,
                block_positions=pre_blocks,
            )
            total_reward += reward
            if terminated:
                success = True
                break
            if truncated:
                break

        path = logger.save_episode(
            instruction=instruction,
            success=success,
            total_reward=total_reward,
        )
        saved_paths.append(path)
        print(f"[COLLECT] episode {ep+1:3d}/{n_episodes} | "
              f"steps={step+1:3d} | reward={total_reward:+.2f} | "
              f"success={success} | {os.path.basename(path)}")

    env.close()
    return saved_paths


# -----------------------------------------------------------------------------
# Evaluate a trained policy
# -----------------------------------------------------------------------------
def evaluate(
    checkpoint: str,
    n_episodes: int,
    max_steps_per_episode: int,
    out_dir: str,
    seed: int = 0,
    multimodal: bool = False,
) -> List[str]:
    """Roll out the trained policy and save trajectories + plots.

    If multimodal=True, evaluates on MultiModalVLAEnv (2 valid zones per
    color). Otherwise evaluates on PushBlockVLAEnv (unimodal).
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = _device()
    print(f"[EVAL] device = {device}")

    # BUG FIX (reviewer round 2): Read training metadata so we use the
    # SAME diffusion_steps and policy_mode the model was trained with.
    # Previously this was hardcoded to diffusion_steps=20 and policy_mode
    # was left at the default "auto" (=MSE), which silently broke DDPM
    # inference: the scheduler used 20 timesteps but the model's time
    # embedding was trained on 100, AND the forward pass routed to MSE
    # instead of DDPM. This made every DDPM checkpoint look broken.
    meta_path = os.path.join(os.path.dirname(checkpoint), "vla_final_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[EVAL] loaded metadata: {meta}")
    diffusion_steps = int(meta.get("diffusion_steps", 100))
    policy_mode = meta.get("policy_mode", "auto")

    if multimodal:
        env = MultiModalVLAEnv(render_mode="rgb_array", seed=seed,
                               max_steps=max_steps_per_episode)
    else:
        env = PushBlockVLAEnv(render_mode="rgb_array", seed=seed,
                              max_steps=max_steps_per_episode)
    # BUG FIX (reviewer round 2): Use diffusion_steps=100 (match training).
    # The previous hardcoded 20 was a silent train/inference mismatch.
    model = ProductionVLA(action_dim=3, diffusion_steps=diffusion_steps).to(device)
    # Set policy mode from metadata (not from re-analyzing data, which was
    # a bug: re-analysis used the unimodal sim_logs and would overwrite a
    # DDPM checkpoint's mode with MSE).
    model.policy_mode = policy_mode
    print(f"[EVAL] policy_mode={policy_mode} | diffusion_steps={diffusion_steps}")
    # Load only the trainable params (checkpoints store denoiser + projections
    # + cross-attention, NOT the frozen CLIP weights)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state)
    print(f"[EVAL] loaded {n_loaded} tensors from checkpoint; "
          f"missing in checkpoint: {len(missing)} (expected — frozen CLIP weights)")
    model.eval()
    print(f"[EVAL] loaded checkpoint: {checkpoint}")

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved_paths: List[str] = []
    success_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1 << 30)))
        instruction = info["instruction"]
        logger = ProductionDataLogger(log_dir=out_dir)
        total_reward = 0.0
        success = False

        for step in range(max_steps_per_episode):
            # Preprocess image
            img_t = (torch.from_numpy(obs["image"])
                     .permute(2, 0, 1)
                     .float().unsqueeze(0).to(device) / 255.0)
            with torch.no_grad():
                out = model(img_t, [instruction])
            action = out["action"].squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            logger.record_step(
                step_idx=step,
                image_frame=obs["image"],
                instruction=instruction,
                action_torque=action,
                end_effector_pos=obs["end_effector"],
                reward=reward,
                block_positions=obs["block_positions"],
            )
            total_reward += reward
            if terminated:
                success = True
                break
            if truncated:
                break

        path = logger.save_episode(
            instruction=instruction,
            success=success,
            total_reward=total_reward,
        )
        saved_paths.append(path)
        if success:
            success_count += 1
        print(f"[EVAL] episode {ep+1:3d}/{n_episodes} | "
              f"steps={step+1:3d} | reward={total_reward:+.2f} | "
              f"success={success} | {instruction}")

        # Render the trajectory plot
        try:
            render_trajectory_analytics(
                path,
                workspace_bounds={
                    "x": (-0.6, 0.6),
                    "y": (-0.4, 0.4),
                    "z": (0.4, 1.0),
                },
            )
        except Exception as e:
            print(f"[EVAL] plotter failed for {path}: {e}")

    env.close()
    rate = success_count / max(1, n_episodes)
    print(f"\n[EVAL] success rate: {success_count}/{n_episodes} = {rate:.1%}")
    return saved_paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLA closed-loop pipeline (collect / evaluate)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect-only", action="store_true",
                     help="Collect random-policy demonstrations")
    mode.add_argument("--evaluate", action="store_true",
                     help="Roll out a trained policy")
    parser.add_argument("--checkpoint", default="checkpoints/vla_final.pt",
                        help="Checkpoint path for --evaluate")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--data-dir", default="data/sim_logs")
    parser.add_argument("--eval-dir", default="data/eval_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multimodal", action="store_true",
                        help="Use the multi-modal env (2 valid zones per color). "
                             "Use this for both --collect-only and --evaluate "
                             "when working with the multi-modal task.")
    args = parser.parse_args()

    if args.collect_only:
        collect_data(
            n_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            out_dir=args.data_dir,
            seed=args.seed,
            multimodal=args.multimodal,
        )
        print(f"\n[COLLECT] Done. {args.episodes} episodes saved to {args.data_dir}/")
        print("Next: train the VLA with `python -m src.optimize --epochs 50`")
    elif args.evaluate:
        if not os.path.exists(args.checkpoint):
            print(f"[ERROR] checkpoint not found: {args.checkpoint}")
            print("Train the model first with `python -m src.optimize --epochs 50`")
            return 1
        evaluate(
            checkpoint=args.checkpoint,
            n_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            out_dir=args.eval_dir,
            seed=args.seed,
            multimodal=args.multimodal,
        )
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
