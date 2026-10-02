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
) -> List[str]:
    """Collect n_episodes of random-policy demonstrations.

    Returns list of saved HDF5 file paths.

    NOTE: Random-policy data is fine for testing the pipeline end-to-end,
    but a real VLA needs expert demonstrations. For T036 (industrial
    material handling), replace this with:
      (a) Teleop from a real xArm or KUKA via ROS2, OR
      (b) A scripted motion-planning expert (MoveIt + OMPL), OR
      (c) RL fine-tuning on top of the random-data pretrained model.
    """
    env = PushBlockVLAEnv(render_mode="rgb_array", seed=seed,
                          max_steps=max_steps_per_episode)
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
            # Use the SCRIPTED EXPERT (Jacobian IK + quasi-static control)
            # to generate expert demonstrations. The expert solves the
            # reach-to-target task on step 1 (100% success rate), so each
            # episode records one expert action that the VLA will learn
            # to imitate. The VLA learns (image, instruction) -> expert_action.
            action = scripted_expert_action(env, obs, info)
            # BUG FIX (reviewer): Save PRE-step image (what the VLA sees BEFORE acting).
            # Previously saved post-step image (arm already at target) → data leakage.
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
) -> List[str]:
    """Roll out the trained policy and save trajectories + plots."""
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = _device()
    print(f"[EVAL] device = {device}")

    env = PushBlockVLAEnv(render_mode="rgb_array", seed=seed,
                          max_steps=max_steps_per_episode)
    model = ProductionVLA(action_dim=3, diffusion_steps=20).to(device)
    # Use inference-time fewer steps (DDPM scheduler is configurable)
    # but DDPMScheduler.timesteps is fixed at construction, so rebuild:
    from diffusers import DDPMScheduler
    model.noise_scheduler = DDPMScheduler(
        num_train_timesteps=20,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    # Load only the trainable params (checkpoints store denoiser + projections
    # + cross-attention, NOT the frozen CLIP weights — see VLAImitationTrainer._save_checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    # strict=False because the checkpoint does not contain CLIP weights
    # (they are always reloaded from HuggingFace at model init)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Report what was loaded (helps debug checkpoint/version mismatches)
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
    args = parser.parse_args()

    if args.collect_only:
        collect_data(
            n_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            out_dir=args.data_dir,
            seed=args.seed,
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
        )
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
