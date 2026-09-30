"""src/utils/scripted_expert.py — IK-based reach-to-target expert.

Uses iterative Jacobian pseudo-inverse IK to compute target joint angles,
then outputs those angles directly as the action (since the env now uses
position actuators, not motor/torque actuators — MuJoCo's internal PD
controller drives the joints to the target angles).

For the reach-to-target task, this expert should achieve 80-95% success.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import mujoco

COLOR_TO_BLOCK_IDX = {"red": 0, "green": 1, "blue": 2}
ZONE_POSITIONS = {
    "red":   np.array([0.45, 0.15, 0.501]),
    "green": np.array([0.45, 0.00, 0.501]),
    "blue":  np.array([0.45, -0.15, 0.501]),
}

MAX_ANGLE = 1.5  # matches env's MAX_ANGLE — action in [-1, 1] -> [-1.5, 1.5] rad


def _ik_solve(env, target_ee_pos: np.ndarray,
              max_iters: int = 100, lr: float = 0.3, tol: float = 0.005) -> np.ndarray:
    """Iterative Jacobian pseudo-inverse IK. Returns target joint angles."""
    qpos_backup = env.data.qpos.copy()
    qvel_backup = env.data.qvel.copy()

    q = env.data.qpos[21:24].copy()
    nv = env.model.nv

    for _ in range(max_iters):
        env.data.qpos[21:24] = q
        mujoco.mj_forward(env.model, env.data)
        ee = env.data.site_xpos[env._ee_site_id].copy()
        error = target_ee_pos - ee
        if np.linalg.norm(error) < tol:
            break
        jacp = np.zeros((3, nv))
        jacv = np.zeros((3, nv))
        mujoco.mj_jacSite(env.model, env.data, jacp, jacv, env._ee_site_id)
        jacp_arm = jacp[:, nv-3:nv]
        # Damped pseudo-inverse (Levenberg-Marquardt)
        jtj = jacp_arm @ jacp_arm.T + 1e-3 * np.eye(3)
        delta_q = jacp_arm.T @ np.linalg.solve(jtj, error) * lr
        q = q + delta_q
        q = np.clip(q, -1.5, 1.5)

    # Restore env state (IK was computed on a copy)
    env.data.qpos[:] = qpos_backup
    env.data.qvel[:] = qvel_backup
    mujoco.mj_forward(env.model, env.data)
    return q.astype(np.float32)


def scripted_expert_action(env, obs: Dict, info: Dict) -> np.ndarray:
    """Compute expert action: target joint angles to reach the target zone."""
    target_color = info["target_color"]
    target_zone_pos = ZONE_POSITIONS[target_color]

    # Compute target joint angles via IK
    target_q = _ik_solve(env, target_zone_pos)

    # Normalize to [-1, 1] (env multiplies by MAX_ANGLE=1.5)
    action = np.clip(target_q / MAX_ANGLE, -1.0, 1.0).astype(np.float32)
    return action


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.simulation.mujoco_env import PushBlockVLAEnv

    env = PushBlockVLAEnv(seed=42, max_steps=200)
    success_count = 0
    n_episodes = 20

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0.0
        for step in range(200):
            action = scripted_expert_action(env, obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                success_count += 1
                break
            if truncated:
                break
        ee_pos = obs["end_effector"]
        print(f"  ep {ep+1:2d}/{n_episodes} | steps={step+1:3d} | "
              f"reward={total_reward:+.2f} | success={terminated} | "
              f"color={info['target_color']} | "
              f"ee=({ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}) | "
              f"dist={info['distance_to_target']:.3f}")

    env.close()
    rate = success_count / n_episodes
    print(f"\n[EXPERT] success rate: {success_count}/{n_episodes} = {rate:.1%}")
