"""src/simulation/mujoco_env_multimodal.py — Multi-modal reach task.

Same 3-DOF arm, same 3 colors, but now each color has TWO valid zones
(one at Y=+0.15, one at Y=-0.15). The expert randomly picks one,
creating a BIMODAL action distribution (std ≈ 0.15 >> 0.01 threshold).

The PolicySelector should detect this and switch to DDPM.
"""
from __future__ import annotations
import numpy as np
import mujoco
from typing import Dict, Any, Optional, Tuple
from src.simulation.mujoco_env import PushBlockVLAEnv, MJCF, COLOR_MAP

# Extended MJCF: add mirror zones (red_2, blue_2) on the opposite Y side
MULTIMODAL_MJCF = MJCF.replace(
    '<site name="zone_red"    type="box" pos="0.45 0.15 0.501"\n          size="0.05 0.05 0.001" rgba="0.8 0.1 0.1 0.35"/>',
    '<site name="zone_red"    type="box" pos="0.45 0.15 0.501"\n          size="0.05 0.05 0.001" rgba="0.8 0.1 0.1 0.35"/>\n        <site name="zone_red_2"  type="box" pos="0.45 -0.15 0.501"\n          size="0.05 0.05 0.001" rgba="0.8 0.1 0.1 0.35"/>'
).replace(
    '<site name="zone_blue"   type="box" pos="0.45 -0.15 0.501"\n          size="0.05 0.05 0.001" rgba="0.1 0.2 0.8 0.35"/>',
    '<site name="zone_blue"   type="box" pos="0.45 -0.15 0.501"\n          size="0.05 0.05 0.001" rgba="0.1 0.2 0.8 0.35"/>\n        <site name="zone_blue_2" type="box" pos="0.45 0.15 0.501"\n          size="0.05 0.05 0.001" rgba="0.1 0.2 0.8 0.35"/>'
)

# Extended zone positions — each color (except green) has 2 valid zones
MULTIMODAL_ZONES = {
    "red":    [np.array([0.45, 0.15, 0.501]), np.array([0.45, -0.15, 0.501])],
    "green":  [np.array([0.45, 0.00, 0.501])],  # green stays single
    "blue":   [np.array([0.45, -0.15, 0.501]), np.array([0.45, 0.15, 0.501])],
}


class MultiModalVLAEnv(PushBlockVLAEnv):
    """Multi-modal reach-to-target: 2 valid zones per color (except green).

    The expert randomly picks one of the 2 valid zones, creating a
    bimodal action distribution. This is the task where DDPM should
    outperform MSE.
    """

    def __init__(self, render_mode: str = "rgb_array", seed: int = 0,
                 max_steps: int = 100, camera_id: int = -1):
        # Override MJCF with the multi-modal version
        self._multimodal_mjcf = MULTIMODAL_MJCF
        # Temporarily replace the parent's MJCF
        import src.simulation.mujoco_env as _me
        _orig_mjcf = _me.MJCF
        _me.MJCF = MULTIMODAL_MJCF
        try:
            super().__init__(render_mode=render_mode, seed=seed,
                             max_steps=max_steps, camera_id=camera_id)
        finally:
            _me.MJCF = _orig_mjcf
        # Re-init zone site IDs for the extended MJCF
        self._zone_sites = {}
        for color in ("red", "green", "blue"):
            for suffix in ("", "_2"):
                name = f"zone_{color}{suffix}"
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    self._zone_sites[f"{color}{suffix}"] = sid

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict, Dict]:
        # Call parent reset but override the zone selection
        obs, info = super().reset(seed=seed, options=options)
        # Pick a random zone from the valid zones for this color
        color = self._current_color
        valid_zones = MULTIMODAL_ZONES[color]
        self._target_zone_idx = int(self._rng.integers(len(valid_zones)))
        self._target_zone_pos = valid_zones[self._target_zone_idx]
        # Update instruction to reflect multi-modal nature
        info["instruction"] = f"reach to a {color} zone"
        info["target_zone_pos"] = self._target_zone_pos.tolist()
        info["multimodal"] = True
        info["num_valid_zones"] = len(valid_zones)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_q = action * self.MAX_ANGLE
        self.data.qpos[21:24] = target_q
        self.data.qvel[18:21] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()

        # BUG FIX (reviewer): Success should check ANY valid zone, not just the selected one.
        # The model doesn't know which zone was randomly selected — both are valid targets.
        ee_pos = obs["end_effector"]
        color = self._current_color
        valid_zones = MULTIMODAL_ZONES[color]
        # Distance to NEAREST valid zone (not just the selected one)
        distances = [float(np.linalg.norm(ee_pos - zone)) for zone in valid_zones]
        dist = min(distances)  # best distance across all valid zones
        reward = -dist
        success = dist < self.ZONE_RADIUS
        if success:
            reward += 5.0
        terminated = success
        truncated = self._step_count >= self.max_steps

        info = {
            "instruction": f"reach to a {self._current_color} zone",
            "target_color": self._current_color,
            "target_zone_pos": self._target_zone_pos.tolist(),
            "distance_to_target": dist,
            "success": bool(success),
            "step": self._step_count,
            "end_effector_pos": obs["end_effector"].copy(),
            "multimodal": True,
            "target_zone_idx": self._target_zone_idx,
        }
        return obs, float(reward), terminated, truncated, info


# ─── Multi-modal expert: randomly picks LEFT or RIGHT zone ───────────────
def multimodal_expert_action(env, obs: Dict, info: Dict) -> np.ndarray:
    """Expert for multi-modal task: uses IK to reach the SELECTED zone."""
    from src.utils.scripted_expert import _ik_solve, MAX_ANGLE
    target_zone = np.array(info["target_zone_pos"], dtype=np.float32)
    target_q = _ik_solve(env, target_zone)
    action = np.clip(target_q / MAX_ANGLE, -1.0, 1.0).astype(np.float32)
    return action


# ─── Smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    env = MultiModalVLAEnv(seed=42, max_steps=200)
    success_count = 0
    for ep in range(20):
        obs, info = env.reset(seed=42 + ep)
        for step in range(200):
            action = multimodal_expert_action(env, obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                success_count += 1
                break
            if truncated:
                break
        zone_idx = info.get("target_zone_idx", "?")
        print(f"  ep {ep+1:2d} | steps={step+1:3d} | success={terminated} | "
              f"color={info['target_color']:5s} | zone={zone_idx} | "
              f"dist={info['distance_to_target']:.3f}")

    env.close()
    print(f"\n[EXPERT] success rate: {success_count}/20 = {success_count/20:.0%}")
    print("\n=== PolicySelector on multi-modal data ===")
    # Collect a few episodes and check the selector
    from src.utils.policy_selector import PolicySelector
    import glob, tempfile
    from src.utils.data_logger import ProductionDataLogger

    env = MultiModalVLAEnv(seed=42, max_steps=50)
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        for ep in range(20):
            obs, info = env.reset(seed=int(rng.integers(0, 1 << 30)))
            logger = ProductionDataLogger(log_dir=tmp)
            for step in range(50):
                action = multimodal_expert_action(env, obs, info)
                pre_obs = obs.copy()
                pre_ee = obs["end_effector"].copy()
                pre_blocks = obs["block_positions"].copy()
                obs, reward, terminated, truncated, info = env.step(action)
                logger.record_step(
                    step_idx=step, image_frame=pre_obs["image"],
                    instruction=info["instruction"], action_torque=action,
                    end_effector_pos=pre_ee, reward=0.0,
                    block_positions=pre_blocks,
                )
                if terminated or truncated:
                    break
            logger.save_episode(instruction=info["instruction"],
                                 success=terminated, total_reward=reward)

        env.close()
        h5_files = sorted(glob.glob(os.path.join(tmp, "episode_*.h5")))
        selector = PolicySelector(threshold=0.01)
        mode = selector.analyze(h5_files)
        print(f"\nSelector chose: {mode.upper()}")
        print(f"Expected: DDPM (multi-modal — 2 valid actions per instruction)")
