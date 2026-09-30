"""src/utils/plotter.py — Interactive trajectory visualizer.

Plotly-based, replaces the matplotlib-only plotter. Produces an interactive
HTML file (zoom, rotate, hover in any browser) AND a static PNG for paper
figures.

Bug-fixes vs the original:
  - NO eval() on CSV columns — the original used eval(pos) on stringified
    Python tuples, which is arbitrary code execution from data. This is a
    security vulnerability and we read positions as floats directly.
  - Time-color gradient on the trajectory line (Viridis) — lets you see
    whether the policy was fast/slow at each phase.
  - Workspace bounds overlay (the table surface as a transparent box).
  - Block positions tracked alongside end-effector — shows which block moved.
  - Episode comparison: pass multiple episode paths to overlay trajectories.
  - Static PNG export via kaleido (fallback: matplotlib if kaleido missing).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_h5(path: str) -> Dict:
    """Load an episode HDF5 file produced by ProductionDataLogger."""
    with h5py.File(path, "r") as f:
        return {
            "steps": f["steps"][:],
            "ee_pos": f["ee_pos"][:],
            "torque": f["torque"][:],
            "block_positions": f["block_positions"][:],
            "rewards": f["rewards"][:],
            "instruction": f.attrs.get("instruction", ""),
            "success": bool(f.attrs.get("success", False)),
            "num_steps": int(f.attrs.get("num_steps", 0)),
        }


def _load_csv(path: str) -> Dict:
    """Fallback loader for legacy CSV format (no eval()!)."""
    df = pd.read_csv(path)
    # Coerce numeric columns — strings become NaN, no code execution
    for col in ("gripper_x", "gripper_y", "gripper_z",
                "torque_0", "torque_1", "torque_2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return {
        "steps": df.get("step", np.arange(len(df))).values.astype(int),
        "ee_pos": df[["gripper_x", "gripper_y", "gripper_z"]].values.astype(float)
                  if {"gripper_x", "gripper_y", "gripper_z"}.issubset(df.columns)
                  else np.zeros((len(df), 3)),
        "torque": df[["torque_0", "torque_1", "torque_2"]].values.astype(float)
                  if {"torque_0", "torque_1", "torque_2"}.issubset(df.columns)
                  else np.zeros((len(df), 3)),
        "block_positions": np.zeros((len(df), 9)),
        "rewards": np.zeros(len(df)),
        "instruction": "",
        "success": False,
        "num_steps": len(df),
    }


def render_trajectory_analytics(
    log_path: str,
    output_html: Optional[str] = None,
    output_png: Optional[str] = None,
    compare_paths: Optional[List[str]] = None,
    workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    """Render an interactive Plotly trajectory plot.

    Args:
        log_path: path to the episode HDF5 file (or legacy CSV).
        output_html: where to write the interactive HTML. Default: alongside log.
        output_png: where to write a static PNG. Default: alongside log.
        compare_paths: optional list of additional episode paths to overlay.
        workspace_bounds: optional dict with keys 'x', 'y', 'z' each mapping
            to (min, max) tuple. Draws a transparent box overlay.

    Returns:
        Path to the generated HTML file.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Trajectory log not found: {log_path}")

    if log_path.endswith(".h5"):
        data = _load_h5(log_path)
    else:
        data = _load_csv(log_path)

    base_dir = os.path.dirname(log_path) or "."
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    if output_html is None:
        output_html = os.path.join(base_dir, f"{base_name}_trajectory.html")
    if output_png is None:
        output_png = os.path.join(base_dir, f"{base_name}_trajectory.png")

    x, y, z = data["ee_pos"][:, 0], data["ee_pos"][:, 1], data["ee_pos"][:, 2]
    steps = data["steps"]
    # Normalize step index -> [0, 1] for color encoding
    if len(steps) > 1:
        t_norm = (steps - steps.min()) / (steps.max() - steps.min())
    else:
        t_norm = np.array([0.0])

    # Build figure with 1 row, 2 cols: 3D trajectory + per-step torque/reward
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(
            "3D End-Effector Trajectory (color = time)",
            "Per-Step Torque & Reward",
        ),
    )

    # --- 3D trajectory ---
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+lines",
            line=dict(width=4, color=t_norm, colorscale="Viridis"),
            marker=dict(size=4, color=t_norm, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Step", x=0.45)),
            name="End-effector",
            text=[f"step {s}<br>reward={r:.3f}"
                  for s, r in zip(steps, data["rewards"])],
            hoverinfo="text+x+y+z",
        ),
        row=1, col=1,
    )
    # Start/End markers
    fig.add_trace(
        go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers", marker=dict(size=10, color="green", symbol="circle"),
            name="Start", showlegend=True,
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers", marker=dict(size=10, color="red", symbol="x"),
            name="End", showlegend=True,
        ), row=1, col=1,
    )

    # Block positions (3 blocks × 3 coords) — show as separate trace per block
    for i, color in enumerate(("red", "green", "blue")):
        if data["block_positions"].shape[1] >= (i + 1) * 3:
            bx = data["block_positions"][:, i * 3]
            by = data["block_positions"][:, i * 3 + 1]
            bz = data["block_positions"][:, i * 3 + 2]
            if np.any(np.linalg.norm(
                data["block_positions"][:, i * 3: i * 3 + 3], axis=1,
            ) > 1e-6):
                fig.add_trace(
                    go.Scatter3d(
                        x=bx, y=by, z=bz,
                        mode="markers+lines",
                        line=dict(width=2, dash="dot"),
                        marker=dict(size=4),
                        name=f"{color} block",
                    ),
                    row=1, col=1,
                )

    # Workspace bounds overlay
    if workspace_bounds:
        x_lim = workspace_bounds.get("x", (-0.5, 0.5))
        y_lim = workspace_bounds.get("y", (-0.5, 0.5))
        z_lim = workspace_bounds.get("z", (0.0, 1.0))
        corners = [
            (x_lim[0], y_lim[0], z_lim[0]),
            (x_lim[1], y_lim[0], z_lim[0]),
            (x_lim[1], y_lim[1], z_lim[0]),
            (x_lim[0], y_lim[1], z_lim[0]),
            (x_lim[0], y_lim[0], z_lim[1]),
            (x_lim[1], y_lim[0], z_lim[1]),
            (x_lim[1], y_lim[1], z_lim[1]),
            (x_lim[0], y_lim[1], z_lim[1]),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[corners[i][0], corners[j][0]],
                    y=[corners[i][1], corners[j][1]],
                    z=[corners[i][2], corners[j][2]],
                    mode="lines",
                    line=dict(color="gray", width=2, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

    # Episode comparison overlay
    if compare_paths:
        for ep_path in compare_paths:
            if not os.path.exists(ep_path):
                continue
            try:
                ep_data = _load_h5(ep_path) if ep_path.endswith(".h5") else _load_csv(ep_path)
            except Exception:
                continue
            ex, ey, ez = ep_data["ee_pos"][:, 0], ep_data["ee_pos"][:, 1], ep_data["ee_pos"][:, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=ex, y=ey, z=ez,
                    mode="lines",
                    line=dict(width=2, dash="dot"),
                    name=f"compare: {os.path.basename(ep_path)}",
                ),
                row=1, col=1,
            )

    # --- Per-step torque & reward ---
    for i, label in enumerate(("τ_shoulder", "τ_elbow", "τ_wrist")):
        if data["torque"].shape[1] > i:
            fig.add_trace(
                go.Scatter(
                    x=steps, y=data["torque"][:, i],
                    mode="lines", name=label,
                ),
                row=1, col=2,
            )
    fig.add_trace(
        go.Scatter(
            x=steps, y=data["rewards"],
            mode="lines", name="reward",
            line=dict(dash="dash", color="black"),
            yaxis="y2",
        ),
        row=1, col=2,
    )

    # Layout
    success_str = "✓ SUCCESS" if data["success"] else "✗ NOT SOLVED"
    title = (
        f"VLA Push-Block Trajectory — {success_str}<br>"
        f"<sub>instruction: \"{data['instruction']}\" | "
        f"steps: {data['num_steps']}</sub>"
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=1500, height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"[VISUALIZER] Interactive HTML: {output_html}")

    # PNG export (best-effort — kaleido not always installed)
    try:
        fig.write_image(output_png, width=1500, height=700, scale=2)
        print(f"[VISUALIZER] Static PNG: {output_png}")
    except Exception as e:
        # Fallback: matplotlib static PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig_m, ax_m = plt.subplots(figsize=(10, 7), subplot_kw={"projection": "3d"})
            ax_m.plot(x, y, z, color="crimson", linewidth=2, marker="o")
            ax_m.scatter([x[0]], [y[0]], [z[0]], color="green", s=100, label="Start")
            ax_m.scatter([x[-1]], [y[-1]], [z[-1]], color="blue", s=100, label="End")
            ax_m.set_xlabel("X (m)"); ax_m.set_ylabel("Y (m)"); ax_m.set_zlabel("Z (m)")
            ax_m.set_title(f"VLA Trajectory — {success_str}\n{data['instruction']}")
            ax_m.legend()
            fig_m.tight_layout()
            fig_m.savefig(output_png, dpi=150)
            plt.close(fig_m)
            print(f"[VISUALIZER] Static PNG (matplotlib fallback): {output_png}")
        except Exception as e2:
            print(f"[VISUALIZER] PNG export skipped: {e}; matplotlib fallback failed: {e2}")

    return output_html


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # Generate a synthetic episode for testing
    with tempfile.TemporaryDirectory() as tmp:
        from src.utils.data_logger import ProductionDataLogger

        logger = ProductionDataLogger(log_dir=tmp)
        for i in range(20):
            img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            t = i / 19.0
            ee = np.array([0.1 + 0.4 * t, 0.0 + 0.1 * np.sin(t * 6), 0.5], dtype=np.float32)
            logger.record_step(
                step_idx=i,
                image_frame=img,
                instruction="push the red block to the red zone",
                action_torque=np.array([0.1, -0.2, 0.3], dtype=np.float32),
                end_effector_pos=ee,
                reward=-0.5 + t,
                block_positions=np.array([0.3, 0.0, 0.5, 0.0, 0.15, 0.5, 0.0, -0.15, 0.5], dtype=np.float32),
            )
        path = logger.save_episode(
            instruction="push the red block to the red zone",
            success=False,
            total_reward=-5.0,
        )
        render_trajectory_analytics(
            path,
            workspace_bounds={"x": (-0.6, 0.6), "y": (-0.4, 0.4), "z": (0.4, 1.0)},
        )
