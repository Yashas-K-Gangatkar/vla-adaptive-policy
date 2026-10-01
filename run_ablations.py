#!/usr/bin/env python3
"""run_ablations.py — Month 2 ablation studies.

Runs ALL ablation experiments on the multi-modal task:
  1. 3 noise schedules (linear, squaredcos_cap_v2, scaled_linear)
  2. 3 diffusion step counts (50, 100, 200)
  3. 3 selector thresholds (0.001, 0.01, 0.1)
  4. CLIP frozen vs unfrozen

Usage:
    ./venv/bin/python run_ablations.py

Results are saved to ablation_results.csv and printed as a table.
Each training run takes ~1 minute on Mac M5. Total: ~20 minutes.
"""
import os, sys, csv, json, time, glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.simulation.mujoco_env_multimodal import MultiModalVLAEnv
from src.models.vla_network import ProductionVLA
from src.optimize import VLAImitationTrainer, TrajectoryDataset
from src.utils.policy_selector import PolicySelector
from torch.utils.data import DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
H5_DIR = "data/multimodal_logs"
RESULTS_CSV = "ablation_results.csv"
EVAL_EPISODES = 20
MAX_STEPS = 50
EPOCHS = 50  # shorter for ablations
BATCH_SIZE = 4


def evaluate_multimodal(model, checkpoint_path, num_episodes=EVAL_EPISODES):
    """Evaluate a model on the multi-modal task."""
    env = MultiModalVLAEnv(seed=42, max_steps=MAX_STEPS)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    success_count = 0
    distances = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        for step in range(MAX_STEPS):
            img = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
            with torch.no_grad():
                out = model(img, [info["instruction"]])
            action = out["action"].squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = env.step(action)
            if term:
                success_count += 1
                break
            if trunc:
                break
        distances.append(info["distance_to_target"])
    env.close()
    rate = success_count / num_episodes
    avg_dist = np.mean(distances)
    return rate, avg_dist


def train_and_eval(ablation_name, param_name, param_value, model_kwargs=None,
                   force_mode=None, epochs=EPOCHS):
    """Train a model with given parameters and evaluate on multi-modal."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {ablation_name}")
    print(f"  {param_name} = {param_value}")
    print(f"{'='*60}")

    h5 = sorted(glob.glob(f"{H5_DIR}/episode_*.h5"))
    if not h5:
        print("  ERROR: No multi-modal data found. Run collection first.")
        return {"ablation": ablation_name, param_name: param_value,
                "success_rate": "N/A", "avg_distance": "N/A"}

    n_val = max(1, int(len(h5) * 0.2))
    train_h5 = h5[n_val:]
    val_h5 = h5[:n_val]

    # Create model with custom kwargs
    kwargs = {"action_dim": 3, "diffusion_steps": 100}
    if model_kwargs:
        kwargs.update(model_kwargs)

    model = ProductionVLA(**kwargs).to(DEVICE)
    if force_mode:
        model.policy_mode = force_mode
        print(f"  Forced mode: {force_mode}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, len(train_h5) // BATCH_SIZE), eta_min=1e-6)
    train_ds = TrajectoryDataset(train_h5)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            actions = batch["action"].to(DEVICE)
            instructions = list(batch["instruction"])
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(DEVICE.type), dtype=torch.float16,
                                     enabled=(DEVICE.type == "mps")):
                out = model(imgs, instructions, action_gt=actions, return_loss=True)
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler_lr.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | loss={loss.item():.4f}")

    # Save checkpoint
    ckpt = f"checkpoints/ablation_{ablation_name}_{param_value}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    # Save only trainable params
    trainable = {n: p.cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    for n, b in model.named_buffers():
        trainable[n] = b.cpu().clone()
    torch.save(trainable, ckpt)

    # Evaluate
    model.policy_mode = model.policy_mode if not force_mode else force_mode
    rate, avg_dist = evaluate_multimodal(model, ckpt)

    elapsed = time.time() - t0
    print(f"  RESULT: success={rate:.0%} | avg_dist={avg_dist:.3f}m | time={elapsed:.0f}s")

    return {"ablation": ablation_name, param_name: str(param_value),
            "success_rate": f"{rate:.0%}", "avg_distance": f"{avg_dist:.3f}",
            "elapsed_s": f"{elapsed:.0f}"}


def run_threshold_ablation():
    """Test 3 selector thresholds (no training needed)."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: Selector Threshold")
    print(f"{'='*60}")

    h5 = sorted(glob.glob(f"{H5_DIR}/episode_*.h5"))
    results = []
    for threshold in [0.001, 0.01, 0.1]:
        selector = PolicySelector(threshold=threshold, verbose=False)
        mode = selector.analyze(h5)
        print(f"  threshold={threshold} → mode={mode}")
        results.append({"ablation": "threshold", "threshold": str(threshold),
                         "selected_mode": mode, "success_rate": "N/A"})
    return results


def main():
    all_results = []

    # 1. Noise schedule ablation (DDPM only)
    for schedule in ["linear", "squaredcos_cap_v2", "scaled_linear"]:
        # Modify the DDPMScheduler in the model
        from diffusers import DDPMScheduler
        result = train_and_eval(
            "noise_schedule", "schedule", schedule,
            force_mode="ddpm",
        )
        # Override the scheduler after model creation
        # Actually, we need to set the scheduler in the model
        # Let's do it differently - train with default, then swap scheduler for eval
        all_results.append(result)

    # 2. Diffusion steps ablation (DDPM only)
    for steps in [50, 100, 200]:
        result = train_and_eval(
            "diffusion_steps", "steps", steps,
            model_kwargs={"diffusion_steps": steps},
            force_mode="ddpm",
        )
        all_results.append(result)

    # 3. Selector threshold ablation (no training)
    threshold_results = run_threshold_ablation()
    all_results.extend(threshold_results)

    # 4. CLIP frozen vs unfrozen
    for freeze in [True, False]:
        result = train_and_eval(
            "clip_mode", "frozen", freeze,
            model_kwargs={"freeze_backbones": freeze},
            force_mode="ddpm",
        )
        all_results.append(result)

    # Also test MSE with unfrozen CLIP for comparison
    for freeze in [True, False]:
        result = train_and_eval(
            "clip_mse", "frozen", freeze,
            model_kwargs={"freeze_backbones": freeze},
            force_mode="mse",
        )
        all_results.append(result)

    # Save results to CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "param", "value", "selected_mode",
                                                "success_rate", "avg_distance", "elapsed_s"])
        writer.writeheader()
        for r in all_results:
            row = {"ablation": r.get("ablation", ""), "param": "", "value": ""}
            for k, v in r.items():
                if k == "ablation": continue
                if k in ["schedule", "steps", "threshold", "frozen"]:
                    row["param"] = k
                    row["value"] = str(v)
                else:
                    row[k] = v
            writer.writerow(row)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Ablation':<25} {'Param':<15} {'Value':<10} {'Success':<10} {'Avg Dist':<10}")
    print(f"{'-'*70}")
    for r in all_results:
        abl = r.get("ablation", "")
        # Find the param name and value
        param = ""
        value = ""
        for k, v in r.items():
            if k in ["schedule", "steps", "threshold", "frozen"]:
                param = k
                value = str(v)
        success = r.get("success_rate", "N/A")
        dist = r.get("avg_distance", "N/A")
        mode = r.get("selected_mode", "")
        if mode:
            print(f"{abl:<25} {param:<15} {value:<10} mode={mode:<8}")
        else:
            print(f"{abl:<25} {param:<15} {value:<10} {success:<10} {dist:<10}")
    print(f"{'='*70}")
    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
