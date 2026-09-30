#!/usr/bin/env python3
"""Patch optimize.py and train.py to use the Adaptive Policy Selector.

This adds automatic mode selection: the trainer analyzes the training data
and chooses between DDPM (diffusion) and MSE (regression) based on the
action distribution's modality.

Run from the vla_robotics_product/ directory:
    python3 patch_adaptive.py
"""
import os, re

# ─── Patch optimize.py ───
p = 'src/optimize.py'
with open(p) as f:
    c = f.read()

# Add import for PolicySelector
if 'PolicySelector' not in c:
    c = c.replace(
        'from src.models.vla_network import ProductionVLA',
        'from src.models.vla_network import ProductionVLA\nfrom src.utils.policy_selector import PolicySelector'
    )

# Add policy selection after model creation
# Find the line after model creation and add selector call
old_trainer_init = '        print(f"[TRAINER] trainable params: {n_trainable/1e6:.1f}M / total {n_total/1e6:.1f}M")'
new_trainer_init = '''        print(f"[TRAINER] trainable params: {n_trainable/1e6:.1f}M / total {n_total/1e6:.1f}M")

        # === ADAPTIVE POLICY SELECTION ===
        # Analyze training data to determine if task is deterministic or multi-modal
        selector = PolicySelector(threshold=0.01, verbose=True)
        selected_mode = selector.analyze(train_h5)
        self.model.policy_mode = selected_mode
        print(f"[TRAINER] Selected policy mode: {selected_mode.upper()}")'''

c = c.replace(old_trainer_init, new_trainer_init)

with open(p, 'w') as f:
    f.write(c)
print(f"[PATCHED] {p}")

# ─── Patch train.py (evaluate function) ───
p = 'src/train.py'
with open(p) as f:
    c = f.read()

# Add import
if 'PolicySelector' not in c:
    c = c.replace(
        'from src.utils.scripted_expert import scripted_expert_action',
        'from src.utils.scripted_expert import scripted_expert_action\nfrom src.utils.policy_selector import PolicySelector'
    )

# After loading checkpoint, analyze eval data and set mode
old_eval = '    model.eval()'
# Find the SECOND occurrence (in evaluate, not collect)
parts = c.split('model.eval()')
if len(parts) >= 3:
    insertion = '''model.eval()

    # === ADAPTIVE POLICY SELECTION for eval ===
    import glob as _glob
    _eval_h5 = sorted(_glob.glob("data/sim_logs/episode_*.h5"))
    if _eval_h5:
        _selector = PolicySelector(threshold=0.01, verbose=True)
        _mode = _selector.analyze(_eval_h5)
        if hasattr(model, 'policy_mode'):
            model.policy_mode = _mode
            print(f"[EVAL] Policy mode: {_mode.upper()}")'''
    c = parts[0] + 'model.eval()' + parts[1] + insertion + parts[2]

with open(p, 'w') as f:
    f.write(c)
print(f"[PATCHED] {p}")

print("\n✅ Adaptive Policy Selector integrated!")
print("The trainer will now automatically choose between DDPM and MSE.")
print("Run: ./venv/bin/python -m src.optimize --epochs 100 --batch-size 4 --no-wandb")
