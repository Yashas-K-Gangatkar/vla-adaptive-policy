# VLA Adaptive Policy Selector

**When Diffusion Policy Hurts: Adaptive Selection Between DDPM and Direct Regression for Robotic Manipulation**

![Status](https://img.shields.io/badge/status-research-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13%2B-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red)
![Success](https://img.shields.io/badge/evaluated-2%20tasks-brightgreen)

## Overview

An open-source Vision-Language-Action (VLA) pipeline that **automatically selects** between DDPM diffusion policy and direct MSE regression based on the action distribution's modality. Evaluated on two tasks:

- **Deterministic** (1 valid action per instruction): MSE achieves **100%**, DDPM achieves **0-15%**
- **Multi-modal** (2 valid actions per instruction): DDPM achieves **40%**, MSE achieves **30%**

The Adaptive Policy Selector correctly identifies each task type and selects the method that maximizes success.

## Key Results — The 2×2 Table

| Task | Method | Success Rate | Avg Distance |
|------|:------:|:------------:|:------------:|
| Deterministic (1 zone) | DDPM | 0-15% | 42-49 cm |
| Deterministic (1 zone) | **MSE** | **100%** | 5.8-13 cm |
| Multi-modal (2 zones) | **DDPM** | **40%** | Variable |
| Multi-modal (2 zones) | MSE | 30% | 5.8-17 cm |

**The selector picks MSE for deterministic → 100%. The selector picks DDPM for multi-modal → 40%.**

### MSE on Multi-modal: Breakdown

| Color | Zones | MSE Success | Why |
|-------|:-----:|:-----------:|-----|
| Green | 1 (deterministic) | 6/6 = 100% | MSE predicts the single correct action |
| Red | 2 (multi-modal) | 0/4 = 0% | MSE averages the two modes → misses both |
| Blue | 2 (multi-modal) | 0/10 = 0% | Same — averages to middle |

### DDPM on Multi-modal: Breakdown

| Color | Zones | DDPM Success | Why |
|-------|:-----:|:------------:|-----|
| Green | 1 (deterministic) | 4/6 = 67% | DDPM works but some stochastic variance |
| Red | 2 (multi-modal) | 1/4 = 25% | DDPM samples one of the two modes |
| Blue | 2 (multi-modal) | 3/10 = 30% | DDPM samples one of the two modes |

**Key insight:** MSE succeeds ONLY on deterministic episodes. DDPM succeeds on ALL colors. DDPM captures multi-modality; MSE collapses it to the mean.

## Novel Contribution: Adaptive Policy Selector

Nobody has published an automatic method for choosing between diffusion and regression based on action distribution modality. Our selector:

1. Analyzes the training data's action distribution (grouped by instruction)
2. Computes the standard deviation of expert actions per instruction
3. If `max_std < 0.01` → task is deterministic → **use MSE**
4. If `max_std >= 0.01` → task is multimodal → **use DDPM**

### Selector Validation

| Data Type | Action std | Selector Output | Correct? |
|-----------|:----------:|:---------------:|:--------:|
| Deterministic (1 zone per color) | 0.000 | MSE | ✅ |
| Multi-modal (2 zones for red/blue) | 0.099 | DDPM | ✅ |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    VLA Pipeline                           │
│                                                          │
│  ┌─────────┐   ┌─────────┐   ┌──────────────────────┐   │
│  │  CLIP   │   │  CLIP   │   │  8-Head Cross-Attn   │   │
│  │  Vision │ + │  Text   │ → │  (Q=vision, KV=text) │   │
│  │ (frozen)│   │ (frozen)│   │  → 256-d cond vector │   │
│  └─────────┘   └─────────┘   └──────────┬───────────┘   │
│                                          │               │
│                              ┌───────────┴───────────┐   │
│                              │  Adaptive Selector     │   │
│                              │  std < 0.01 → MSE     │   │
│                              │  std >= 0.01 → DDPM   │   │
│                              └───────────┬───────────┘   │
│                              ┌───────────┴───────────┐   │
│                              │  FiLM Action Denoiser  │   │
│                              │  (2.4M trainable params)│   │
│                              │  4 residual blocks      │   │
│                              │  Sinusoidal time embed │   │
│                              └───────────┬───────────┘   │
│                                          │               │
│                              ┌───────────┴───────────┐   │
│                              │  3-DOF MuJoCo Arm      │   │
│                              │  3 colored zones       │   │
│                              │  Language-conditioned  │   │
│                              └───────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
git clone https://github.com/Yashas-K-Gangatkar/vla-adaptive-policy.git
cd vla-adaptive-policy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Task 1: Deterministic (selector chooses MSE)
python -m src.train --collect-only --episodes 100 --max-steps 50
python -m src.optimize --epochs 100 --batch-size 4 --no-wandb
python -m src.train --evaluate --episodes 20 --max-steps 50 --checkpoint checkpoints/vla_final.pt

# Task 2: Multi-modal (selector chooses DDPM)
# See src/simulation/mujoco_env_multimodal.py for the multi-modal env
```

## File Structure

```
vla-adaptive-policy/
├── src/
│   ├── simulation/
│   │   ├── mujoco_env.py              # Deterministic env (1 zone per color)
│   │   └── mujoco_env_multimodal.py   # Multi-modal env (2 zones for red/blue)
│   ├── models/
│   │   └── vla_network.py              # CLIP + cross-attn + FiLM denoiser (dual-mode)
│   ├── utils/
│   │   ├── data_logger.py             # HDF5 trajectory logging
│   │   ├── plotter.py                  # Plotly 3D trajectory visualization
│   │   ├── scripted_expert.py          # Jacobian IK expert (100% success)
│   │   └── policy_selector.py          # ⭐ ADAPTIVE POLICY SELECTOR (novel)
│   ├── optimize.py                    # Trainer (AMP + cosine LR + grad clip)
│   └── train.py                       # Closed-loop pipeline (collect + evaluate)
├── README.md
├── requirements.txt
├── PAPER_DRAFT.md                     # Workshop paper draft
├── VLA_Paper_Final.pdf                # Formatted paper PDF
├── patch_adaptive.py                  # Integration script
└── LICENSE                            # MIT
```

## Citation

```bibtex
@misc{gangatkar2026vla,
  title={When Diffusion Policy Hurts: Adaptive Selection Between DDPM and Direct Regression for Robotic Manipulation},
  author={Gangatkar, Yashas K},
  year={2026},
  howpublished={\url{https://github.com/Yashas-K-Gangatkar/vla-adaptive-policy}}
}
```

## License

MIT License — free for academic and commercial use.

## Acknowledgments

- CLIP pretrained weights: OpenAI (clip-vit-base-patch32)
- MuJoCo physics engine: Google DeepMind
- Diffusers library: HuggingFace
