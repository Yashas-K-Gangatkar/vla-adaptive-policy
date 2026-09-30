# VLA Adaptive Policy Selector

**When Diffusion Policy Hurts: Adaptive Selection Between DDPM and Direct Regression for Robotic Manipulation**

![Status](https://img.shields.io/badge/status-research-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13%2B-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-red)
![Success](https://img.shields.io/badge/success%20rate-100%25-brightgreen)

## Overview

An open-source Vision-Language-Action (VLA) pipeline that **automatically selects** between DDPM diffusion policy and direct MSE regression based on the action distribution's modality. On deterministic single-step manipulation tasks, direct MSE achieves **100% success** while DDPM achieves only **0-15%** — our Adaptive Policy Selector correctly identifies the task as deterministic and selects MSE.

## Key Results

| Method | Success Rate (15cm) | Avg Distance |
|--------|:-------------------:|:------------:|
| DDPM (100 steps, 1 sample) | 0-5% | 42-49 cm |
| DDPM (20-sample averaging) | 15% | 10-15 cm |
| **Direct MSE (our selector)** | **100%** | **5.8-13 cm** |
| Expert (Jacobian IK) | 100% | 4-8 mm |

## Novel Contribution: Adaptive Policy Selector

Nobody has published an automatic method for choosing between diffusion and regression based on action distribution modality. Our selector:

1. Analyzes the training data's action distribution (grouped by instruction)
2. Computes the standard deviation of expert actions per instruction
3. If `max_std < 0.01` → task is deterministic → **use MSE** (deterministic, fast)
4. If `max_std >= 0.01` → task is multimodal → **use DDPM** (stochastic, captures multi-modality)

```python
from src.utils.policy_selector import PolicySelector
selector = PolicySelector(threshold=0.01)
mode = selector.analyze(h5_paths)  # Returns "mse" or "ddpm"
```

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
# Clone
git clone https://github.com/Yashas-K-Gangatkar/vla-adaptive-policy.git
cd vla-adaptive-policy

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python -m src.train --collect-only --episodes 100 --max-steps 50
python -m src.optimize --epochs 100 --batch-size 4 --no-wandb
python -m src.train --evaluate --episodes 20 --max-steps 50 --checkpoint checkpoints/vla_final.pt
```

## File Structure

```
vla-adaptive-policy/
├── src/
│   ├── simulation/
│   │   └── mujoco_env.py          # 3-DOF arm + 3 colored zones (gym.Env)
│   ├── models/
│   │   └── vla_network.py          # CLIP + cross-attn + FiLM denoiser
│   ├── utils/
│   │   ├── data_logger.py          # HDF5 trajectory logging
│   │   ├── plotter.py             # Plotly 3D trajectory visualization
│   │   ├── scripted_expert.py     # Jacobian IK expert (100% success)
│   │   └── policy_selector.py     # ⭐ ADAPTIVE POLICY SELECTOR (novel)
│   ├── optimize.py                # Trainer (AMP + cosine LR + grad clip)
│   └── train.py                   # Closed-loop pipeline (collect + evaluate)
├── README.md
├── requirements.txt
├── PAPER_DRAFT.md                 # 4-page workshop paper draft
└── patch_adaptive.py              # Integration script
```

## Why DDPM Fails on Deterministic Tasks

DDPM's reverse process starts from random Gaussian noise and adds stochastic noise at each of 100 denoising steps. For a **deterministic task** (where each input maps to exactly ONE correct action), this stochastic variance is pure noise — it pushes the sampled action AWAY from the correct one.

| | DDPM | Direct MSE |
|---|---|---|
| Sampling | 100-step stochastic reverse chain | 1 forward pass |
| Output for same input | Different every time | Same every time |
| Variance | 0.08-0.12 per component | 0 |
| Success on deterministic task | 0-15% | 80-100% |

**The diagnostic**: Compute action std on training data. If `std < 0.01`, use MSE. Otherwise, use DDPM.

## Citation

```bibtex
@misc{gangatkar2026vla,
  title={When Diffusion Policy Hurts: Adaptive Selection Between DDPM and Direct Regression for Robotic Manipulation},
  author={Gangatkar, Yashas K and Team B25\_86},
  year={2026},
  howpublished={CoRE-SoET Research Initiative, Sapthagiri NPS University},
  url={https://github.com/Yashas-K-Gangatkar/vla-adaptive-policy}
}
```

## Team

**Team B25_86** — CoRE-SoET Research Initiative, Sapthagiri NPS University
- Topic: T036 — AI-Based Autonomous Robot for Industrial Material Handling

## License

MIT License — free for academic and commercial use.

## Acknowledgments

- CLIP pretrained weights: OpenAI (clip-vit-base-patch32)
- MuJoCo physics engine: Google DeepMind
- Diffusers library: HuggingFace
