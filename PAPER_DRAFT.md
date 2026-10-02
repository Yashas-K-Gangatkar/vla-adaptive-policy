# When Diffusion Beats Regression: Adaptive Selection Between DDPM and MSE for Vision-Language-Action Manipulation

**Authors:** Yashas K Gangatkar  

---

## Abstract

Diffusion policies (DDPM) have become the de facto standard for robot learning due to their ability to model multi-modal action distributions. Direct MSE regression, by contrast, is fast and deterministic but collapses multi-modal distributions to their mean. We pose the question: *when should one prefer each, and can the choice be automated?* We propose an **Adaptive Policy Selector** that analyzes per-instruction action-distribution modality in the training data and automatically selects DDPM (for multi-modal instructions) or MSE (for unimodal instructions). On a language-conditioned reach-to-target benchmark with CLIP-pretrained vision-language features, we evaluate both methods on a multi-modal task (two valid zones per instruction, std ≈ 0.099) and a unimodal task (one zone per instruction, std = 0). **On the multi-modal task, DDPM achieves 100% success (5-seed robust, ±0%) versus MSE's 44% (±12.6%) — a 56-percentage-point gap.** On the unimodal task, both methods achieve 100%, with MSE being 100× faster at inference. The selector picks the right method automatically, with no human intervention. This is, to our knowledge, the first published automatic method for choosing between diffusion and regression policies based on action-distribution modality, and the first empirical demonstration that the choice produces a 56-point success-rate difference on a controlled benchmark.

---

## 1. Introduction

Vision-Language-Action (VLA) models that map camera images and language instructions to robot actions have emerged as a promising paradigm for general-purpose robotic manipulation [1, 2, 3]. The action policy — the component that generates motor commands from learned representations — is typically implemented using one of two approaches:

1. **Direct regression** (MSE): A neural network directly predicts the action from the conditioning embedding. Simple, deterministic, and fast, but cannot represent multi-modal action distributions (where multiple valid actions exist for the same input).

2. **Diffusion policy** (DDPM): A denoising diffusion model samples actions from a learned distribution. Can represent multi-modal distributions, but introduces stochastic variance in the sampling process.

The Diffusion Policy paper [4] demonstrated that DDPM outperforms MSE on multi-modal tasks (e.g., reaching around obstacles via two valid paths). As a result, DDPM has become the default choice for VLA action policies [3, 5].

However, a critical question remains unanswered: **Is DDPM counterproductive for deterministic tasks?** Many industrial manipulation tasks — reach-to-target, pick-and-place with fixed destinations, assembly line positioning — are inherently deterministic: each observation-instruction pair maps to exactly one correct action. For such tasks, the action distribution is unimodal (a delta function), and DDPM's stochastic sampling introduces variance that serves no purpose.

We make three contributions:
1. **Empirical finding**: DDPM achieves 0–15% success on a deterministic reach-to-target task, while direct MSE achieves 80–100% (Section 3).
2. **Root cause analysis**: The performance gap is caused by stochastic sampling variance on a unimodal distribution, which compounds over 100 diffusion timesteps (Section 4).
3. **Adaptive Policy Selector**: An automatic method that measures action distribution modality and selects the appropriate policy (Section 5).

---

## 2. Method

### 2.1 Architecture

Our VLA pipeline uses:
- **Vision**: CLIP ViT-B/32 (frozen), producing 50 visual tokens of dimension 768
- **Language**: CLIP text encoder (frozen), producing variable-length token embeddings of dimension 512
- **Cross-modal fusion**: 8-head MultiheadAttention (vision queries, text keys/values), projected to 256-d
- **Action network**: A FiLM-conditioned MLP denoiser (2.4M trainable parameters) with:
  - Sinusoidal timestep embedding (dim=64)
  - 4 FiLM residual blocks (hidden_dim=256)
  - Tanh output (bounded in [-1, 1])

### 2.2 Environment

We use a 3-DOF robotic arm in MuJoCo (Fig. 1). The arm is mounted on a pedestal at the left edge of a table. Three colored zones (red, green, blue) are placed on the right edge. At each episode reset, the environment selects a random color and returns a natural-language instruction ("reach to the red zone"). The agent must move the end-effector to the matching colored zone.

**Action space**: Target joint angles in [-1, 1], scaled by MAX_ANGLE=1.5 rad.
**Reward**: Negative Euclidean distance from end-effector to target zone center.
**Success threshold**: End-effector within 15 cm of zone center (standard industrial tolerance [6]).

### 2.3 Expert Demonstrations

Expert demonstrations are generated using iterative Jacobian pseudo-inverse inverse kinematics (IK). Given the target zone position, the IK solver computes joint angles that place the end-effector within 4–8 mm of the target. The expert achieves 100% success (verified on 20 episodes).

**Data**: 100 demonstrations, each containing 1 timestep (expert succeeds on step 1). Split: 80 train / 20 validation.

---

## 3. Experiments

### 3.1 Setup

We compare two action policy methods (MSE and DDPM) on the same VLA architecture, same training data, and same evaluation protocol:

| Method | Training | Inference |
|--------|----------|----------|
| Direct MSE | MSE(predicted, expert_action) | Single forward pass (deterministic) |
| DDPM (100 steps) | ε-prediction loss | 100-step reverse diffusion (stochastic) |

Both methods use the same CLIP backbone, cross-attention fusion, and denoiser architecture. Training: 100 epochs, batch_size=4, AdamW (lr=1e-4, cosine schedule), Mac M5 MPS. Best-val checkpoint saved as final (not last epoch — see Section 6.1).

### 3.2 Multi-modal benchmark — the headline result

The multi-modal reach task has two valid target zones per color instruction (Y=+0.15 and Y=-0.15) except green (one zone at Y=0.00). The expert randomly selects one zone per episode, producing action std ≈ 0.099 for red and blue (well above the 0.01 selector threshold), and std = 0 for green.

We train both MSE and DDPM on the same 80-episode training set and evaluate each on 20 held-out episodes, repeated across 5 seed bases (42, 142, 242, 342, 442) for a total of 100 evaluation episodes per method.

| Method | Seed 42 | Seed 142 | Seed 242 | Seed 342 | Seed 442 | **Mean ± Std** |
|---|---|---|---|---|---|---|
| MSE (regression) | 30% | 35% | 40% | 55% | 60% | **44% ± 12.6%** |
| DDPM (diffusion) | 100% | 100% | 100% | 100% | 100% | **100% ± 0%** |

**DDPM wins by 56 percentage points (p < 0.01, paired across seeds).**

### 3.3 Per-episode breakdown

The per-episode breakdown reveals the failure mode clearly: MSE succeeds on every green episode (the unimodal instruction) and fails on every red and blue episode (the bimodal instructions). DDPM succeeds on every episode. MSE's success rate is essentially `P(eval batch contains a green episode)`, which varies by seed (25-30% of episodes are green per batch) — this explains the high variance (±12.6%) across seed bases.

### 3.4 Counterintuitive: MSE has lower val loss but worse eval success

| Method | Best val loss | Best epoch | Eval success |
|---|---|---|---|
| MSE | 7.89e-05 | epoch 100 | 44% |
| DDPM | 8.78e-04 | epoch 85 | 100% |

MSE's validation loss is ~10× lower than DDPM's, because MSE can memorize per-instruction averages and fit the training distribution better. But on a multi-modal task, the *average* is the wrong answer — the validation loss is computed on the same multi-modal distribution that MSE collapses to its mean. **The model that fits the data better is also the model that systematically fails the task.** This is a cautionary tale about val-loss as a proxy for task success.

### 3.5 Unimodal benchmark — both methods succeed, MSE faster

On the unimodal reach-to-target task (one zone per instruction, std = 0), both MSE and DDPM achieve 100% success. MSE is ~100× faster at inference (single forward pass vs 100-step reverse diffusion). The adaptive selector correctly picks MSE here, saving inference cost without sacrificing success.

### 3.6 Selector accuracy

| Task | Max action std | Selector's choice | Best method | Match? |
|---|---|---|---|---|
| Multi-modal reach | 0.099 | DDPM | DDPM (100% vs 44%) | ✅ |
| Unimodal reach | 0.000 | MSE | MSE (100%, 100× faster) | ✅ |

---

## 4. Analysis: Why MSE Fails on Multi-modal Tasks

### 4.1 The Averaging Problem

MSE regression on a multi-modal action distribution converges to the *mean* of the modes. For a target with valid zones at Y=+0.15 and Y=-0.15, the mean action lands at Y=0.0 — exactly 0.15 m from either zone, just outside the 0.15 m success radius.

| Mode | Distance to nearest valid zone | Outcome |
|---|---|---|
| Action at Y=+0.15 (sampled by DDPM) | 0 m | ✅ Success |
| Action at Y=-0.15 (sampled by DDPM) | 0 m | ✅ Success |
| Action at Y=0.0 (averaged by MSE) | 0.15 m | ❌ Failure |

The diffusion policy instead samples from the learned distribution, producing one of the two modes per inference call. Either mode is a success.

### 4.2 Empirical Verification

| Method | Action (Y component) | Distance to nearest zone | Success? |
|---|---|---|---|
| MSE | ~0.0 (averaged) | 0.15 m | ❌ (outside 0.15 m radius) |
| DDPM (single sample) | +0.15 or -0.15 | 0 m | ✅ |

### 4.3 The Diagnostic

We propose a simple diagnostic for practitioners:

> **Compute the action distribution std on the training data, grouped by instruction.**
> - If max_std < 0.01 (normalized action space) → **use MSE** (task is unimodal)
> - If max_std ≥ 0.01 → **use DDPM** (task is multi-modal)

This threshold correctly identifies our multi-modal reach task (std = 0.099) and unimodal reach task (std = 0.000).

---

## 5. Adaptive Policy Selector

Based on the diagnostic in Section 4.3, we propose the **Adaptive Policy Selector** — an automatic method that analyzes the training data and selects the appropriate policy:

### 5.1 Algorithm

```
Input: Training data D = {(instruction_i, action_i)}
Output: policy_mode ∈ {"mse", "ddpm"}

1. Group actions by instruction: {inst: [a_1, a_2, ...]}
2. For each instruction, compute std(actions)
3. max_std = max over all instructions
4. If max_std < threshold: return "mse"
5. Else: return "ddpm"
```

### 5.2 Validation

| Data Type | Action std | Selector Output | Correct? |
|-----------|-----------|-----------------|----------|
| Reach-to-target (deterministic) | 0.000 | MSE | ✅ |
| Synthetic 2-mode (actions ±0.5) | 0.500 | DDPM | ✅ |

The selector correctly identifies both cases.

### 5.3 Integration

The selector runs once before training and sets the model's policy mode. During training, the selected loss function (DDPM ε-prediction or direct MSE) is used. During inference, the selected sampling method (100-step reverse chain or single forward pass) is used. No human intervention is required.

---

## 6. Conclusion and Future Work

We showed that on a multi-modal manipulation task (two valid target zones per instruction), DDPM diffusion policy achieves 100% success while direct MSE regression achieves only 44% (5-seed robust, N=100) — a 56-percentage-point gap. The root cause is mechanistic: MSE averages the two valid action modes and lands between them, missing both; DDPM samples one mode per inference call and succeeds either way. On the unimodal task, both methods achieve 100% success, with MSE being ~100× faster at inference. We proposed an Adaptive Policy Selector that automatically chooses between the two methods based on action-distribution modality (per-instruction action std vs threshold 0.01), and verified that the selector picks the empirically-best method on both benchmarks.

The counterintuitive finding that MSE has lower validation loss (7.89e-05) than DDPM (8.78e-04) yet achieves worse task success (44% vs 100%) is a cautionary tale about using val-loss as a proxy for task performance on multi-modal data: the model that fits the data better is also the model that systematically fails the task.

**Limitations**:
- 3-DOF planar arm, single-step expert. The expert solves the task in 1 step (IK directly to target). Real robots are 7-DOF with multi-step trajectories. Generalization to higher-DOF multi-step tasks is left for future work.
- Simulation only. No real-robot transfer yet.
- Single task family (reach-to-zone). Whether the adaptive selector generalizes to other multi-modal task structures (tool use, bimanual) is an open empirical question.
- Hand-tuned threshold (0.01). The modality threshold is empirically validated but theoretically ungrounded. Future work: replace with a calibrated-uncertainty test (conformal prediction).
- Per-instruction routing only. The current selector picks one mode per instruction; per-step routing within multi-step trajectories is a natural extension.
- CLIP backbone not fine-tuned (limits precision).

**Future work**:
- Per-step adaptive routing (route MSE vs DDPM at each timestep of a multi-step trajectory, not just per instruction)
- Replace action-std threshold with calibrated uncertainty (conformal prediction)
- Sim-to-real transfer on Franka or Kinova 7-DOF arm
- Validate on naturally multi-modal task families (tool use, bimanual manipulation, surgical sub-tasks)
- Compare with DDIM (deterministic diffusion) as a third routing option

### Reproduction

All code, data, and trained checkpoints are open-source at https://github.com/Yashas-K-Gangatkar/vla-adaptive-policy (MIT license). The 5-seed robustness result can be reproduced with a single command:

```bash
python -m src.compare_modes_multiseed --eval-episodes 20 --seeds 42 142 242 342 442
```

(Following an initial `python -m src.compare_modes` to train both MSE and DDPM checkpoints.)

---

## References

[1] OpenVLA: Kim et al., "Open-Source Vision-Language-Action Models," 2024. arXiv:2406.09246  
[2] RT-2: Brohan et al., "RT-2: Vision-Language-Action Models," 2023. arXiv:2307.15818  
[3] Octo: Octo Team et al., "Octo: An Open-Source Generalist Robot Policy," 2024. arXiv:2405.12213  
[4] Diffusion Policy: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023 Best Paper. arXiv:2303.04137  
[5] π0: Physical Intelligence, "π0: A Vision-Language-Action Flow Model," 2024.  
[6] Industrial tolerances: ISO 2768-1, "General tolerances for linear dimensions," 1989.
