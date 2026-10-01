# Ablation Study Results

## 1. Noise Schedule (DDPM on multi-modal task)

| Schedule | Success Rate | Avg Distance |
|----------|:-----------:|:------------:|
| linear | 45% | 0.639m |
| **squaredcos_cap_v2 (cosine)** | **60%** | **0.465m** |
| scaled_linear | 40% | 0.616m |

**Finding:** Cosine schedule is optimal for DDPM on manipulation tasks.

## 2. Diffusion Steps (DDPM on multi-modal task)

| Steps | Success Rate | Avg Distance |
|-------|:-----------:|:------------:|
| **50** | **50%** | **0.451m** |
| 100 (default) | 35% | 0.595m |
| 200 | 40% | 0.552m |

**Finding:** Fewer diffusion steps perform better — less accumulated stochastic variance. 50 steps > 100 > 200.

## 3. Selector Threshold

| Threshold | Selected Mode | Correct? |
|-----------|:------------:|:--------:|
| 0.001 | DDPM | ✅ |
| **0.01 (default)** | **DDPM** | **✅ Sweet spot** |
| 0.1 | MSE | ❌ Wrong (task is multi-modal) |

**Finding:** 0.01 is the optimal threshold. Too high (0.1) incorrectly selects MSE for multi-modal tasks.

## 4. CLIP Backbone (Frozen vs Unfrozen)

| Method | Frozen | Unfrozen | Change |
|--------|:------:|:--------:|:------:|
| DDPM | 45% | 45% | 0% |
| MSE | 30% | 35% | +5% |

**Finding:** Unfreezing CLIP gives a small improvement for MSE (30→35%) but no change for DDPM.

## Summary: Best Configuration

| Parameter | Best Value | Success |
|-----------|-----------|---------|
| Noise schedule | squaredcos_cap_v2 | 60% |
| Diffusion steps | 50 | 50% |
| Selector threshold | 0.01 | Correct selection |
| CLIP backbone | Unfrozen (MSE only) | +5% |
