# GRPO Mode (Experimental)

> This document systematically explains the newly added GRPO training mode in this repository. All parameter names strictly reuse those from the original project; no field renaming or aliasing is introduced.  
> Experimental results are left as placeholders to be filled in later.

## 1. Background and Objectives

The original Nabla-GFN mode uses a GFlowNet structure (forward / reverse / flow residual) to shape distributions via reward functions.  

**GRPO Mode Objectives**: Within the Stable Diffusion + LoRA framework, utilize PPO-style policy gradients + reward models (HPSv2 / PickScore / Aesthetic, etc.) to directly optimize generation quality while maintaining:

- Timestep subsampling to reduce cost (`model.timestep_fraction`)
- Frozen base UNet, training only LoRA adapter layers
- Explicit deviation control: latent KL + UNet output L2 regularization
- Simplified sampling: unified DDIM eta=1 (DDPM equivalent)

## 2. Module Structure Changes

| Module | Nabla-GFN | GRPO |
|--------|-----------|------|
| UNet Weights | Frozen + Flow residual overlay | Frozen + LoRA training |
| Flow Branch | `res_logflowscore_model` | Disabled (`model.no_flow=True`) |
| Reward Gradients | Reward backprop to latents | Standard reward value → advantage |
| KL Regularization | Built into symmetric structure | Recompute latent mean L2 / log-prob MSE |
| Sampling Cache | Save all steps | Save subset only (index-based) |
| Reference Comparison | Separate forward | Same UNet with LoRA disabled (`disable_adapters()`) |

## 3. Core Algorithm Steps (Single Batch Overview)

1. **Sample**: Generate `group_size` branch images using DDIM(eta=1) while retaining latents at specified timesteps.
2. **Compute Rewards**: Map to `(batch_size, group_size)` shape.
3. **Normalize and Scale**: Multiply by `model.reward_scale` to obtain advantages.
4. **Recompute Subset Timesteps**: Calculate log-probs (policy with LoRA enabled).
5. **Disable LoRA**: Obtain reference log-probs (and latent means).
6. **Compute Losses**:
   - PPO clipped loss
   - Latent KL: `((μ_policy - μ_ref)^2 / (2σ_t^2))`
   - UNet reg: `||ε_policy - ε_ref||^2`
7. **Total Loss**: `loss_policy + beta * kl_latent + unet_reg_scale * unet_reg`
8. **Gradient Accumulation & Backward** (supports AMP + GradScaler).
9. **Log Metrics and Visualizations** (optional DreamSim diversity).

## 4. Parameter Details (Original Field Names)

### Optimizer Related (`training.*`)
- `lr`: LoRA parameter learning rate; too large → reward curve spikes then collapses; recommended 1e-4.
- `adam_beta2`: Set to 0.95 for non-stationarity response; 0.999 smoother but slower.
- `max_grad_norm`: Prevent single-step explosion; diagnose via `gradient/norm_pre_clip` vs `post_clip` comparison.

### GRPO Specific (`grpo.*`)
| Field | Description | Tuning Recommendations |
|-------|-------------|------------------------|
| `group_size` | Parallel branches per prompt | 4 (more improves variance, increases memory) |
| `beta` | Latent KL coefficient | Normal 0.02; increase if diverging, decrease if too strong |
| `clip_range` | PPO ratio clipping interval | 0.1–0.2; too small prevents learning |
| `adv_clip_max` | Advantage maximum absolute value | 3–5; disabled=0 |
| `micro_batch_size` | Micro-batch size for log-prob recomputation | 2–4; memory-limited |

### Model Regularization (`model.*`)
| Field | Description | Recommended |
|-------|-------------|-------------|
| `reward_scale` | Reward difference amplification factor | See quick ref table by reward function |
| `timestep_fraction` | Fraction of timesteps participating in gradients | 0.1 (50 steps) |
| `unet_reg_scale` | UNet output difference regularization coefficient | 0 / 20 / 100 / 1e3 |
| `unet_reg_max_frac_of_policy` | UNet reg contribution cap as fraction | 0.1 (enabled control) |
| `no_flow` | Disable flow branch | GRPO requires True |

### Sampling (`sampling.*`)
- `num_steps`: Fixed at 50 (DDIM/DDPM equivalent).
- `guidance_scale`: Keep at 1.0 during training for stability; can increase for evaluation.
- `low_var_subsampling`: Enable trunk chunking to reduce high-variance timestep omission.

### Training Batch Control
| Field | Explanation | Impact on Stability |
|-------|-------------|---------------------|
| `training.batch_size` | Micro-batch size | Too small + large lr easily fluctuates |
| `gradient_accumulation_steps` | Simulate larger batch | Improves stability (cost: time) |
| `sampling.batch_size` | Number of prompts per sampling | Determines reward estimation variance |

## 5. Environment Variable Hooks

| Variable | Purpose | Scenario |
|----------|---------|----------|
| `GRPO_DEBUG=1` | Print policy/reference value difference once | Verify LoRA actually participates |
| `ADAPT_UNET_REG=1` | Dynamically adjust `unet_reg_scale` | When regularization fluctuates |
| `ADAPT_UNET_REG_TARGET=val` | Specify target regularization strength | Fine-tuning phase |
| `ADAPT_UNET_REG_LR=val` | Adaptive adjustment learning rate | Fast convergence or prevent oscillation |

## 6. Monitoring Interpretation

| Metric | Abnormal Pattern | Action Suggestion |
|--------|------------------|-------------------|
| `ratio/mean ≈ 1.000` unchanging | Policy not updating | Increase `clip_range` or `reward_scale` |
| `ratio/clip_fraction > 0.4` | Excessive clipping | Reduce `reward_scale` |
| `loss/kl_latent ≈ 0` and no LoRA gradients | LoRA not updating or KL too weak | Check parameter collection / increase `beta` |
| `loss/unet_reg` monotonically increasing | Regularization too weak / reward too strong | Reduce `reward_scale` or increase `unet_reg_scale` |
| `gradient/max_value > 1e-1` | Gradient explosion warning | Reduce `lr` or increase `max_grad_norm` |
| `advantage/clip_frac > 0.5` | Advantage saturation | Reduce `reward_scale` or increase `adv_clip_max` |

## 7. Recommended Starter Configurations (by reward_fn)

| reward_fn | Starter Configuration |
|-----------|----------------------|
| `hpscore` | lr=1e-4, reward_scale=1200, clip_range=0.1, adv_clip_max=5, beta=0.02, unet_reg_scale=20 |
| `pickscore` | lr=1e-4, reward_scale=5e5, clip_range=0.2, adv_clip_max=5, beta=0.02, unet_reg_scale=0 |
| `aesthetic_score` | lr=3e-4→1e-4 (later), reward_scale=5e3, clip_range=0.15, adv_clip_max=5, beta=0.02, unet_reg_scale=100 |
| `imagereward` | lr=1e-4, reward_scale=2e3, clip_range=0.1, adv_clip_max=5, beta=0.02, unet_reg_scale=20 |

## 8. Running Examples

```bash
# PickScore GRPO
python -m torch.distributed.run --nproc_per_node=2 train_nablagfn.py \
  --config=config/pickscore.py \
  --mode=grpo \
  --exp_name=pickscore_grpo \
  --seed=0

# HPSv2 GRPO (with adaptive UNet regularization)
ADAPT_UNET_REG=1 ADAPT_UNET_REG_TARGET=0.08 python -m torch.distributed.run --nproc_per_node=2 train_nablagfn.py \
  --config=config/hpsv2.py \
  --mode=grpo \
  --exp_name=hpsv2_grpo \
  --seed=0
```

## 9. Experimental Results (Placeholder)

To be added:  
- Ratio/clip_fraction stability regions under different reward_scale values  
- Adaptive vs fixed UNet regularization comparison  
- DreamSim diversity comparisons  
- Integrated convergence curve plots  

## 10. Future Extension Directions (Placeholder)

- Dynamic `reward_scale` and `beta` corridor control  
- EMA LoRA weights for stable evaluation improvement  
- Multi-reward weighting (e.g., HPSv2 + Aesthetic)  

## 11. Quick Checklist

| Check Item | Aligned |
|------------|---------|
| All field names reuse original project | ✅ |
| No new naming introduced | ✅ |
| No bug timeline | ✅ |
| Experimental results placeholder | ✅ |
| Documentation covers algorithm/parameters/monitoring logic | ✅ |

---
