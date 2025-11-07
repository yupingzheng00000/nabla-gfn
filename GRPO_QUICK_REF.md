# GRPO Quick Reference

> This document provides a quick overview of GRPO (Generalized Reward Policy Optimization) mode configuration. All field names strictly reuse the original project's `config.*` and training script variables without renaming. For detailed explanations, see `docs/GRPO_EXPERIMENTAL.md`.

## 1. Mode Comparison

| Dimension | Nabla-GFN (`mode=nablagfn`) | GRPO (`mode=grpo`) |
|-----------|----------------------------|---------------------|
| Core Algorithm | GFlowNet (forward + reverse + score matching) | PPO-style policy gradient + KL + UNet regularization |
| UNet Training | Coupled via flow/residual with reward gradients | Only LoRA adapter layers (`.pf.`) trained |
| Sampling Scheduler | Optional DDIM / SDE DPM solver | Fixed DDIM eta=1.0 (DDPM equivalent) |
| Timestep Participation | `model.timestep_fraction` applies to flow loss | Same, determines subset for log-prob recomputation |
| KL Regularization | Inherent in flow structure | Explicit latent-space KL (`kl_latent`) + log-prob MSE diagnostics |
| Reward Usage | Backward reward gradients + forward construction | Standard advantage normalization then scaled |
| Regularization | `unet_reg_scale` (optional) | `beta * kl_latent + unet_reg_scale * unet_reg` |
| Adapter Control | `model.no_flow` disables flow branch | GRPO enforces `model.no_flow = True` |

## 2. Key Hyperparameters (Original Names)

| Category | Field | Meaning (GRPO) | Recommended Range (by reward) |
|----------|-------|----------------|-------------------------------|
| Optimizer | `training.lr` | LoRA parameter learning rate | 1e-3 (HPSv2/PickScore), 1e-4 (Aesthetic, may reduce) |
| Optimizer | `training.adam_beta1` | AdamW β1 | 0.9 |
| Optimizer | `training.adam_beta2` | AdamW β2 | 0.95 (RL recommended) or 0.999 (conservative) |
| Optimizer | `training.adam_weight_decay` | Weight decay | 1e-4 |
| Optimizer | `training.max_grad_norm` | Gradient clipping threshold | 1.0 |
| Precision | `training.mixed_precision` | Compute precision | bf16 recommended |
| LoRA | `model.lora_rank` | Low-rank adapter dimension | 8 |
| Scheduler | `sampling.num_steps` | Diffusion steps | 50 (DDIM-DDPM) |
| CFG | `sampling.guidance_scale` | CFG during training | 1.0 (stable), eval can use 7.5–9 |
| Subsampling | `model.timestep_fraction` | Fraction of timesteps participating in gradients | 0.1 (50 steps), 0.4 (20 steps) |
| PPO | `grpo.group_size` | Number of branches per prompt | 4 |
| PPO | `grpo.beta` | KL coefficient (latent KL) | 0.02 (adjustable 0.004–0.04) |
| PPO | `grpo.clip_range` | Ratio clipping range | 0.1–0.2 (default 0.2, widen if stuck, narrow if diverging) |
| PPO | `grpo.adv_clip_max` | Advantage clipping upper bound | 3–5 (0=disabled) |
| PPO | `grpo.micro_batch_size` | Sub-batch size for UNet recomputation | 2–4 (memory-limited) |
| Regularization | `model.unet_reg_scale` | UNet output L2 regularization coefficient | 0 (disabled) / 20 / 100 / 1e3 (task-dependent) |
| Reward | `model.reward_scale` | Advantage amplification factor | Aesthetic:1e3–1e4 HPSv2:1e3–1e4 PickScore:5e4–5e5 ImageReward:1e3–5e3 |
| Sampling | `sampling.num_batches_per_epoch` | Number of sampling batches per epoch | 4 (default) |
| Batch | `sampling.batch_size` | Number of prompts during sampling | 16 |
| Training | `training.batch_size` | Micro-batch during training | 2 (HPSv2) / 4 (PickScore/Aesthetic) |
| Accumulation | `training.gradient_accumulation_steps` | Gradient accumulation steps | 4–16 (increase when UNet is large) |
| Saving | `logging.save_freq` | Checkpoint save frequency (epochs) | 5 |
| Evaluation | DreamSim | Diversity metric | Every 5 epochs, sample 16 images |

## 3. Recommended `model.reward_scale` by Reward Function

| `experiment.reward_fn` | Raw Output Range | Recommended `model.reward_scale` | Notes |
|------------------------|------------------|----------------------------------|-------|
| `aesthetic_score` | ~5–7 | 5–10 | Don't over-scale, gradients stable |
| `hpscore` (HPSv2) | ~0.15–0.30 | 1e3 | Too large (>=3e3) causes oscillation |
| `pickscore` | ~20–25 | 6e3 | Keep normalization clean |

## 4. Environment Variable Switches

| Environment Variable | Effect | Default |
|---------------------|--------|---------|
| `GRPO_DEBUG=1` | Print policy vs reference difference once (first batch) | 0 |
| `ADAPT_UNET_REG=1` | Adaptively scale `unet_reg_scale` | 0 |
| `ADAPT_UNET_REG_TARGET` | Specify regularization target (default 0.1 * |loss_policy|) | unset |
| `ADAPT_UNET_REG_LR` | Adaptive adjustment step size (default 0.05) | unset |

## 5. Logging Metrics (WandB Paths)

| Name | Meaning |
|------|---------|
| `loss/policy` | PPO main loss (with clipping) |
| `loss/kl_latent` | Latent-space KL (not log-prob) |
| `loss/unet_reg` | Noise prediction L2 difference |
| `loss/total` | Combined loss (with coefficients) |
| `reward/mean` / `reward/std` | Raw reward distribution |
| `advantage/*` | Scaled advantage statistics |
| `ratio/mean` / `ratio/clip_fraction` | PPO ratio deviation and clipping fraction |
| `kl/logprob_mse` | Diagnostic: MSE of log-prob difference |
| `gradient/norm_*` | Gradient magnitude (pre-clip/post-clip/LoRA-only/max) |
| `diversity/dreamsim` | DreamSim embedding variance (×100 scaled) |

## 6. Effective Batch Size Calculation

The effective batch size per gradient update is calculated as:

```
Effective Batch Size = sampling.batch_size × grpo.group_size × gradient_accumulation_steps × num_gpus
```

**Example Configuration (Current Experiments on A800 GPUs)**:
- `sampling.batch_size = 4` (prompts per sampling step)
- `grpo.group_size = 4` (branches per prompt)
- `gradient_accumulation_steps = 4`
- `num_gpus = 2` (A800)

**Calculation**:
- Samples per forward pass: 4 × 4 = 16
- Per gradient update: 16 × 4 × 2 = **128 samples**
- Per epoch (with `num_batches_per_epoch=4`): 128 × 4 = **512 samples**

**Hardware**: Experiments conducted on 2× NVIDIA A800 GPUs (80GB).

## 7. Experimental Results (Placeholder)

To be added: Convergence curves / diversity comparisons / ratio stability regions for different reward functions.

---
