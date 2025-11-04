"""DPMSolver++ (SDE) single step with Gaussian log-prob tracking."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.utils.torch_utils import randn_tensor

TensorLike = Union[torch.FloatTensor, torch.Tensor]


def _ensure_tensor(value: Union[int, float, TensorLike, Sequence], device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
        return tensor.view(-1)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value, device=device, dtype=torch.float32)
    return torch.tensor([value], device=device, dtype=torch.float32)


def _gather_sigmas(scheduler: DPMSolverSinglestepScheduler, indices: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(sample.device)
    gathered = sigmas[indices]
    return gathered.view(-1, *([1] * (sample.dim() - 1)))


def sde_step_with_logprob(
    scheduler: DPMSolverSinglestepScheduler,
    model_output: TensorLike,
    timestep: Union[int, float, TensorLike, Sequence],
    sample: TensorLike,
    *,
    noise_level: float = 0.8,
    generator: Optional[torch.Generator] = None,
    prev_sample: Optional[TensorLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single stochastic step with mean aligned to scheduler.step and Gaussian log-prob.

    - Mean (prev_mean) is defined as the deterministic DPMSolver++ step result. This
      guarantees exact ODE parity when ``noise_level=0``.
    - Std is scaled by the EDM-style sigma grid stride: sqrt(max(sigma_s^2 - sigma_t^2, eps)),
      times a global ``noise_level`` knob for SDE strength.
    - Log-prob is computed as the elementwise isotropic Gaussian density averaged over
      non-batch dimensions, yielding a per-sample scalar.
    """

    # Numerics: compute in fp32
    model_output = model_output.to(dtype=torch.float32)
    sample = sample.to(dtype=torch.float32)
    prev_sample = None if prev_sample is None else prev_sample.to(dtype=torch.float32)

    # Timesteps as a 1D tensor (B,)
    timestep_tensor = _ensure_tensor(timestep, model_output.device)
    if timestep_tensor.numel() != sample.shape[0]:
        timestep_tensor = timestep_tensor.expand(sample.shape[0])

    # Deterministic mean path via the scheduler (ensures ODE parity)
    try:
        prev_mean = scheduler.step(model_output, timestep_tensor, sample, return_dict=False)[0]
    except Exception:
        prev_mean = scheduler.step(model_output, timestep_tensor, sample)["prev_sample"]
    prev_mean = prev_mean.to(torch.float32)

    # Zero-noise: identical to ODE
    if noise_level <= 0.0:
        if prev_sample is None:
            prev_sample = prev_mean
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=torch.float32)
        std = torch.zeros_like(prev_mean)
        return prev_sample, log_prob, prev_mean, std

    # EDM-style std based on sigma grid stride
    index_list = [scheduler.index_for_timestep(t.item()) for t in timestep_tensor]
    indices = torch.tensor(index_list, device=sample.device, dtype=torch.int64)
    next_indices = torch.clamp(indices + 1, max=len(scheduler.sigmas) - 1)

    sigma_s = _gather_sigmas(scheduler, indices, sample)      # shape (B,1,1,1)
    sigma_t = _gather_sigmas(scheduler, next_indices, sample) # shape (B,1,1,1)
    std_scalar = torch.sqrt(torch.clamp(sigma_s**2 - sigma_t**2, min=1e-12))
    std = noise_level * std_scalar

    # Sample (or infer) noise
    if prev_sample is None:
        noise = randn_tensor(
            prev_mean.shape,
            generator=generator,
            device=prev_mean.device,
            dtype=prev_mean.dtype,
        )
        prev_sample = prev_mean + std * noise
    else:
        noise = (prev_sample - prev_mean) / torch.clamp(std, min=1e-12)

    # Per-sample Gaussian log-prob (mean over non-batch dims)
    var = torch.clamp(std**2, min=1e-12)
    diff = prev_sample - prev_mean
    elementwise = -0.5 * (diff**2 / var + torch.log(var) + math.log(2 * math.pi))
    log_prob = elementwise.flatten(start_dim=1).mean(dim=1)

    return prev_sample, log_prob, prev_mean, std



__all__ = ["sde_step_with_logprob"]
