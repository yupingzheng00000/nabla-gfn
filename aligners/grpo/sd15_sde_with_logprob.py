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
    """Perform one stochastic update and return log-prob components."""

    model_output = model_output.to(dtype=torch.float32)
    sample = sample.to(dtype=torch.float32)
    prev_sample = None if prev_sample is None else prev_sample.to(dtype=torch.float32)

    timestep_tensor = _ensure_tensor(timestep, model_output.device)
    if timestep_tensor.numel() != sample.shape[0]:
        timestep_tensor = timestep_tensor.expand(sample.shape[0])

    index_list = [scheduler.index_for_timestep(t.item()) for t in timestep_tensor]
    indices = torch.tensor(index_list, device=sample.device, dtype=torch.int64)
    next_indices = torch.clamp(indices + 1, max=len(scheduler.sigmas) - 1)

    sigma_t = _gather_sigmas(scheduler, indices, sample)
    sigma_next = _gather_sigmas(scheduler, next_indices, sample)

    alpha_t_sq = 1.0 / (1.0 + sigma_t**2)
    alpha_next_sq = 1.0 / (1.0 + sigma_next**2)
    sqrt_alpha_t = torch.sqrt(alpha_t_sq)
    sqrt_alpha_next = torch.sqrt(alpha_next_sq)
    sqrt_one_minus_alpha_next = torch.sqrt(torch.clamp(1.0 - alpha_next_sq, min=1e-12))

    # Predict x0 (clean sample) using the epsilon (noise) prediction model.
    x0_pred = (sample - sigma_t * model_output) / sqrt_alpha_t
    mean = sqrt_alpha_next * x0_pred

    if noise_level == 0.0:
        if prev_sample is None:
            prev_sample = mean
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=torch.float32)
        std = torch.zeros_like(sigma_next)
        return prev_sample, log_prob, mean, std

    std = noise_level * sqrt_one_minus_alpha_next

    if prev_sample is None:
        noise = randn_tensor(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        prev_sample = mean + std * noise
    else:
        noise = (prev_sample - mean) / torch.clamp(std, min=1e-12)

    var = torch.clamp(std**2, min=1e-12)
    diff = prev_sample - mean

    diff_sq_over_var = (diff**2 / var).flatten(start_dim=1).sum(dim=1)
    log_det_var = torch.log(var).flatten(start_dim=1).sum(dim=1)

    num_dims = diff[0].numel()
    log_prob = -0.5 * (diff_sq_over_var + log_det_var + num_dims * math.log(2 * math.pi))

    return prev_sample, log_prob, mean, std


__all__ = ["sde_step_with_logprob"]
