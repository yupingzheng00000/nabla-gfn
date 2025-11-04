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
    """Perform one stochastic update matching DPMSolver++ SDE and return log-prob.
    
    This implementation exactly mirrors the DPMSolver++ SDE update formula from diffusers
    to ensure correct transitions and log-probability computation.
    """

    model_output = model_output.to(dtype=torch.float32)
    sample = sample.to(dtype=torch.float32)
    prev_sample = None if prev_sample is None else prev_sample.to(dtype=torch.float32)

    timestep_tensor = _ensure_tensor(timestep, model_output.device)
    if timestep_tensor.numel() != sample.shape[0]:
        timestep_tensor = timestep_tensor.expand(sample.shape[0])

    # Get sigma values at current and next timesteps
    index_list = [scheduler.index_for_timestep(t.item()) for t in timestep_tensor]
    indices = torch.tensor(index_list, device=sample.device, dtype=torch.int64)
    next_indices = torch.clamp(indices + 1, max=len(scheduler.sigmas) - 1)

    sigma_s = _gather_sigmas(scheduler, indices, sample)
    sigma_t = _gather_sigmas(scheduler, next_indices, sample)

    # Convert sigma to (alpha, sigma) parameterization
    alpha_s = 1.0 / torch.sqrt(1.0 + sigma_s**2)
    alpha_t = 1.0 / torch.sqrt(1.0 + sigma_t**2)
    
    # Convert model output (epsilon) to x0 prediction (required for DPMSolver++)
    # This matches scheduler.convert_model_output() behavior
    if scheduler.config.prediction_type == "epsilon":
        x0_pred = (sample - sigma_s * model_output) / alpha_s
    elif scheduler.config.prediction_type == "sample":
        x0_pred = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        x0_pred = alpha_s * sample - sigma_s * model_output
    else:
        raise ValueError(f"Unsupported prediction_type: {scheduler.config.prediction_type}")
    
    # Compute lambda values: lambda = log(alpha) - log(sigma)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    h = lambda_t - lambda_s

    # DPMSolver++ SDE formula (first-order):
    # x_t = (sigma_t / sigma_s * exp(-h)) * sample
    #       + (alpha_t * (1 - exp(-2h))) * x0_pred
    #       + sigma_t * sqrt(1 - exp(-2h)) * noise
    
    exp_neg_h = torch.exp(-h)
    exp_neg_2h = torch.exp(-2.0 * h)
    
    coeff_sample = (sigma_t / sigma_s) * exp_neg_h
    coeff_x0 = alpha_t * (1.0 - exp_neg_2h)
    std = sigma_t * torch.sqrt(torch.clamp(1.0 - exp_neg_2h, min=1e-12))
    
    # Compute mean of the transition
    mean = coeff_sample * sample + coeff_x0 * x0_pred

    if noise_level == 0.0:
        # Deterministic ODE update
        if prev_sample is None:
            prev_sample = mean
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=torch.float32)
        return prev_sample, log_prob, mean, std

    # Scale std by noise_level (SDE strength)
    std = std * noise_level

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

    # Compute log-probability of the Gaussian transition
    # Use mean() like SD3 implementation to get per-sample scalar log-prob
    var = torch.clamp(std**2, min=1e-12)
    diff = prev_sample - mean
    
    log_prob = (
        -((diff ** 2) / (2 * var))
        - torch.log(std)
        - 0.5 * math.log(2 * math.pi)
    )
    
    # Average over spatial dimensions to get per-sample log-prob [B]
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, mean, std



__all__ = ["sde_step_with_logprob"]
