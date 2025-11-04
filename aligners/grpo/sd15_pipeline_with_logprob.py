"""Group sampler for Stable Diffusion 1.5 with per-step log-prob tracking.

This module provides a `sample_group_with_sde_window` helper that can be
monkey-patched onto a `diffusers.StableDiffusionPipeline` instance. It mirrors
the Flow-GRPO "Fast" recipe: run ODE steps until a randomly selected window,
switch to SDE (DPMSolver++ SDE variant) inside that window, and record the
Gaussian transition log-probabilities for RL updates.

The implementation assumes deterministic CFG-free sampling (guidance=1.0)
and returns only the data required for GRPO training:

* imgs[B, K, 3, H, W]           — decoded images in [0, 1]
* traj_logprobs[B, K, W]        — per-step log-probabilities inside window
* timesteps_window[List[int]]   — scheduler timesteps used for the window
* logp_sum_old[B, K]            — sum of per-step log-probs (for Δlogp clipping)

Where B is the prompt batch size and K is the group size (branches per prompt).
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from .sd15_sde_with_logprob import sde_step_with_logprob


def _prepare_generator(
    generator: Optional[torch.Generator | Sequence[Optional[torch.Generator]]],
    device: torch.device,
) -> Tuple[Optional[torch.Generator], Optional[List[torch.Generator]]]:
    """Normalise generator inputs for diffusers helpers.

    Returns a tuple `(main_generator, generator_list)` where the former is a
    single generator to use for sampling latents/window indices, and the latter
    is a list suitable for diffusers' `prepare_latents` API (or ``None`` if not
    required).
    """

    if generator is None:
        return None, None

    if isinstance(generator, torch.Generator):
        return generator, None

    # Diffusers allows passing a list of generators (per sample). Preserve it
    # but also expose the first usable generator for miscellaneous sampling.
    generator_list: List[torch.Generator] = []
    first: Optional[torch.Generator] = None
    for item in generator:
        if item is None:
            item = torch.Generator(device=device)
            item.manual_seed(torch.seed())
        generator_list.append(item)
        if first is None:
            first = item
    return first, generator_list


def _compute_window_indices(
    total_steps: int,
    window_size: int,
    window_range: Tuple[float, float],
    generator: Optional[torch.Generator],
) -> List[int]:
    """Map fractional window range to discrete indices and sample start index."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if total_steps < 2:
        raise ValueError("scheduler must provide at least two timesteps")

    frac_lo, frac_hi = window_range
    if not (0.0 <= frac_lo < frac_hi <= 1.0):
        raise ValueError("window_range must satisfy 0 <= lo < hi <= 1")

    # Highest index we allow is total_steps - 2 to avoid the final step (which
    # can explode in SDE mode due to tiny variance).
    max_valid_index = total_steps - 2
    lo_index = max(int(math.floor(frac_lo * total_steps)), 0)
    hi_index = min(int(math.floor(frac_hi * total_steps)) - 1, max_valid_index)
    if hi_index < lo_index:
        hi_index = lo_index

    window_span = hi_index - lo_index + 1
    if window_span < window_size:
        window_size = window_span
        lo_index = max(hi_index - window_size + 1, 0)

    start_min = lo_index
    start_max = hi_index - window_size + 1
    if start_max < start_min:
        start_min = start_max

    if generator is not None:
        start_offset = torch.randint(start_min, start_max + 1, (1,), generator=generator).item()
    else:
        start_offset = random.randint(start_min, start_max)

    return list(range(start_offset, start_offset + window_size))


@torch.no_grad()
def sample_group_with_sde_window(
    self,
    prompts: Sequence[str],
    group_size: int,
    window_size: int,
    window_range: Tuple[float, float],
    *,
    num_inference_steps: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator | Sequence[Optional[torch.Generator]]] = None,
    same_latent: bool = True,
    noise_level: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    """Sample groups with an SDE window and return per-step log-probabilities."""

    if guidance_scale != 1.0:
        raise ValueError("Flow-GRPO training requires guidance_scale=1.0 (no CFG)")

    # Align pipeline state with CFG-free sampling expectations.
    self._guidance_scale = 1.0
    self.do_classifier_free_guidance = False

    batch_size = len(prompts)
    if batch_size == 0:
        raise ValueError("prompts must be non-empty")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    device = self._execution_device
    dtype = self.unet.dtype

    main_generator, generator_list = _prepare_generator(generator, device)

    # Encode prompts once (CFG disabled, so only positive prompts).
    prompt_embeds = self._encode_prompt(
        prompt=list(prompts),
        device=device,
        num_images_per_prompt=group_size,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    # Prepare latent noise.
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    num_channels = self.unet.in_channels

    latent_batch = batch_size * group_size if not same_latent else batch_size
    latents = self.prepare_latents(
        latent_batch,
        num_channels,
        height,
        width,
        dtype,
        device,
        generator=generator_list or main_generator,
        latents=None,
    )
    if same_latent:
        latents = latents.unsqueeze(1).repeat(1, group_size, 1, 1, 1)
        latents = latents.view(batch_size * group_size, num_channels, latents.size(-2), latents.size(-1))

    # Ensure scheduler uses the requested number of steps.
    num_inference_steps = num_inference_steps or getattr(self.scheduler.config, "num_train_timesteps", 50)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    window_indices = _compute_window_indices(len(timesteps), window_size, window_range, main_generator)
    window_index_set = set(window_indices)
    timesteps_window = [int(timesteps[i].item()) for i in window_indices]

    latents = latents.to(dtype=torch.float32)
    prompt_embeds = prompt_embeds.to(dtype=torch.float32)
    prompt_embeds_model = prompt_embeds.to(dtype)

    traj_logprobs: List[torch.Tensor] = []
    window_inputs: List[torch.Tensor] = []
    window_outputs: List[torch.Tensor] = []

    for step_idx, timestep in enumerate(timesteps):
        if step_idx in window_index_set:
            window_inputs.append(latents.clone())

        latent_model_input = self.scheduler.scale_model_input(latents.to(dtype), timestep)
        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds_model,
            return_dict=False,
        )[0]

        step_noise = noise_level if step_idx in window_index_set else 0.0
        latents, log_prob, _, _ = sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            timestep,
            latents,
            noise_level=step_noise,
            generator=main_generator,
        )

        if step_idx in window_index_set:
            traj_logprobs.append(log_prob)
            window_outputs.append(latents.clone())

    # Decode latents to images in [0, 1].
    latents = latents / self.vae.config.scaling_factor
    latents = latents.to(dtype=self.vae.dtype)
    images = self.decode_latents(latents)
    images = images.clamp(0.0, 1.0)

    num_branches = batch_size * group_size
    if traj_logprobs:
        traj_logprob_tensor = torch.stack(traj_logprobs, dim=1)
    else:
        traj_logprob_tensor = torch.zeros(num_branches, 0, device=latents.device)

    imgs = images.view(batch_size, group_size, *images.shape[1:])
    traj_logprob_tensor = traj_logprob_tensor.view(batch_size, group_size, -1)
    logp_sum_old = traj_logprob_tensor.sum(dim=-1)

    if window_inputs:
        window_inputs_tensor = torch.stack(window_inputs, dim=0)
        window_outputs_tensor = torch.stack(window_outputs, dim=0)
        window_steps = window_inputs_tensor.size(0)
        latent_h, latent_w = latents.shape[-2:]
        window_inputs_tensor = window_inputs_tensor.view(window_steps, batch_size, group_size, num_channels, latent_h, latent_w)
        window_outputs_tensor = window_outputs_tensor.view(window_steps, batch_size, group_size, num_channels, latent_h, latent_w)
    else:
        latent_h, latent_w = latents.shape[-2:]
        window_inputs_tensor = torch.empty(0, batch_size, group_size, num_channels, latent_h, latent_w, device=latents.device)
        window_outputs_tensor = window_inputs_tensor.clone()

    window_state = {
        "inputs": window_inputs_tensor,
        "outputs": window_outputs_tensor,
    }

    return imgs, traj_logprob_tensor, timesteps_window, logp_sum_old, window_state


__all__ = ["sample_group_with_sde_window"]
