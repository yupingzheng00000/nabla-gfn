"""Slow-path sampler for Stable Diffusion 1.5 with full-trajectory log-prob tracking.

This is a reference implementation that mirrors the SD3 slow pipeline: every step
uses SDE sampling and records log-probabilities. Unlike the fast "window-based"
sampler, this version provides complete trajectory information at the cost of
higher computation.

Use cases:
- Numerical verification against SD3 slow pipeline
- Full-trajectory debugging and analysis
- Baseline for comparing window-based acceleration

For production GRPO training, use `sd15_pipeline_with_logprob.py` (window-based)
which is significantly faster.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .sd15_sde_with_logprob import sde_step_with_logprob


def _prepare_generator(
    generator: Optional[torch.Generator | Sequence[Optional[torch.Generator]]],
    device: torch.device,
) -> Tuple[Optional[torch.Generator], Optional[List[torch.Generator]]]:
    """Normalise generator inputs for diffusers helpers."""
    
    if generator is None:
        return None, None
    
    if isinstance(generator, torch.Generator):
        return generator, None
    
    # Diffusers allows passing a list of generators (per sample).
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


@torch.no_grad()
def pipeline_with_logprob_slow(
    self,
    prompt: str | List[str],
    *,
    num_inference_steps: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    guidance_scale: float = 1.0,
    negative_prompt: Optional[str | List[str]] = None,
    generator: Optional[torch.Generator | Sequence[Optional[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: str = "pt",
    noise_level: float = 0.7,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Sample with full SDE trajectory and per-step log-prob tracking.
    
    Args:
        prompt: Text prompt(s).
        num_inference_steps: Number of denoising steps.
        height: Image height in pixels.
        width: Image width in pixels.
        guidance_scale: CFG scale. For GRPO training, must be 1.0.
        negative_prompt: Negative prompt(s) for CFG.
        generator: Random generator(s) for reproducibility.
        latents: Optional initial latents.
        output_type: "pt" for torch.Tensor, "pil" for PIL images.
        noise_level: SDE noise injection level (0.0 = deterministic ODE).
    
    Returns:
        Tuple of (images, all_latents, all_log_probs):
        - images: [B, 3, H, W] decoded images in [0, 1]
        - all_latents: List of [B, 4, h, w] latents at each step (length = num_steps + 1)
        - all_log_probs: List of [B] log-probs at each step (length = num_steps)
    """
    
    # Enforce CFG constraint for training
    if guidance_scale != 1.0:
        raise ValueError(
            "Slow pipeline enforces guidance_scale=1.0 for training consistency. "
            "For CFG visualization, use the fast pipeline with noise_level=0.0."
        )
    
    # 1. Setup
    device = self._execution_device
    dtype = self.unet.dtype
    
    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    
    main_generator, generator_list = _prepare_generator(generator, device)
    
    # 2. Encode prompts (no CFG, single embedding)
    tok = self.tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=self.tokenizer.model_max_length,
    ).to(device)
    prompt_embeds = self.text_encoder(tok.input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    
    # 3. Prepare latents
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    num_channels = self.unet.in_channels
    
    if latents is None:
        latents = self.prepare_latents(
            batch_size,
            num_channels,
            height,
            width,
            dtype,
            device,
            generator=generator_list or main_generator,
            latents=None,
        )
    latents = latents.to(dtype=torch.float32)
    
    # 4. Prepare timesteps
    num_inference_steps = num_inference_steps or getattr(self.scheduler.config, "num_train_timesteps", 50)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    
    # 5. Prepare storage
    all_latents = [latents.clone()]
    all_log_probs = []
    
    prompt_embeds_float = prompt_embeds.to(dtype=torch.float32)
    prompt_embeds_model = prompt_embeds.to(dtype)
    
    # 6. Denoising loop (full SDE path, no window optimization)
    self.set_progress_bar_config(
        position=0,
        disable=False,
        leave=False,
        desc="Slow SDE sampling",
        dynamic_ncols=True,
    )
    
    for step_idx, timestep in enumerate(self.progress_bar(timesteps)):
        # UNet forward
        latent_model_input = self.scheduler.scale_model_input(latents.to(dtype), timestep)
        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds_model,
            return_dict=False,
        )[0]
        
        if noise_level == 0.0:
            # Deterministic ODE update via scheduler for exact parity
            try:
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            except Exception:
                latents = self.scheduler.step(noise_pred, timestep, latents)["prev_sample"]
            # For ODE updates, the transition probability is a Dirac delta; log-prob is zero.
            log_prob = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
        else:
            # SDE step with log-prob
            latents, log_prob, _, _ = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                timestep,
                latents,
                noise_level=noise_level,
                generator=main_generator,
            )
        
        all_latents.append(latents.clone())
        all_log_probs.append(log_prob)
    
    # 7. Decode to images
    latents = latents / self.vae.config.scaling_factor
    images = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
    
    # Denormalize from [-1, 1] to [0, 1]
    images = images / 2.0 + 0.5
    
    if output_type == "pt":
        images = images.clamp(0.0, 1.0)
    elif output_type == "pil":
        images = self.image_processor.postprocess(images, output_type="pil")
    
    return images, all_latents, all_log_probs


__all__ = ["pipeline_with_logprob_slow"]
