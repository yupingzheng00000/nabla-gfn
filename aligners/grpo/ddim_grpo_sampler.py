"""
DDIM-based Group Sampler for GRPO Training

This module provides a simplified group sampling function using DDIM with eta=1.0,
adapted from the ddpo-pytorch implementation in stable_diff_patch/.

For GRPO, we need:
- Group-based sampling (multiple samples per prompt for advantage estimation)
- Log-prob tracking for policy gradient
- Support for CFG-free mode (guidance_scale=1.0)
"""

import torch
from typing import Optional, List, Union
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

# Import the mature ddim_step_with_logprob from stable_diff_patch
from .stable_diff_patch.ddim_with_logprob import ddim_step_with_logprob


@torch.no_grad()
def sample_group_ddim(
    pipeline: StableDiffusionPipeline,
    prompts: List[str],
    group_size: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    eta: float = 1.0,
    generator: Optional[torch.Generator] = None,
    height: int = 512,
    width: int = 512,
    timestep_indices_to_save: Optional[torch.Tensor] = None,
):
    """
    Sample multiple images per prompt using DDIM with log-prob tracking.
    
    Args:
        pipeline: StableDiffusionPipeline instance
        prompts: List of text prompts
        group_size: Number of samples per prompt (for advantage estimation)
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale (1.0 = no guidance, recommended for GRPO)
        eta: DDIM stochasticity (1.0 = full stochastic, 0.0 = deterministic)
        generator: Random generator for reproducibility
        height, width: Image dimensions
        timestep_indices_to_save: Optional indices of timesteps to save latents for (memory optimization).
                                   If None, saves all latents. Shape: (num_train_timesteps,)
    
    Returns:
        images: Tensor of shape (batch_size, group_size, 3, H, W)
        log_probs_per_step: Tensor of shape (batch_size, group_size, num_steps)
        log_probs_sum: Tensor of shape (batch_size, group_size) - sum of log_probs
        all_latents: Dict mapping step_idx -> Tensor of shape (batch_size * group_size, 4, H/8, W/8)
                     Only contains latents at timestep_indices_to_save (+ their next latents)
    """
    device = pipeline._execution_device
    batch_size = len(prompts)
    
    # Encode prompts
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    
    prompt_ids = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids.to(device)
    
    prompt_embeds = text_encoder(prompt_ids)[0]  # (batch_size, 77, 768)
    
    # Repeat for group_size
    prompt_embeds = prompt_embeds.repeat_interleave(group_size, dim=0)  # (batch_size * group_size, 77, 768)
    
    # Setup scheduler
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # Prepare latents
    num_channels_latents = pipeline.unet.config.in_channels
    shape = (batch_size * group_size, num_channels_latents, height // 8, width // 8)
    latents = torch.randn(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    # CFG setup (though for GRPO we typically use guidance_scale=1.0)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        # Prepare unconditional embeddings
        uncond_ids = tokenizer(
            [""] * batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids.to(device)
        uncond_embeds = text_encoder(uncond_ids)[0]
        uncond_embeds = uncond_embeds.repeat_interleave(group_size, dim=0)
        
        # Concatenate for CFG
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds])
    else:
        prompt_embeds_cfg = prompt_embeds
    
    # Denoising loop
    # Only save latents at specified indices (memory optimization)
    if timestep_indices_to_save is not None:
        indices_to_save_set = set(timestep_indices_to_save.tolist())
        # Also need the next latent for each saved index
        indices_to_save_set.update((idx + 1 for idx in timestep_indices_to_save.tolist()))
    else:
        indices_to_save_set = None  # Save all
    
    all_latents = {}
    if indices_to_save_set is None or 0 in indices_to_save_set:
        all_latents[0] = latents.cpu()  # Move to CPU to save GPU memory
    
    all_log_probs = []
    
    for i, t in enumerate(timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds_cfg,
            return_dict=False,
        )[0]
        
        # Perform CFG
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # DDIM step with log-prob
        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        latents, log_prob = ddim_step_with_logprob(
            scheduler,
            model_output=noise_pred,
            timestep=timestep_int,
            sample=latents,
            eta=eta,
            use_clipped_model_output=False,
            generator=generator,
            prev_sample=None,
        )
        
        # Save latent only if needed
        step_idx = i + 1  # latents after step i
        if indices_to_save_set is None or step_idx in indices_to_save_set:
            all_latents[step_idx] = latents.cpu()  # Move to CPU to save GPU memory
        
        all_log_probs.append(log_prob)
    
    # Decode latents to images
    latents = latents / pipeline.vae.config.scaling_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    
    # Reshape to (batch_size, group_size, C, H, W)
    images = images.view(batch_size, group_size, 3, height, width)
    
    # Stack log_probs: (num_steps, batch_size * group_size) -> (batch_size, group_size, num_steps)
    log_probs_per_step = torch.stack(all_log_probs, dim=1)  # (batch_size * group_size, num_steps)
    log_probs_per_step = log_probs_per_step.view(batch_size, group_size, num_inference_steps)
    
    # Sum log_probs across timesteps
    log_probs_sum = log_probs_per_step.sum(dim=-1)  # (batch_size, group_size)
    
    return images, log_probs_per_step, log_probs_sum, all_latents


__all__ = ["sample_group_ddim", "ddim_step_with_logprob"]
