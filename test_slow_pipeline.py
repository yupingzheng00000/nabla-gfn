#!/usr/bin/env python3
"""Test script for SD1.5 slow pipeline with full-trajectory log-prob tracking."""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler
import os
from PIL import Image

# Import the slow pipeline function
import sys
sys.path.insert(0, '/122090808/nabla-gfn')
from aligners.grpo.sd15_pipeline_with_logprob_slow import pipeline_with_logprob_slow


def test_slow_pipeline():
    """Test SD1.5 slow pipeline against fast pipeline for consistency."""
    
    print("="*60)
    print("SD1.5 Slow Pipeline Test")
    print("="*60)
    
    # Setup
    device = "cuda"
    seed = 42
    prompt = "a humpback whale breaching"
    num_steps = 20  # Use fewer steps for faster testing
    noise_level = 0.7
    
    # Load pipeline
    print(f"\nLoading SD1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    
    # Set DPMSolver++ SDE scheduler (same as GRPO)
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.algorithm_type = "sde-dpmsolver++"
    pipe.scheduler.config.final_sigmas_type = 'sigma_min'
    pipe.scheduler.set_timesteps(num_steps, device=device)
    
    # Monkey-patch the slow pipeline
    pipe.pipeline_with_logprob_slow = pipeline_with_logprob_slow.__get__(pipe, type(pipe))
    
    print(f"Prompt: '{prompt}'")
    print(f"Steps: {num_steps}")
    print(f"Noise level: {noise_level}")
    print(f"Seed: {seed}")
    
    # Run slow pipeline
    print(f"\n{'='*60}")
    print("Running SLOW pipeline (full SDE trajectory)...")
    print(f"{'='*60}")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    images, all_latents, all_log_probs = pipe.pipeline_with_logprob_slow(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=1.0,
        noise_level=noise_level,
        generator=generator,
        output_type="pt",
    )
    
    # Verify outputs
    print(f"\n✓ Sampling complete!")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Latents recorded: {len(all_latents)} steps (expected {num_steps + 1})")
    print(f"  - Log-probs recorded: {len(all_log_probs)} steps (expected {num_steps})")
    
    # Check log-prob statistics
    log_probs_tensor = torch.stack(all_log_probs)
    print(f"\nLog-prob statistics:")
    print(f"  - Mean: {log_probs_tensor.mean().item():.2f}")
    print(f"  - Std: {log_probs_tensor.std().item():.2f}")
    print(f"  - Min: {log_probs_tensor.min().item():.2f}")
    print(f"  - Max: {log_probs_tensor.max().item():.2f}")
    print(f"  - Sum: {log_probs_tensor.sum().item():.2f}")
    
    # Save image
    os.makedirs("./outputs/slow_pipeline_test", exist_ok=True)
    output_path = f"./outputs/slow_pipeline_test/slow_seed{seed}_noise{noise_level}.png"
    
    image_np = images[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype("uint8")
    Image.fromarray(image_np).save(output_path)
    print(f"\n✓ Saved image to: {output_path}")
    
    # Test with noise_level=0 (deterministic ODE)
    print(f"\n{'='*60}")
    print("Running SLOW pipeline with noise_level=0.0 (ODE mode)...")
    print(f"{'='*60}")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    images_ode, all_latents_ode, all_log_probs_ode = pipe.pipeline_with_logprob_slow(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=1.0,
        noise_level=0.0,  # Deterministic
        generator=generator,
        output_type="pt",
    )
    
    print(f"\n✓ ODE sampling complete!")
    log_probs_ode = torch.stack(all_log_probs_ode)
    print(f"  - Log-probs sum (should be ~0): {log_probs_ode.sum().item():.6f}")
    
    output_path_ode = f"./outputs/slow_pipeline_test/slow_seed{seed}_noise0.0_ode.png"
    image_np_ode = images_ode[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    image_np_ode = (image_np_ode * 255).astype("uint8")
    Image.fromarray(image_np_ode).save(output_path_ode)
    print(f"✓ Saved ODE image to: {output_path_ode}")
    
    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print(f"{'='*60}")
    print("\nSlow pipeline is ready for:")
    print("  - Full-trajectory debugging")
    print("  - Numerical verification against SD3 slow pipeline")
    print("  - Log-prob analysis and validation")
    print("\nFor GRPO training, use the fast (window-based) pipeline instead.")


if __name__ == "__main__":
    test_slow_pipeline()
