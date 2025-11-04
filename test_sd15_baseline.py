#!/usr/bin/env python3
"""
Ultimate sanity check: Pure SD1.5 generation with different CFG values.
This bypasses all GRPO logic to isolate CFG behavior.
"""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler
from PIL import Image
import os

def generate_with_cfg(prompt, guidance_scale, num_steps=50, seed=0, output_dir="./outputs/sd15_baseline"):
    """Generate image with specified CFG and save it."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SD1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    # Use DPMSolver++ SDE (same as GRPO)
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.algorithm_type = "sde-dpmsolver++"
    pipe.scheduler.config.final_sigmas_type = 'sigma_min'
    
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    
    # Generate
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Generating: '{prompt}'")
    print(f"CFG: {guidance_scale}, Steps: {num_steps}, Seed: {seed}")
    print(f"{'='*60}")
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    # Save
    output_path = os.path.join(output_dir, f"cfg{guidance_scale}_steps{num_steps}_seed{seed}.png")
    image.save(output_path)
    print(f"✓ Saved to: {output_path}")
    
    return image


if __name__ == "__main__":
    # Test prompts (same as GRPO uses)
    prompts = [
        "a humpback whale breaching",
        "a photo of a cat",
        "a beautiful landscape",
    ]
    
    cfg_values = [1.0, 5.0, 7.5]
    num_steps = 20
    seed = 0
    
    print("\n" + "="*60)
    print("SD1.5 BASELINE TEST")
    print("="*60)
    print(f"Model: runwayml/stable-diffusion-v1-5")
    print(f"Scheduler: DPMSolverSinglestep (SDE mode)")
    print(f"CFG values: {cfg_values}")
    print(f"Steps: {num_steps}")
    print(f"Seed: {seed}")
    print("="*60)
    
    for prompt in prompts:
        for cfg in cfg_values:
            generate_with_cfg(prompt, cfg, num_steps, seed)
    
    print("\n" + "="*60)
    print("✓ All images generated!")
    print("Check ./outputs/sd15_baseline/")
    print("="*60)
