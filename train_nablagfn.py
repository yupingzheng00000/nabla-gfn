import os
from collections import defaultdict
import contextlib
import datetime
import time
import wandb
from functools import partial
import tempfile
from PIL import Image
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import logging
import copy
import pickle, gzip

import math

import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverSinglestepScheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available

from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.distributed import init_distributed_singlenode, set_seed, setup_for_distributed

import lib.reward_func.prompts
import lib.reward_func.rewards
from lib.diffusion.sample_trajectory import sample_trajectory
from lib.diffusion.inference_step import inference_step, predict_clean, get_alpha_prod_t
from aligners.grpo.sd15_pipeline_with_logprob_fast import sample_group_with_sde_window
from aligners.grpo.sd15_sde_with_logprob import sde_step_with_logprob
from aligners.grpo.sd15_pipeline_with_logprob_slow import pipeline_with_logprob_slow
from aligners.grpo.ddim_grpo_sampler import sample_group_ddim, ddim_step_with_logprob

# DreamSim for diversity calculation
try:
    from dreamsim import dreamsim
    DREAMSIM_AVAILABLE = True
except ImportError:
    DREAMSIM_AVAILABLE = False
    print("Warning: dreamsim not available. Diversity metrics will be disabled.")


from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from torch.nn.attention import SDPBackend, sdpa_kernel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_integer("seed", 0, "Seed.")
flags.DEFINE_enum("mode", "nablagfn", ["nablagfn", "grpo"], "Training mode (nablagfn or grpo).")
flags.DEFINE_bool("sde_sanity", False, "Run a zero-noise SDE window sanity check and exit.")
flags.DEFINE_float("sde_sanity_cfg", None, "Override guidance_scale for sanity check. If None, uses training value (1.0 for GRPO).")

def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main(args):
    train()

def setup(local_rank, is_local_main_process):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    config = FLAGS.config

    config.gpu_type = torch.cuda.get_device_name() \
                            if torch.cuda.is_available() else "CPU"
    if is_local_main_process:
        logger.info(f"GPU type: {config.gpu_type}")

    # config.config_name = f"{FLAGS.config}"
    if FLAGS.seed is not None:
        config.seed = FLAGS.seed
    else:
        config.seed = 0

    if FLAGS.mode == "grpo":
        config.model.no_flow = True
    if config.model.no_flow:
        config.model.reverse_loss_scale = 0.0

    wandb_name = f"{config.experiment.reward_fn.split('_')[0]}_{FLAGS.exp_name}_seed{config.seed}"


    if config.logging.use_wandb:
        # Try to get wandb key from config, env var, or skip login (let wandb handle it)
        wandb_key = config.logging.wandb_key
        if wandb_key and wandb_key != 'PLACEHOLDER':
            wandb.login(key=wandb_key)
        elif 'WANDB_API_KEY' in os.environ:
            wandb.login(key=os.environ['WANDB_API_KEY'])
        # else: wandb will use ~/.netrc or prompt for login
        
        wandb.init(project=config.logging.proj_name, name=wandb_name, config=config.to_dict(),
           dir=config.logging.wandb_dir,
           save_code=True, mode="online" if is_local_main_process else "disabled")

    os.makedirs(config.saving.output_dir, exist_ok=True)

    if is_local_main_process:
        logger.info(f"\n{config}")
    set_seed(config.seed)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision, torch_dtype=weight_dtype,
    )
    scheduler_config = {}

    scheduler_config.update(pipeline.scheduler.config)
    num_inference_steps = config.sampling.num_steps
    if FLAGS.mode == "grpo":
        if is_local_main_process:
            logger.info("GRPO mode: using DDIMScheduler with eta=1.0 (DDPM-equivalent)")
        pipeline.scheduler = DDIMScheduler.from_config(scheduler_config)
        # Note: config.sampling.eta is already 1 in default_config.py (int type)
        # DDIM with eta=1 is equivalent to DDPM (full stochastic)
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    elif config.sampling.scheduler == 'DPM-solver':
        if is_local_main_process:
            logger.info("Using SDE DPM-solver (1st order)")
        pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(scheduler_config)
        pipeline.scheduler.config.algorithm_type = "sde-dpmsolver++"  # Switch to SDE mode
        pipeline.scheduler.config.final_sigmas_type = 'sigma_min'
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    else:
        pipeline.scheduler = DDIMScheduler.from_config(scheduler_config)
    pipeline.sample_group_with_sde_window = sample_group_with_sde_window.__get__(pipeline, type(pipeline))
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=weight_dtype)
    pipeline.text_encoder.to(device, dtype=weight_dtype)
    pipeline.scheduler.set_timesteps(config.sampling.num_steps, device=device)  # set_timesteps(): 1000 steps -> 50 steps

    pipeline.safety_checker = None
    # Save memory during VAE encode/decode
    try:
        pipeline.enable_vae_slicing()
    except Exception:
        pass
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    unet = pipeline.unet
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        param.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    unet_lora_config = LoraConfig(
        r=config.model.lora_rank, lora_alpha=config.model.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config, adapter_name="pf") ## LoRA

    # Enable gradient checkpointing to reduce activation memory
    if getattr(config.training, "gradient_checkpointing", True):
        try:
            unet.enable_gradient_checkpointing()
            if is_local_main_process:
                logger.info("Enabled UNet gradient checkpointing")
        except Exception as e:
            if is_local_main_process:
                logger.warning(f"Failed to enable gradient checkpointing: {e}")

    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if is_local_main_process:
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        unet.enable_xformers_memory_efficient_attention()
        if is_local_main_process:
            logger.info("xformers is enabled for memory efficient attention")
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    if not config.model.no_flow:
        res_logflowscore_model = UNet2DConditionModel(
            in_channels=4, block_out_channels=config.model.flow_channel_width,
            layers_per_block=config.model.flow_layers_per_block, cross_attention_dim=pipeline.text_encoder.config.hidden_size
        )

        ### Zero initilaization
        try:
            res_logflowscore_model.conv_out.bias.data *= 0.0
        except:
            pass
    else:
        res_logflowscore_model = None

    unet.set_adapter("pf")
    if config.training.mixed_precision in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)
        if not config.model.no_flow:
            cast_training_params(res_logflowscore_model, dtype=torch.float32)

    pf_params = [param for name, param in unet.named_parameters() if '.pf.' in name]
    if not config.model.no_flow:
        flow_params = filter(lambda p: p.requires_grad, res_logflowscore_model.parameters())

    if config.training.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.amp.GradScaler(
            "cuda",
            growth_interval=config.training.gradscaler_growth_interval
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 is True by default
        torch.backends.cudnn.benchmark = True

    optimizer_cls = torch.optim.AdamW

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length, # 77
        ).input_ids.to(device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sampling.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.training.batch_size, 1, 1)

    unet.to(device)
    unet = DDP(unet, device_ids=[local_rank])
    pipeline.unet = unet.module
    if not config.model.no_flow:
        res_logflowscore_model.to(device)
        res_logflowscore_model = DDP(res_logflowscore_model, device_ids=[local_rank])

    #######################################################
    #################### FOR GFN ##########################
    if not config.model.no_flow:
        params = [
            {"params": pf_params, "lr": config.training.lr},
            {"params": flow_params, "lr": config.training.flow_lr, 'weight_decay': config.training.flow_wd},
        ]
    else:
        params = [
            {"params": pf_params, "lr": config.training.lr},
        ]

    optimizer = optimizer_cls(
        params,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    # Load DreamSim model for diversity calculation (only on main process)
    dreamsim_model = None
    dreamsim_preprocess = None
    if DREAMSIM_AVAILABLE and is_local_main_process:
        try:
            dreamsim_model, dreamsim_preprocess = dreamsim(
                pretrained=True, 
                device=device,
                cache_dir="./models"
            )
            dreamsim_model.eval()
            for param in dreamsim_model.parameters():
                param.requires_grad = False
            logger.info("DreamSim model loaded for diversity calculation")
        except Exception as e:
            logger.warning(f"Failed to load DreamSim: {e}. Diversity metrics disabled.")
            dreamsim_model = None

    return config, pipeline, optimizer, unet, res_logflowscore_model, sample_neg_prompt_embeds, train_neg_prompt_embeds, logger, scaler, dreamsim_model, dreamsim_preprocess


def train_grpo(
    local_rank,
    global_rank,
    world_size,
    config,
    pipeline,
    optimizer,
    unet,
    logger,
    scaler,
    device,
    is_local_main_process,
    dreamsim_model=None,
    dreamsim_preprocess=None,
):
    pipeline.unet.eval()
    unet.train()

    prompt_fn = getattr(lib.reward_func.prompts, config.experiment.prompt_fn)
    reward_ctor = getattr(lib.reward_func.rewards, config.experiment.reward_fn)
    reward_fn = reward_ctor(torch.float32, device)

    group_size = config.grpo.group_size
    clip_range = float(config.grpo.clip_range)
    adv_clip_max = float(getattr(config.grpo, "adv_clip_max", 0.0) or 0.0)
    beta = float(config.grpo.beta)
    grad_accum = max(1, int(config.training.gradient_accumulation_steps))
    batch_size = int(config.sampling.batch_size)
    num_batches = int(config.sampling.num_batches_per_epoch)
    num_epochs = int(config.training.num_epochs)
    eta = float(config.sampling.eta)  # For DDIM, eta=1.0 gives full stochasticity
    num_inference_steps = int(config.sampling.num_steps)
    
    # Timestep subsampling: use only a fraction of timesteps for gradient computation
    # README: 0.1 for 50-step DDPM, 0.4 for 20-step SDE-DPM-Solver
    timestep_fraction = float(config.model.timestep_fraction)
    num_train_timesteps = max(1, int(num_inference_steps * timestep_fraction))
    
    # Reward scale: amplify reward signal (teacher's baseline values)
    # aesthetic: 1e4, hps: 3e6, pickscore: 5e5
    reward_scale = float(config.model.reward_scale)
    
    if is_local_main_process:
        logger.info("="*70)
        logger.info("[GRPO] Configuration:")
        logger.info(f"  Sampler: DDIM (eta={eta})")
        logger.info(f"  Inference steps: {num_inference_steps}")
        logger.info(f"  Timestep fraction: {timestep_fraction} ({num_train_timesteps}/{num_inference_steps} steps for gradient)")
        logger.info(f"  Reward function: {config.experiment.reward_fn}")
        logger.info(f"  Reward scale: {reward_scale:.2e}")
        logger.info(f"  KL beta: {beta}")
        logger.info(f"  PPO clip range: {clip_range}")
        if adv_clip_max > 0:
            logger.info(f"  Advantage clip max: {adv_clip_max}")
        else:
            logger.info("  Advantage clip max: disabled")
        logger.info(f"  Group size: {group_size}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {grad_accum}")
        logger.info(f"  Grad checkpointing: {getattr(config.training, 'gradient_checkpointing', True)}")
        logger.info(f"  Micro-batch size: {getattr(config.grpo, 'micro_batch_size', 4)}")
        logger.info("="*70)

    unet_module = unet.module if hasattr(unet, "module") else unet
    unet_dtype = next(unet_module.parameters()).dtype

    amp_dtype = None
    if config.training.mixed_precision == "fp16":
        amp_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    autocast_ctx = torch.cuda.amp.autocast if amp_dtype is not None else contextlib.nullcontext
    autocast_kwargs = {"dtype": amp_dtype} if amp_dtype is not None else {}

    # Frozen reference pipeline for KL regularisation.
    # IMPORTANT: Share the same UNet with policy pipeline (which has LoRA)
    # We'll disable LoRA adapters during reference computation
    ref_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=amp_dtype or torch.float32,
    ).to(device)
    # Replace ref_pipeline's UNet with the policy UNet (which has LoRA loaded)
    ref_pipeline.unet = unet
    ref_pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    ref_pipeline.scheduler.set_timesteps(config.sampling.num_steps, device=device)
    ref_pipeline.text_encoder.requires_grad_(False)
    ref_pipeline.vae.requires_grad_(False)
    # Get the underlying module (unwrap DDP if needed) for disable_adapters to work
    ref_unet = ref_pipeline.unet.module if hasattr(ref_pipeline.unet, 'module') else ref_pipeline.unet
    ref_unet_dtype = next(ref_unet.parameters()).dtype

    generator = torch.Generator(device=device)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(num_epochs):
        if is_local_main_process:
            logger.info(f"[GRPO] Epoch {epoch}")

        for batch_idx in range(num_batches):
            seed = config.seed + epoch * num_batches + batch_idx + global_rank
            generator.manual_seed(seed)

            prompt_batch = [prompt_fn(**config.experiment.prompt_fn_kwargs) for _ in range(batch_size)]
            prompts, prompt_metadata = zip(*prompt_batch)
            prompts = list(prompts)
            prompt_metadata = list(prompt_metadata)

            # Pre-compute timestep indices for gradient computation (memory optimization)
            timesteps = pipeline.scheduler.timesteps
            if num_train_timesteps < len(timesteps):
                if config.sampling.low_var_subsampling:
                    # Low-variance trunk-based subsampling
                    # Divide timesteps into n_trunks, sample one step from each trunk
                    # Prioritize later trunks (more important for denoising)
                    n_trunks = num_train_timesteps
                    assert len(timesteps) % n_trunks == 0, f"num_steps ({len(timesteps)}) must be divisible by timestep_fraction*num_steps ({n_trunks})"
                    
                    trunk_size = len(timesteps) // n_trunks
                    step_indices = torch.arange(len(timesteps), device=device)
                    trunks = step_indices.view(n_trunks, trunk_size)  # (n_trunks, trunk_size)
                    
                    # Sample one random step from each trunk
                    # Reverse order to prioritize later steps (stronger signal)
                    timestep_indices = []
                    for i in reversed(range(n_trunks)):
                        trunk = trunks[i]
                        # Random index within trunk (changes each batch for variance reduction)
                        idx = torch.randint(0, trunk_size, (1,), device=device, generator=generator)
                        timestep_indices.append(trunk[idx].item())
                    timestep_indices = torch.tensor(sorted(timestep_indices), dtype=torch.long, device=device)
                else:
                    # Uniform subsampling (deterministic, evenly spaced)
                    timestep_indices = torch.linspace(0, len(timesteps) - 1, num_train_timesteps, dtype=torch.long)
            else:
                timestep_indices = torch.arange(len(timesteps), dtype=torch.long)

            unet.eval()
            with torch.inference_mode():
                # Sample using DDIM with eta=1.0 (full stochastic, equivalent to DDPM)
                # Only save latents at timestep_indices to reduce memory usage
                imgs, log_probs_per_step, log_probs_sum_old, all_latents = sample_group_ddim(
                    pipeline,
                    prompts,
                    group_size,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=config.sampling.guidance_scale,  # Training guidance (default 1.0)
                    eta=eta,
                    generator=generator,
                    height=512,
                    width=512,
                    timestep_indices_to_save=timestep_indices,
                )
            unet.train()

            num_branches = batch_size * group_size
            imgs = imgs.to(device=device, dtype=torch.float32)
            log_probs_per_step = log_probs_per_step.to(device=device, dtype=torch.float32)
            log_probs_sum_old = log_probs_sum_old.to(device=device, dtype=torch.float32)

            # IMPORTANT: Align old log-prob sum to the same subset of timesteps used for recomputation
            # Otherwise, comparing sum over all steps (old) vs sum over subset (new) will explode the ratio.
            step_idx_dev = timestep_indices.to(log_probs_per_step.device)
            log_probs_sum_old_subset = log_probs_per_step.index_select(dim=-1, index=step_idx_dev).sum(dim=-1)

            # Compute rewards.
            images_flat = imgs.view(num_branches, *imgs.shape[2:])
            prompt_repeat = [prompt for prompt in prompts for _ in range(group_size)]
            metadata_repeat = [meta for meta in prompt_metadata for _ in range(group_size)]
            rewards_flat, _ = reward_fn(images_flat, prompt_repeat, metadata_repeat)
            rewards = rewards_flat.to(device=device, dtype=torch.float32).view(batch_size, group_size)
            reward_mean = rewards.mean().item()
            reward_std = rewards.std(unbiased=False).item()

            # Compute advantages (group-wise normalization) and apply reward_scale
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            advantages = advantages / (rewards.std(dim=1, keepdim=True) + 1e-6)
            advantages = advantages * reward_scale  # Amplify the training signal
            # Store pre-clipping advantage for saturation analysis
            advantages_pre_clip = advantages.clone() if adv_clip_max > 0 else None
            if adv_clip_max > 0:
                advantages = advantages.clamp_(-adv_clip_max, adv_clip_max)

            # Encode prompts manually (CFG-free) to match sampler behavior.
            tok = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).to(device)
            prompt_embeds = pipeline.text_encoder(tok.input_ids)[0]
            prompt_embeds = prompt_embeds.repeat_interleave(group_size, dim=0)
            prompt_embeds_model_full = prompt_embeds.to(unet_dtype)

            guidance_scale = float(config.sampling.guidance_scale)
            do_classifier_free_guidance = guidance_scale > 1.0 + 1e-6
            if do_classifier_free_guidance:
                tok_uncond = pipeline.tokenizer(
                    [""] * len(prompts),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).to(device)
                uncond_embeds = pipeline.text_encoder(tok_uncond.input_ids)[0]
                uncond_embeds = uncond_embeds.repeat_interleave(group_size, dim=0)
                uncond_embeds_model_full = uncond_embeds.to(unet_dtype)
            else:
                uncond_embeds = None
                uncond_embeds_model_full = None

            # Recompute log-probs using timestep subsampling (for efficiency)
            # timestep_indices already computed before sampling
            # OPTIMIZATION: Micro-batch over the flattened batch dimension to reduce peak memory
            total = batch_size * group_size
            micro_bs_cfg = int(getattr(config.grpo, "micro_batch_size", 4) or 4)
            micro_bs = max(1, min(total, micro_bs_cfg))  # tuneable via config.grpo.micro_batch_size
            logp_sum_new_flat = torch.zeros(total, device=device, dtype=torch.float32)
            
            # For latent-space KL (flow_grpo style): collect UNet output differences
            kl_latent_loss = 0.0
            num_kl_steps = 0

            with (autocast_ctx(**autocast_kwargs) if autocast_kwargs else autocast_ctx()):
                for start in range(0, total, micro_bs):
                    end = min(start + micro_bs, total)
                    acc_slice = torch.zeros(end - start, device=device, dtype=torch.float32)

                    embeds_slice = prompt_embeds_model_full[start:end]
                    if do_classifier_free_guidance:
                        uncond_embeds_slice = uncond_embeds_model_full[start:end]
                    else:
                        uncond_embeds_slice = None

                    for step_idx in timestep_indices:
                        step_idx = int(step_idx.item()) if isinstance(step_idx, torch.Tensor) else int(step_idx)
                        t = timesteps[step_idx]

                        # Load only the current micro-batch latents on-demand (keep float32 for log-prob accuracy)
                        latents_in = all_latents[step_idx][start:end].to(device=device, dtype=torch.float32)  # (micro_bs, 4, H/8, W/8)
                        latents_out = all_latents[step_idx + 1][start:end].to(device=device, dtype=torch.float32)

                        # Predict noise with current policy for the micro-batch (heavy math can stay in mixed precision)
                        latents_in_unet = latents_in.to(unet_dtype)
                        if do_classifier_free_guidance:
                            latent_model_input = torch.cat([latents_in_unet, latents_in_unet], dim=0)
                        else:
                            latent_model_input = latents_in_unet
                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                        if do_classifier_free_guidance:
                            encoder_hidden_states = torch.cat([uncond_embeds_slice, embeds_slice], dim=0)
                        else:
                            encoder_hidden_states = embeds_slice

                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred_policy = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            noise_pred_policy = noise_pred
                        noise_pred_policy = noise_pred_policy.to(torch.float32)

                        # Compute latent-space KL (flow_grpo style): compare next-step latent means with timestep normalization
                        with torch.no_grad():
                            # Get reference model prediction (disable LoRA)
                            unet_module = unet.module if hasattr(unet, 'module') else unet
                            unet_module.disable_adapters()
                            noise_pred_ref_full = unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=encoder_hidden_states,
                                return_dict=False,
                            )[0]
                            unet_module.enable_adapters()
                        
                        if do_classifier_free_guidance:
                            noise_pred_ref_uncond, noise_pred_ref_text = noise_pred_ref_full.chunk(2)
                            noise_pred_ref = noise_pred_ref_uncond + guidance_scale * (noise_pred_ref_text - noise_pred_ref_uncond)
                        else:
                            noise_pred_ref = noise_pred_ref_full
                        noise_pred_ref = noise_pred_ref.to(torch.float32)
                        
                        # Compute DDIM prev_sample_mean for both policy and reference (formula 12 from DDIM paper)
                        # prev_sample_mean = sqrt(α_{t-1}) * x0 + sqrt(1 - α_{t-1} - σ²_t) * ε
                        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                        prev_timestep_int = max(0, timestep_int - pipeline.scheduler.config.num_train_timesteps // pipeline.scheduler.num_inference_steps)
                        
                        alpha_prod_t = pipeline.scheduler.alphas_cumprod[timestep_int].to(device)
                        alpha_prod_t_prev = pipeline.scheduler.alphas_cumprod[prev_timestep_int].to(device) if prev_timestep_int >= 0 else pipeline.scheduler.final_alpha_cumprod.to(device)
                        beta_prod_t = 1 - alpha_prod_t
                        
                        # Predict x0 from noise following scheduler prediction_type
                        pred_type = getattr(pipeline.scheduler.config, "prediction_type", "epsilon")
                        if pred_type == "epsilon":
                            # Standard case for SD1.5
                            x0_policy = (latents_in - torch.sqrt(beta_prod_t) * noise_pred_policy) / torch.sqrt(alpha_prod_t)
                            x0_ref = (latents_in - torch.sqrt(beta_prod_t) * noise_pred_ref) / torch.sqrt(alpha_prod_t)
                        elif pred_type == "sample":
                            # Model directly predicts x0
                            x0_policy = noise_pred_policy
                            x0_ref = noise_pred_ref
                        elif pred_type == "v_prediction":
                            # pred_original_sample = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
                            x0_policy = torch.sqrt(alpha_prod_t) * latents_in - torch.sqrt(beta_prod_t) * noise_pred_policy
                            x0_ref = torch.sqrt(alpha_prod_t) * latents_in - torch.sqrt(beta_prod_t) * noise_pred_ref
                        else:
                            raise ValueError(f"Unsupported scheduler prediction_type: {pred_type}")
                        
                        # Compute DDIM variance: σ_t² = η² * β_{t-1} / β_t * (1 - α_t / α_{t-1})
                        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                        std_dev_t = eta * torch.sqrt(variance)
                        
                        # Compute prev_sample_mean (next-step latent mean)
                        # prev_sample_mean = sqrt(α_{t-1}) * x0 + sqrt(1 - α_{t-1} - σ²_t) * ε
                        pred_sample_direction_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
                        prev_sample_mean_policy = torch.sqrt(alpha_prod_t_prev) * x0_policy + pred_sample_direction_coeff * noise_pred_policy
                        prev_sample_mean_ref = torch.sqrt(alpha_prod_t_prev) * x0_ref + pred_sample_direction_coeff * noise_pred_ref
                        
                        # KL loss: L2 distance between next-step means, normalized by DDIM noise std
                        kl_step = ((prev_sample_mean_policy - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3)) / (2 * std_dev_t ** 2 + 1e-8)
                        kl_latent_loss += kl_step.sum().item()
                        num_kl_steps += (end - start)
                        
                        del noise_pred_ref, noise_pred_ref_full, x0_policy, x0_ref, prev_sample_mean_policy, prev_sample_mean_ref

                        # Compute log-prob using DDIM formula (micro-batch)
                        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                        _, log_prob_new = ddim_step_with_logprob(
                            pipeline.scheduler,
                            model_output=noise_pred_policy,
                            timestep=timestep_int,
                            sample=latents_in,
                            eta=eta,
                            prev_sample=latents_out,
                        )
                        acc_slice = acc_slice + log_prob_new.to(torch.float32)

                        # Free GPU memory for this micro-batch slice immediately
                        del latents_in, latents_out, latents_in_unet, latent_model_input, noise_pred_policy, noise_pred
                        torch.cuda.empty_cache()

                    logp_sum_new_flat[start:end] = acc_slice

            logp_sum_new = logp_sum_new_flat.view(batch_size, group_size)

            # Compute reference log-probs for KL regularization (use same subsampled timesteps)
            with torch.inference_mode():
                # CRITICAL: Disable LoRA for reference policy to get base model behavior
                # Need to access .module for DDP-wrapped models
                unet_module = unet.module if hasattr(unet, 'module') else unet
                unet_module.disable_adapters()
                
                # KL reference log-probs (micro-batched over batch dimension)
                total = batch_size * group_size
                micro_bs = max(1, min(total, micro_bs_cfg))
                logp_ref_flat = torch.zeros(total, device=device, dtype=torch.float32)
                # Reference embeddings computed with reference text encoder for stability
                tok_ref = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).to(device)
                prompt_embeds_ref_full = ref_pipeline.text_encoder(tok_ref.input_ids)[0]
                prompt_embeds_ref_full = prompt_embeds_ref_full.repeat_interleave(group_size, dim=0).to(ref_unet_dtype)
                if do_classifier_free_guidance and uncond_embeds is not None:
                    uncond_embeds_ref_full = uncond_embeds.to(ref_unet_dtype)
                else:
                    uncond_embeds_ref_full = None

                for start in range(0, total, micro_bs):
                    end = min(start + micro_bs, total)
                    acc_slice = torch.zeros(end - start, device=device, dtype=torch.float32)
                    embeds_ref_slice = prompt_embeds_ref_full[start:end]

                    for step_idx in timestep_indices:
                        step_idx = int(step_idx.item()) if isinstance(step_idx, torch.Tensor) else int(step_idx)
                        t = timesteps[step_idx]
                        # Load on-demand for micro-batch (keep float32 for accurate KL)
                        latents_in = all_latents[step_idx][start:end].to(device=device, dtype=torch.float32)
                        latents_out = all_latents[step_idx + 1][start:end].to(device=device, dtype=torch.float32)

                        latents_in_ref = latents_in.to(ref_unet_dtype)
                        if do_classifier_free_guidance and uncond_embeds_ref_full is not None:
                            latent_model_input = torch.cat([latents_in_ref, latents_in_ref], dim=0)
                            encoder_hidden_states_ref = torch.cat([
                                uncond_embeds_ref_full[start:end],
                                embeds_ref_slice,
                            ], dim=0)
                        else:
                            latent_model_input = latents_in_ref
                            encoder_hidden_states_ref = embeds_ref_slice

                        latent_model_input = ref_pipeline.scheduler.scale_model_input(latent_model_input, t)
                        noise_pred_ref_full = ref_unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=encoder_hidden_states_ref,
                            return_dict=False,
                        )[0]
                        if do_classifier_free_guidance:
                            noise_pred_ref_uncond, noise_pred_ref_text = noise_pred_ref_full.chunk(2)
                            noise_pred_ref = noise_pred_ref_uncond + guidance_scale * (noise_pred_ref_text - noise_pred_ref_uncond)
                        else:
                            noise_pred_ref = noise_pred_ref_full
                        noise_pred_ref = noise_pred_ref.to(torch.float32)

                        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                        _, log_prob_ref = ddim_step_with_logprob(
                            ref_pipeline.scheduler,
                            model_output=noise_pred_ref,
                            timestep=timestep_int,
                            sample=latents_in,
                            eta=eta,
                            prev_sample=latents_out,
                        )
                        acc_slice = acc_slice + log_prob_ref.to(torch.float32)

                        # Free GPU memory for this micro-batch slice immediately
                        del latents_in, latents_out, latents_in_ref, latent_model_input, noise_pred_ref, noise_pred_ref_full
                        torch.cuda.empty_cache()

                    logp_ref_flat[start:end] = acc_slice

                logp_ref = logp_ref_flat.view(batch_size, group_size)
                
                # Re-enable LoRA for policy updates
                unet_module.enable_adapters()

            # Use subset-aligned old log-prob sum for a fair comparison
            delta_logp = logp_sum_new - log_probs_sum_old_subset
            # Standard PPO clipped-ratio objective
            ratio = torch.exp(delta_logp)
            unclipped_loss = -advantages * ratio
            if clip_range > 0.0:
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                clipped_loss = -advantages * clipped_ratio
                loss_policy = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            else:
                loss_policy = torch.mean(unclipped_loss)
            
            # Compute latent-space KL divergence (flow_grpo style)
            # Normalized by timestep noise variance for stability across different diffusion steps
            kl_latent_mean = kl_latent_loss / max(num_kl_steps, 1)
            loss_kl = kl_latent_mean
            
            # Legacy unet_reg support (deprecated, use latent KL instead)
            if config.model.unet_reg_scale > 0:
                logger.warning("[GRPO] unet_reg_scale is deprecated, latent-space KL is now used automatically")
            
            loss = loss_policy + beta * loss_kl
            
            # DEBUG: Print logp values to diagnose KL=0
            if is_local_main_process and batch_idx == 0:
                logger.info(f"[DEBUG] logp_sum_new: mean={logp_sum_new.mean().item():.4f} std={logp_sum_new.std().item():.4f}")
                logger.info(f"[DEBUG] logp_ref: mean={logp_ref.mean().item():.4f} std={logp_ref.std().item():.4f}")
                logger.info(f"[DEBUG] diff: mean={(logp_sum_new - logp_ref).mean().item():.6f} std={(logp_sum_new - logp_ref).std().item():.6f}")
                logger.info(f"[DEBUG] latent_kl: {kl_latent_mean:.6f}")
            
            loss = loss / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # Compute monitoring metrics
            ratio = torch.exp(delta_logp.detach()).mean().item()
            # Log-prob based KL (for diagnostics only; main regularizer uses latent-space KL)
            kl_logprob = ((logp_sum_new.detach() - logp_ref) ** 2).mean().item()
            clip_fraction = ((torch.abs(torch.exp(delta_logp.detach()) - 1.0) > clip_range).float().mean().item() if clip_range > 0 else 0.0)
            advantage_mean = advantages.mean().item()
            advantage_std = advantages.std().item()
            # Compute advantage clipping saturation ratio: fraction of samples with |s*Z| > C
            advantage_clip_frac = 0.0
            if adv_clip_max > 0 and advantages_pre_clip is not None:
                advantage_clip_frac = (torch.abs(advantages_pre_clip) > adv_clip_max).float().mean().item()
            
            # Warning if ratio deviates significantly from 1.0 (on-policy target)
            if is_local_main_process and abs(ratio - 1.0) > 0.02:
                logger.warning(f"[GRPO] ratio deviates from on-policy: {ratio:.3f}")
            
            # Warning if KL divergence is too high
            if is_local_main_process and loss_kl > 0.3:
                logger.warning(f"[GRPO] High latent KL: {loss_kl:.4f} (consider increasing beta or reducing reward_scale)")

            if is_local_main_process:
                log_msg = (
                    f"[GRPO] epoch {epoch} batch {batch_idx}/{num_batches} | "
                    f"loss={loss_policy.item():.4f} kl_latent={loss_kl:.4f} kl_logprob={kl_logprob:.4f}"
                )
                log_msg += (
                    f" | reward={reward_mean:.3f}±{reward_std:.3f} adv={advantage_mean:.2e}±{advantage_std:.2e} | "
                    f"ratio={ratio:.3f} clip_frac={clip_fraction:.3f} adv_clip_frac={advantage_clip_frac:.3f}"
                )
                logger.info(log_msg)
                
                if config.logging.use_wandb:
                    # Use unique step identifier (epoch*num_batches+batch_idx) to avoid overwriting
                    # This ensures every batch gets logged with its own step
                    wandb_step = epoch * num_batches + batch_idx
                    log_dict = {
                        "loss/policy": loss_policy.item(),
                        "loss/kl_latent": loss_kl,
                        "loss/total": loss.item() * grad_accum,
                        "reward/mean": reward_mean,
                        "reward/std": reward_std,
                        "advantage/mean": advantage_mean,
                        "advantage/std": advantage_std,
                        "advantage/clip_frac": advantage_clip_frac,
                        "ratio/mean": ratio,
                        "ratio/clip_fraction": clip_fraction,
                        "kl/logprob_mse": kl_logprob,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                    }
                    if adv_clip_max > 0:
                        log_dict["advantage/clip_max"] = adv_clip_max
                    wandb.log(log_dict, step=wandb_step)
            
            # Save images (optional, last batch of each epoch for visualization)
            if batch_idx == num_batches - 1:  # Last batch of epoch
                if is_local_main_process and (epoch % 5 == 0 or epoch == 0):  # Every 5 epochs + epoch 0
                    images_dir = os.path.join(config.saving.output_dir, f"images_epoch{epoch}")
                    os.makedirs(images_dir, exist_ok=True)
                    # imgs shape: (batch_size, group_size, 3, H, W), already in [0,1] range
                    imgs_flat = imgs.view(-1, *imgs.shape[2:])  # (batch*group, 3, H, W)
                    for i in range(min(16, imgs_flat.shape[0])):
                        image_np = imgs_flat[i].cpu().float().numpy()  # (3, H, W)
                        image_np = (image_np.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
                        pil = Image.fromarray(image_np)
                        pil = pil.resize((256, 256))
                        pil.save(os.path.join(images_dir, f"{i}.jpg"))
                
                # Upload to wandb (last batch of epoch)
                if config.logging.use_wandb and is_local_main_process and (epoch % 5 == 0 or epoch == 0):
                    import tempfile
                    # imgs shape: (batch_size, group_size, 3, H, W)
                    imgs_flat = imgs.view(-1, *imgs.shape[2:])  # (batch*group, 3, H, W)
                    num_imgs_to_log = min(16, imgs_flat.shape[0])
                    # Use unique step for image logging (epoch-based to avoid conflicts)
                    image_step = epoch * num_batches + batch_idx
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for i in range(num_imgs_to_log):
                            image_np = imgs_flat[i].cpu().float().numpy()  # (3, H, W)
                            image_np = (image_np.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
                            pil = Image.fromarray(image_np)
                            pil = pil.resize((256, 256))
                            pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                        
                        # Build caption with prompt and reward
                        wandb_images = []
                        for i in range(num_imgs_to_log):
                            prompt_idx = i // group_size
                            prompt_text = prompts[prompt_idx] if prompt_idx < len(prompts) else "N/A"
                            reward_val = rewards_flat[i].item() if i < len(rewards_flat) else 0.0
                            caption = f"{prompt_text[:50]}... | R={reward_val:.2f}"
                            wandb_images.append(
                                wandb.Image(
                                    os.path.join(tmpdir, f"{i}.jpg"),
                                    caption=caption
                                )
                            )
                        
                        wandb.log(
                            {"sample_images": wandb_images},
                            step=image_step,
                        )
            
            # Clear latents dict to free CPU memory for next batch
            all_latents.clear()
            torch.cuda.empty_cache()
        
        # Calculate diversity using DreamSim every 5 epochs
        if dreamsim_model is not None and epoch % 5 == 0:
            if is_local_main_process:
                try:
                    with torch.inference_mode():
                        # Sample a batch of images for diversity calculation
                        sample_seed = config.seed + epoch * 999
                        sample_generator = torch.Generator(device=device).manual_seed(sample_seed)
                        sample_prompts = [prompt_fn(**config.experiment.prompt_fn_kwargs) for _ in range(min(16, batch_size))]
                        sample_prompts_text = [p[0] for p in sample_prompts]
                        
                        # Generate images using pipeline
                        output = pipeline(
                            sample_prompts_text,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=config.sampling.guidance_scale,
                            generator=sample_generator,
                            output_type="pil",
                            return_dict=True,
                        )
                        sample_imgs_pil = output.images
                        
                        # Extract embeddings using DreamSim
                        embeddings_list = []
                        for img_pil in sample_imgs_pil:
                            # Preprocess and extract embedding
                            img_tensor = dreamsim_preprocess(img_pil).unsqueeze(0).to(device)
                            embed = dreamsim_model.embed(img_tensor)  # (1, embed_dim) or (1, n_patches, embed_dim)
                            # Flatten to (embed_dim,)
                            if len(embed.shape) > 2:
                                embed = embed.mean(dim=1)  # Average over patches if patch model
                            embeddings_list.append(embed.squeeze(0))
                        
                        # Stack embeddings: (n_samples, embed_dim)
                        embeddings = torch.stack(embeddings_list)
                        
                        # Compute variance across samples (diversity metric)
                        # Higher variance = higher diversity
                        diversity_dreamsim = embeddings.var(dim=0).mean().item() * 100  # Scale to match paper units (x10^-2)
                        
                        logger.info(f"[Diversity] Epoch {epoch}: DreamSim diversity = {diversity_dreamsim:.2f} (x10^-2)")
                        
                        if config.logging.use_wandb:
                            wandb.log({
                                "diversity/dreamsim": diversity_dreamsim,
                                "epoch": epoch,
                            }, step=global_step)
                        
                        del sample_imgs_pil, embeddings_list, embeddings
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Failed to compute diversity: {e}")
        
        # Save checkpoint every save_freq epochs (default: 5)
        if epoch % config.logging.save_freq == 0 or epoch == num_epochs - 1:
            if is_local_main_process:
                save_path = os.path.join(config.saving.output_dir, f"checkpoint_epoch{epoch}")
                os.makedirs(save_path, exist_ok=True)
                
                # Save LoRA weights (GRPO uses "pf" adapter, not "default")
                unwrapped_unet = unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
                )
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )
                
                # Save optimizer state (for resuming training)
                optimizer_path = os.path.join(save_path, "optimizer.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, optimizer_path)

                # Save conv_out weights to match non-GRPO repo artifacts
                try:
                    conv_out_weights = unwrapped_unet.conv_out.state_dict()
                    torch.save(
                        conv_out_weights,
                        os.path.join(config.saving.output_dir, f"conv_out_weights_epoch{epoch}.pt"),
                    )
                except Exception as e:
                    logger.warning(f"[GRPO] Failed to save conv_out weights: {e}")

                # Create underscore alias for checkpoint directory for compatibility
                try:
                    alias_path = os.path.join(config.saving.output_dir, f"checkpoint_epoch_{epoch}")
                    if not os.path.exists(alias_path):
                        rel_target = os.path.relpath(save_path, os.path.dirname(alias_path))
                        os.symlink(rel_target, alias_path)
                except Exception as e:
                    logger.warning(f"[GRPO] Failed to create alias 'checkpoint_epoch_{epoch}': {e}")
                
                logger.info(f"[GRPO] Saved checkpoint to {save_path}")
            
            dist.barrier()

def run_sde_sanity(config, pipeline, device, logger):
    prompt_fn = getattr(lib.reward_func.prompts, config.experiment.prompt_fn)
    batch_size = int(config.sampling.batch_size)
    group_size = int(config.grpo.group_size) if hasattr(config, "grpo") else 1

    prompt_batch = [prompt_fn(**config.experiment.prompt_fn_kwargs) for _ in range(batch_size)]
    prompts, prompt_metadata = zip(*prompt_batch)
    prompts = list(prompts)
    prompt_metadata = list(prompt_metadata)

    # Temporarily disable LoRA adapters for a clean baseline image.
    lora_restore = None
    try:
        if hasattr(pipeline.unet, "disable_adapters"):
            pipeline.unet.disable_adapters()
            lora_restore = True
    except Exception:
        pass

    generator = torch.Generator(device=device).manual_seed(config.seed)
    
    # Allow overriding guidance_scale for sanity check visualization
    guidance_scale = FLAGS.sde_sanity_cfg if FLAGS.sde_sanity_cfg is not None else 1.0
    if FLAGS.sde_sanity_cfg is not None:
        logger.info(f"Sanity check: Using CFG guidance_scale={guidance_scale} (override)")
    
    # Use sample_group_with_sde_window with noise_level=0.0 (sanity mode)
    # This now supports CFG when guidance_scale != 1.0
    imgs, _, timesteps_window, _, _ = pipeline.pipeline_with_logprob_slow(
        prompts,
        group_size,
        config.sample.sde_window_size,
        tuple(config.sample.sde_window_range),
        num_inference_steps=config.sampling.num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        same_latent=config.sample.same_latent,
        noise_level=0.0,  # Sanity mode enables CFG support
    )
    
    # Restore LoRA after sampling
    try:
        if lora_restore and hasattr(pipeline.unet, "enable_adapters"):
            pipeline.unet.enable_adapters()
    except Exception:
        pass
    
    first_image = imgs[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    first_image = (first_image * 255).astype("uint8")
    rank = dist.get_rank() if dist.is_initialized() else 0
    os.makedirs(config.saving.output_dir, exist_ok=True)
    cfg_suffix = f"_cfg{guidance_scale}" if FLAGS.sde_sanity_cfg is not None else ""
    output_path = os.path.join(config.saving.output_dir, f"sde_sanity_rank{rank}{cfg_suffix}.png")
    Image.fromarray(first_image).save(output_path)
    logger.info(f"Saved SDE sanity image to {output_path}; guidance_scale={guidance_scale}, window timesteps {timesteps_window}")


def train():

    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    config, pipeline, optimizer, unet, res_logflowscore_model, sample_neg_prompt_embeds, train_neg_prompt_embeds, logger, scaler, dreamsim_model, dreamsim_preprocess = setup(local_rank, is_local_main_process)

    device = torch.device(local_rank)

    if FLAGS.sde_sanity:
        run_sde_sanity(config, pipeline, device, logger)
        return

    if FLAGS.mode == "grpo":
        train_grpo(
            local_rank,
            global_rank,
            world_size,
            config,
            pipeline,
            optimizer,
            unet,
            logger,
            scaler,
            device,
            is_local_main_process,
            dreamsim_model,
            dreamsim_preprocess,
        )
        return

    def decode(latents, clamp=True):
        image = pipeline.vae.decode(
            latents / pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        image = image / 2.0 + 0.5
        if clamp:
            image = image.clamp(0, 1)
        return image

    # prepare prompt and reward fn
    prompt_fn = getattr(lib.reward_func.prompts, config.experiment.prompt_fn)
    reward_fn = getattr(lib.reward_func.rewards, config.experiment.reward_fn)(torch.float32, device)

    def flow_cast_float32():
        return torch.cuda.amp.autocast(dtype=torch.float32)

    autocast = contextlib.nullcontext # LoRA weights are actually float32, but other part of SD are in bf16/fp16
    if config.model.reverse_loss_scale == 0:
        ref_compute_mode = torch.inference_mode
    else:
        ref_compute_mode = contextlib.nullcontext

    result = defaultdict(dict)
    result["config"] = config.to_dict()
    start_time = time.time()

    #######################################################
    # Start!
    samples_per_epoch = (
        config.sampling.batch_size * num_processes
        * config.sampling.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.training.batch_size * num_processes
        * config.training.gradient_accumulation_steps
    )

    if is_local_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {config.training.num_epochs}")
        logger.info(f"  Sample batch size per device = {config.sampling.batch_size}")
        logger.info(f"  Train batch size per device = {config.training.batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
        )
        logger.info("")
        logger.info(f"  Total number of samples per epoch = test_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {config.training.num_inner_epochs}")

    assert config.sampling.batch_size >= config.training.batch_size
    assert config.sampling.batch_size % config.training.batch_size == 0 # not necessary
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = -1 ## epoch -1 only to collect data; training starts from epoch 0
    global_step = 0
    curr_samples = None

    num_inference_steps = config.sampling.num_steps
    scheduler_dt = pipeline.scheduler.timesteps[0] - pipeline.scheduler.timesteps[1]
    num_train_timesteps = int(num_inference_steps * config.model.timestep_fraction)
    if num_train_timesteps != num_inference_steps:
        num_train_timesteps += 1
    accumulation_steps = config.training.gradient_accumulation_steps * num_train_timesteps

    for epoch in range(first_epoch, config.training.num_epochs):

        #################### SAMPLING ####################
        torch.cuda.empty_cache()
        unet.zero_grad()
        unet.eval()
        if not config.model.no_flow:
            res_logflowscore_model.zero_grad()

        samples = []
        prompts = []
        with torch.inference_mode():
            for i in tqdm(
                range(config.sampling.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not is_local_main_process,
                position=0,
            ):
                # generate prompts
                prompts, prompt_metadata = zip(
                    *[
                        prompt_fn(**config.experiment.prompt_fn_kwargs)
                        for _ in range(config.sampling.batch_size)
                    ]
                )

                # encode prompts
                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                # sample
                with autocast():
                    ret_tuple = sample_trajectory(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=config.sampling.guidance_scale,
                        eta=config.sampling.eta,
                        output_type="pt",
                        return_unetoutput=config.model.unet_reg_scale > 0.,
                    )

                if config.model.unet_reg_scale > 0:
                    images, _, latents, scores, unet_outputs = ret_tuple
                    unet_outputs = torch.stack(unet_outputs, dim=1)  # (batch_size, num_steps, 3, 32, 32)
                else:
                    images, _, latents, scores = ret_tuple

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                scores = torch.stack(scores, dim=1)  # (batch_size, num_steps, 1)
                timesteps = pipeline.scheduler.timesteps.repeat(
                    config.sampling.batch_size, 1
                )  # (bs, num_steps)  (981, 961, ..., 21, 1) corresponds to "next_latents"
                step_index = torch.arange(timesteps.size(1), device=timesteps.device, dtype=torch.int64).view(1, -1).expand(timesteps.size(0), -1)

                rewards = reward_fn(images.float(), prompts, prompt_metadata) # (reward, reward_metadata)
                samples.append(
                    {
                        "prompts": prompts, # tuple of strings
                        "prompt_metadata": prompt_metadata,
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[
                            :, :-1
                        ],
                        "next_latents": latents[
                            :, 1:
                        ],
                        "scores": scores,
                        "rewards": rewards,
                        "step_index": step_index
                    }
                )
                if config.model.unet_reg_scale > 0:
                    samples[-1]["unet_outputs"] = unet_outputs


            # wait for all rewards to be computed
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not is_local_main_process,
                position=0,
            ):
                rewards, reward_metadata = sample["rewards"]
                sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            new_samples = {}
            for k in samples[0].keys():
                if k in ["prompts", "prompt_metadata"]:
                    # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                    new_samples[k] = [item for s in samples for item in s[k]]
                else:
                    new_samples[k] = torch.cat([s[k] for s in samples])
            samples = new_samples

            if epoch >= 0:
                # Save images to output directory
                if is_local_main_process:
                    images_dir = os.path.join(config.saving.output_dir, f"images_epoch{epoch}_step{global_step}")
                    os.makedirs(images_dir, exist_ok=True)
                    for i, image in enumerate(images):
                        pil = Image.fromarray(
                            (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        pil = pil.resize((256, 256))
                        pil.save(os.path.join(images_dir, f"{i}.jpg"))
                
                # this is a hack to force wandb to log the images as JPEGs instead of PNGs
                if config.logging.use_wandb:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for i, image in enumerate(images):
                            pil = Image.fromarray(
                                (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            )
                            pil = pil.resize((256, 256))
                            pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                        if is_local_main_process:
                            wandb.log(
                                {
                                    "images": [
                                        wandb.Image(
                                            os.path.join(tmpdir, f"{i}.jpg"),
                                            caption=f"{prompt} | {reward:.2f}",
                                        )
                                        for i, (prompt, reward) in enumerate(
                                            zip(prompts, rewards)
                                        )
                                    ],
                                },
                                step=global_step,
                            )

                rewards = torch.zeros(world_size * len(samples["rewards"]),
                            dtype=samples["rewards"].dtype, device=device)
                dist.all_gather_into_tensor(rewards, samples["rewards"])
                rewards = rewards.detach().cpu().float().numpy()
                result["reward_mean"][global_step] = rewards.mean()
                result["reward_std"][global_step] = rewards.std()

                if is_local_main_process:
                    logger.info(f"global_step: {global_step}  rewards: {rewards.mean().item():.3f}")
                    if config.logging.use_wandb:
                        wandb.log(
                            {
                                "reward_mean": rewards.mean(),
                                "reward_std": rewards.std(),
                            },
                            step=global_step,
                        )

                del samples["prompt_ids"]

                total_batch_size, num_timesteps = samples["timesteps"].shape
                assert (
                    total_batch_size
                    == config.sampling.batch_size * config.sampling.num_batches_per_epoch
                )
                assert num_timesteps == num_inference_steps
            


        ### No sampling for Epoch -1 
        if curr_samples is None:
            curr_samples = samples
            continue

        #################### TRAINING ####################
        for inner_epoch in range(config.training.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=device)
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    curr_samples[k] = [v[i] for i in perm]
                elif k in ["unet_outputs"]:
                    curr_samples[k] = v[perm]
                else:
                    curr_samples[k] = v[perm]

            if config.model.timestep_fraction < 1:
                if config.sampling.low_var_subsampling:
                    n_trunks = int(num_inference_steps * config.model.timestep_fraction)
                    assert n_trunks >= 1, "Must have at least one trunk"
                    assert num_inference_steps % n_trunks == 0, "num_inference_steps must be divisible by n_trunks"

                    trunk_size = num_inference_steps // n_trunks
                    step_indices = torch.arange(num_inference_steps, device=device)
                    trunks = step_indices.view(n_trunks, trunk_size)  # shape: (n_trunks, trunk_size)

                    # Precompute trunk access pattern (reversed order, repeated)
                    trunk_order = list(reversed(range(n_trunks))) * trunk_size  # len = num_inference_steps

                    perms_list = []
                    for _ in range(total_batch_size):
                        tmp = []
                        for i in trunk_order:
                            trunk = trunks[i]
                            index = torch.randint(0, trunk_size, (1,))
                            tmp.append(trunk[index])
                        interleaved = torch.cat(tmp)
                        perms_list.append(torch.cat([torch.tensor([num_inference_steps - 1], device=device), interleaved]))

                    perms = torch.stack(perms_list)  # shape: (batch_size, 1 + chunk_size * n_trunks)
                else:
                    perms = torch.stack(
                        [
                            torch.randperm(num_timesteps - 1, device=device)
                            for _ in range(total_batch_size)
                        ]
                    ) # (total_batch_size, num_steps)
                    perms = torch.cat([num_timesteps - 1 + torch.zeros_like(perms[:, :1]), perms], dim=1)
            else:
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=device)
                        for _ in range(total_batch_size)
                    ]
                ) # (total_batch_size, num_steps)

            # "prompts" & "prompt_metadata" are constant along time dimension
            key_ls = ["timesteps", "latents", "next_latents", "scores", "step_index"]
            for key in key_ls:
                curr_samples[key] = curr_samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]
            if config.model.unet_reg_scale > 0:
                curr_samples["unet_outputs"] = \
                    curr_samples["unet_outputs"][torch.arange(total_batch_size, device=device)[:, None], perms]

            ### rebatch for training
            samples_batched = {}
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples_batched[k] = [v[i:i + config.training.batch_size]
                                for i in range(0, len(v), config.training.batch_size)]
                elif k in ["unet_outputs"]:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])
                else:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            unet.train()
            if not config.model.no_flow:
                res_logflowscore_model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_local_main_process,
            ):
                """
                sample: [
                ('prompts', list of strings, len=train_bs), ('prompt_metadata', list of dicts),
                (bf16) ('prompt_embeds', torch.Size([1, 77, 768])),
                (int64) ('timesteps', torch.Size([1, 50])),
                (bf16) ('latents', torch.Size([1, 50, 4, 64, 64])), ('next_latents', torch.Size([1, 50, 4, 64, 64])),
                ]
                """
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [train_neg_prompt_embeds, sample["prompt_embeds"]]
                )

                buffer = []
                for j in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not is_local_main_process):
                    with autocast():
                        latent_tmp = sample["latents"][:, j].clone().detach()
                        latent_tmp.requires_grad_(True)

                        # Before inference, disable the LoRA adapters
                        unet.module.disable_adapters()  # This should deactivate any applied LoRA adapter

                        with ref_compute_mode():
                            noise_pred_ref = unet(
                                torch.cat([latent_tmp] * 2),
                                torch.cat([sample["timesteps"][:, j]] * 2),
                                embeds,
                            ).sample
                            noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(2)
                            noise_pred_ref = (
                                    noise_pred_uncond_ref
                                    + config.sampling.guidance_scale
                                    * (noise_pred_text_ref - noise_pred_uncond_ref)
                            )
                            noise_pred_uncond_ref = noise_pred_text_ref = None

                        with torch.inference_mode():
                            _, score_pf_ref = inference_step(
                                pipeline.scheduler, noise_pred_ref,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sampling.eta,
                                prev_sample=sample["next_latents"][:, j],
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                            )

                        if config.model.reverse_loss_scale > 0:
                            _, score_pf_ref_reverse = inference_step(
                                pipeline.scheduler, noise_pred_ref,
                                sample["timesteps"][:, j],
                                latent_tmp,
                                eta=config.sampling.eta,
                                prev_sample=sample["next_latents"][:, j],
                                reverse_grad=True,
                                retain_graph=False,
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                            )
                            score_pf_ref_reverse = score_pf_ref_reverse.detach()
                        _ = noise_pred_ref = None


                        # Optionally, you can re-enable the LoRA adapters after inference if needed
                        unet.module.enable_adapters()  # Re-apply LoRA configuration
                        unet.module.set_adapter("pf")

                        noise_pred = unet(
                            torch.cat([latent_tmp] * 2),
                            torch.cat([sample["timesteps"][:, j]] * 2),
                            embeds,
                        ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = (
                                noise_pred_uncond
                                + config.sampling.guidance_scale
                                * (noise_pred_text - noise_pred_uncond)
                        )
                        noise_pred_uncond = noise_pred_text = None
                        if config.model.unet_reg_scale > 0:
                            unetdiff = (noise_pred - sample["unet_outputs"][:, j]).pow(2)
                            unetreg = torch.mean(unetdiff, dim=(1, 2, 3))
                            unetdiffnorm = unetdiff.sum(dim=[1,2,3]).sqrt()

                        _, score_pf = inference_step(
                            pipeline.scheduler, noise_pred,
                            sample["timesteps"][:, j],
                            latent_tmp,
                            eta=config.sampling.eta,
                            prev_sample=sample["next_latents"][:, j],
                            step_index=sample["step_index"][:, j],
                        )
                        _ = None

                        if config.model.reverse_loss_scale > 0:
                            _, score_pf_reverse = inference_step(
                                pipeline.scheduler, noise_pred,
                                sample["timesteps"][:, j],
                                latent_tmp, eta=config.sampling.eta,
                                prev_sample=sample["next_latents"][:, j],
                                reverse_grad=True,
                                retain_graph=True,
                                allow_2nd=False,
                                strength=config.model.pretrained_strength,
                                step_index=sample["step_index"][:, j],
                            )
                        _ = None

                        #######################################################
                        #################### GFN ALGORITHM ####################
                        #######################################################
                        timestep_next = torch.clamp(sample["timesteps"][:, j] - scheduler_dt, min=0)
                        end_mask = sample["timesteps"][:, j] == pipeline.scheduler.timesteps[-1] # RHS is 1

                        latent_next_tmp = sample["next_latents"][:, j].detach().clone()
                        latent_next_tmp.requires_grad_()
                        unet.module.set_adapter("pf")


                        noise_pred_next_tmp = unet(
                            torch.cat([latent_next_tmp] * 2),
                            torch.cat([timestep_next] * 2),
                            embeds,
                        ).sample
                        noise_pred_uncond_next_tmp, noise_pred_next_text_tmp = noise_pred_next_tmp.chunk(2)
                        noise_pred_next_tmp = (
                                noise_pred_uncond_next_tmp
                                + config.sampling.guidance_scale
                                * (noise_pred_next_text_tmp - noise_pred_uncond_next_tmp)
                        )
                        noise_pred_uncond_next_tmp = noise_pred_next_text_tmp = None
                        pred_z0_next = predict_clean(
                            pipeline.scheduler,
                            noise_pred_next_tmp,
                            latent_next_tmp,
                            timestep_next
                        )
                        noise_pred_next_tmp = None
                        pred_xdata_next = decode(pred_z0_next).float()

                        with torch.cuda.amp.autocast(enabled=False):
                            logr_next_tmp = reward_fn(pred_xdata_next, prompts, prompt_metadata)[0]
                            score_r_next_tmp = torch.autograd.grad(
                                outputs=logr_next_tmp.sum(),    # The value whose gradient we want
                                inputs=latent_next_tmp,         # The intermediate node we want the gradient with respect to
                                retain_graph=False,             # Retain graph for further gradient computations
                                create_graph=False              # If higher-order gradients are needed
                            )[0].detach()
                            latent_next_tmp = None
                            score_r_next = config.model.reward_scale * score_r_next_tmp
                            alpha_prod_next = get_alpha_prod_t(pipeline.scheduler, timestep_next, sample["next_latents"][:, j])
                            if config.model.reward_adaptive_mode == 'squared':
                                score_r_next = score_r_next * alpha_prod_next
                            else:
                                score_r_next = score_r_next * alpha_prod_next.sqrt()

                        score_r_next_tmp = None

                        if config.model.reverse_loss_scale > 0:
                            pred_z0 = predict_clean(pipeline.scheduler, noise_pred, latent_tmp, sample["timesteps"][:, j])
                            noise_pred = None
                            pred_xdata = decode(pred_z0).float()
                            with torch.cuda.amp.autocast(enabled=False):
                                logr_tmp = reward_fn(pred_xdata, prompts, prompt_metadata)[0]
                                score_r_tmp = torch.autograd.grad(
                                    outputs=logr_tmp.sum(),    # The value whose gradient we want
                                    inputs=latent_tmp,         # The intermediate node we want the gradient with respect to
                                    retain_graph=True,         # Retain graph for further gradient computations
                                    create_graph=False         # If higher-order gradients are needed
                                )[0].detach()
                                latent_tmp = None
                                score_r = config.model.reward_scale * score_r_tmp
                            score_r_tmp = None
                            alpha_prod = get_alpha_prod_t(pipeline.scheduler, sample["timesteps"][:, j], sample["next_latents"][:, j])
                            if config.model.reward_adaptive_mode == 'squared':
                                score_r = score_r * alpha_prod
                            else:
                                score_r = score_r * alpha_prod.sqrt()

                    if not config.model.no_flow:
                        with flow_cast_float32():
                            if config.experiment.reward_fn == 'aesthetic':
                                flow_prompt_embeds = torch.zeros_like(sample["prompt_embeds"]).float()
                            else:
                                flow_prompt_embeds = sample["prompt_embeds"].float()
                            res_logflow_next = res_logflowscore_model(sample["next_latents"][:, j].float(), timestep_next, flow_prompt_embeds).sample
                            res_logflow = res_logflowscore_model(sample["latents"][:, j].float(), sample["timesteps"][:, j], flow_prompt_embeds).sample

                    if config.model.no_flow:
                        score_pf_target = (score_pf_ref.float() + score_r_next).float()
                    else:
                        score_pf_target = (score_pf_ref.float() + res_logflow_next + score_r_next).float()
                    score_pf_target[end_mask] = (score_pf_ref[end_mask].float() + score_r_next[end_mask].float()).detach()

                    if config.model.reverse_loss_scale > 0:
                        score_pf_reverse_target = (score_pf_ref_reverse.float() - res_logflow - score_r.detach()).float()
                    else:
                        flow_reverse_target = None

                    if config.model.reverse_loss_scale > 0:
                        score_pf_reverse_target[end_mask] = torch.zeros_like(score_pf_reverse_target[end_mask])

                    with torch.inference_mode():
                        grad_norm_score_ref = score_pf_ref.pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_res_score = (score_pf - score_pf_ref).pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_score_r = score_r_next.pow(2).sum(dim=[1,2,3]).sqrt()
                        if not config.model.no_flow:
                            grad_norm_resflow = res_logflow_next.pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_score_pf_target = score_pf_target.pow(2).sum(dim=[1,2,3]).sqrt()
                        if config.model.reverse_loss_scale > 0:
                            grad_norm_score_pf_reverse_target = score_pf_reverse_target.pow(2).sum(dim=[1,2,3]).sqrt()

                    score_pf_ref = None
                    score_r_next = None
                    score_r = None
                    score_pf_ref_reverse = None
                    score_pb_reverse = None


                    if not config.model.no_flow:
                        loss_terminal = (res_logflow_next.pow(2).mean(dim=[1,2,3]) * end_mask.float()).sum() / (end_mask.float().sum() + 1e-6)
                    else:
                        loss_terminal = torch.zeros(1, device=score_pf.device)

                    losses_forward = (score_pf - score_pf_target).pow(2)
                    score_pf = score_pf_target = None
                    loss_forward_mean = losses_forward.mean()

                    if config.model.reverse_loss_scale > 0:
                        losses_backward = (score_pf_reverse - score_pf_reverse_target).pow(2)  # (bs,)
                        score_pf_reverse = score_pf_reverse_target = None
                        loss_backward_mean = losses_backward.mean()
                    else:
                        losses_backward = torch.zeros_like(losses_forward)
                        loss_backward_mean = 0.0

                    losses = (losses_forward + config.model.reverse_loss_scale * losses_backward + loss_terminal).mean()

                    if config.model.unet_reg_scale > 0:
                        losses = losses + config.model.unet_reg_scale * unetreg.mean()
                    loss = torch.mean(losses)

                    loss = loss / accumulation_steps
                    if scaler:
                        # Backward passes under autocast are not recommended
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()


                    #### Log
                    info["loss"].append(loss_forward_mean + loss_backward_mean)
                    info["loss_terminal"].append(loss_terminal)
                    info["loss_forward"].append(loss_forward_mean.detach())
                    if config.model.reverse_loss_scale > 0:
                        info["losses_backward"].append(loss_backward_mean.detach())

                    with torch.inference_mode():
                        info["norm_score_ref_mean"].append(grad_norm_score_ref.mean())
                        info["norm_score_ref_min"].append(grad_norm_score_ref.min())
                        info["norm_score_ref_max"].append(grad_norm_score_ref.max())
                        info["norm_score_residual_mean"].append(grad_norm_res_score.mean())
                        info["norm_score_residual_min"].append(grad_norm_res_score.min())
                        info["norm_score_residual_max"].append(grad_norm_res_score.max())
                        info["norm_score_r_mean"].append(grad_norm_score_r.mean())
                        info["norm_score_r_min"].append(grad_norm_score_r.min())
                        info["norm_score_r_max"].append(grad_norm_score_r.max())
                        if config.model.unet_reg_scale > 0:
                            info["norm_unet_diff_mean"].append(unetdiffnorm.mean())
                            info["norm_unet_diff_mean"].append(unetdiffnorm.min())
                            info["norm_unet_diff_mean"].append(unetdiffnorm.max())

                    info["losses_forward_max"].append(losses_forward.max())
                    info["losses_backward_max"].append(losses_backward.max())
                    info["losses_bidir_max"].append((losses_forward + config.model.reverse_loss_scale * losses_backward).max())
                    if config.model.unet_reg_scale > 0:
                        info["unetreg"].append(unetreg.mean().detach())


                    # prevent OOM
                    image = None
                    noise_pred_uncond = noise_pred_text = noise_pred = None
                    logr_next_tmp = logr_tmp = None
                    _ = log_pf = log_pb = None
                    score_pb_reverse = None
                    unetreg = losses =  None
                    score_pf_ref = score_pf = None
                    score_r_next = score_r_next_tmp = None
                    noise_pred_uncond_ref = noise_pred_text_ref = noise_pred_ref = None
                    score_pf_target = None
                    res_logflow = res_logflow_next = None
                    grad_norm_score_pf_target = grad_norm_score_pf_reverse_target = None



                if ((j == num_train_timesteps - 1) and
                        (i + 1) % config.training.gradient_accumulation_steps == 0):
                    if scaler:
                        scaler.unscale_(optimizer)
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in unet.named_parameters() if '.pf.' in name], config.training.max_grad_norm)
                        if not config.model.no_flow:
                            flow_update_grad = torch.nn.utils.clip_grad_norm_(res_logflowscore_model.parameters(), config.training.max_grad_norm)

                        scaler.step(optimizer)
                        # optimizer.step()
                        scaler.update()
                    else:
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in unet.named_parameters() if '.pf.' in name], config.training.max_grad_norm)
                        if not config.model.no_flow:
                            flow_update_grad = torch.nn.utils.clip_grad_norm_(res_logflowscore_model.parameters(), config.training.max_grad_norm)

                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    ### avoid memory leak
                    if not config.model.no_flow:
                        for param in res_logflowscore_model.parameters():
                            param.grad = None
                    for param in unet.parameters():
                        param.grad = None

                    old_info = info
                    info = {}
                    for k, v in old_info.items():
                        if '_min' in k:
                            info[k] = torch.min(torch.stack(v))
                        elif '_max' in k:
                            info[k] = torch.max(torch.stack(v))
                        else:
                            try:
                                info[k] = torch.mean(torch.stack(v))
                            except Exception as e:
                                print(k)
                                print(v)
                                raise e

                    dist.barrier()
                    for k, v in info.items():
                        if '_min' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MIN)
                        elif '_max' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MAX)
                        else:
                            dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    info = {k: v / num_processes if ('_min' not in k and '_max' not in k) else v for k, v in info.items()}
                    for k, v in info.items():
                        result[k][global_step] = v.item()

                    info.update({"epoch": epoch})
                    info.update({"global_step": global_step})
                    result["epoch"][global_step] = epoch
                    result["time"][global_step] = time.time() - start_time


                    if is_local_main_process:
                        if scaler:
                            info.update({"grad_scale": scaler.get_scale()})
                            result["grad_scale"] = scaler.get_scale()


                    if is_local_main_process:
                        if config.logging.use_wandb:
                            wandb.log(info, step=global_step)
                        logger.info(f"global_step={global_step}  " +
                              " ".join([f"{k}={v:.6f}" for k, v in info.items()]))
                    info = defaultdict(list) # reset info dict


        curr_samples = samples
        if is_local_main_process:
            pickle.dump(result, gzip.open(os.path.join(config.saving.output_dir, f"result.json"), 'wb'))
        dist.barrier()

        if epoch % config.logging.save_freq == 0 or epoch == config.training.num_epochs - 1:
            if is_local_main_process:
                save_path = os.path.join(config.saving.output_dir, f"checkpoint_epoch{epoch}")
                unwrapped_unet = unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
                )
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )
                conv_out_weights = unwrapped_unet.conv_out.state_dict()  # Extract only the weights of the conv_out layer
                torch.save(conv_out_weights, os.path.join(config.saving.output_dir, f"conv_out_weights_epoch{epoch}.pt"))
                logger.info(f"Saved state to {save_path}")

            dist.barrier()

    if is_local_main_process:
        save_path = os.path.join(config.saving.output_dir, f"checkpoint_epoch{epoch}")
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            is_main_process=is_local_main_process,
            safe_serialization=True,
        )
        conv_out_weights = unwrapped_unet.conv_out.state_dict()  # Extract only the weights of the conv_out layer
        torch.save(conv_out_weights, os.path.join(config.saving.output_dir, f"conv_out_weights_epoch{epoch}.pt"))
        logger.info(f"Saved state to {save_path}")
    dist.barrier()

    if config.logging.use_wandb and is_local_main_process:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
  app.run(main)
