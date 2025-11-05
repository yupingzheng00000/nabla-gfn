git config --global user.name "yupingzheng00000"
git config --global user.email "dd719876254@outlook.com"

# Kill any existing training process
pkill -f "train_nablagfn.py" || true
sleep 2

# Clean log
rm -f nohup.out

# Start training with WandB enabled and gradient accumulation
# Updated hyperparameters (方案2):
# - reward_scale: 1e4 → 1e3 (reduce advantage magnitude)
# - beta: 0.05 → 0.1 (stronger KL constraint)
# - lr: 1e-3 → 3e-4 (smaller learning rate)
# - unet_reg_scale: 0 → 1e3 (add UNet regularization)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup setsid torchrun --standalone --nproc_per_node=2 train_nablagfn.py \
  --config=config/aesthetic.py \
  --mode=grpo \
  --seed=0 \
  --config.model.reward_scale=1e3 \
  --config.model.timestep_fraction=0.1 \
  --config.model.unet_reg_scale=2e3 \
  --config.sampling.batch_size=3 \
  --config.sampling.num_batches_per_epoch=5 \
  --config.sampling.low_var_subsampling=True \
  --config.training.lr=3e-4 \
  --config.training.gradient_accumulation_steps=5 \
  --config.training.num_epochs=400 \
  --config.grpo.beta=0.15 \
  --config.grpo.clip_range=0.1 \
  --config.grpo.group_size=4 \
  --config.logging.use_wandb=True \
  --config.logging.proj_name=nabla-gfn-grpo \
  --config.saving.output_dir=./outputs/aesthetic_grpo_ddim \
  --exp_name=aesthetic_grpo_ddim \
  > nohup.out 2>&1 &

echo "Training started. Monitor with: tail -f nohup.out"
