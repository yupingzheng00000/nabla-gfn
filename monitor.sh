#!/bin/bash

echo "=== Training Monitor ==="
echo ""
echo "WandB Dashboard:"
echo "https://wandb.ai/dd719876254-the-chinese-university-of-hong-kong/nabla-gfn-grpo"
echo ""
echo "=== Latest Logs ==="
tail -n 30 nohup.out | grep -E "(epoch|loss|reward|ratio|clip_frac)"
echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "=== Process Status ==="
ps aux | grep train_nablagfn.py | grep -v grep
