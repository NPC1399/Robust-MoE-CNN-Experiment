#!/bin/bash
#SBATCH --job-name=pl_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=pl_train_%j.log

# Activate your environment if needed
# source ~/env/bin/activate

srun python3 train_lightning.py \
    --arch resnet50_imagenet_moe \
    --dataset ImageNet \
    --normalize \
    --batch-size 256 \
    --gpus 4 --num-nodes 2
