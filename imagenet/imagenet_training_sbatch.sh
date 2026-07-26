#!/bin/bash
#SBATCH --account=bhxl-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=imagenet256_maxrl
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/work/nvme/bhxl/jgu3/imagenet256_checkpoints/logs/%x_%j.out
#SBATCH --error=/work/nvme/bhxl/jgu3/imagenet256_checkpoints/logs/%x_%j.err

bash /u/jgu3/maxrl/imagenet/imagenet_training_script.sh
