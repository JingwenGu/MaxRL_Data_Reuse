echo "Script started"
module load python/anaconda3/2.10.0
conda activate maxrl

export HF_HUB_HTTP_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/work/nvme/bhxl/jgu3/hf_cache

LEARNING_RATE=0.1
SEED=69

# "all": every sampled rollout is an SFT example.
# "correct": only rollouts where the sampled action matched the true label (see
# rollout_sft.py's module docstring for what this mode collapses to mathematically).
ROLLOUT_FILTER=correct

# Rollouts saved by the MaxRL run being replayed
ROLLOUT_DIR=/work/nvme/bhxl/jgu3/imagenet256_checkpoints/reinforce_with_p_normalization_1024

# Which of that run's saved epochs to replay, in order
START_EPOCH=1
END_EPOCH=20

RUN_NAME=sft_on_rollouts_${ROLLOUT_FILTER}_reinforce_with_p_normalization_1024_lr_${LEARNING_RATE}

python -m verl.cifar10_experiments.rollout_sft \
    --wandb \
    --lr $LEARNING_RATE \
    --seed $SEED \
    --rollout-dir $ROLLOUT_DIR \
    --rollout-filter $ROLLOUT_FILTER \
    --start-epoch $START_EPOCH \
    --end-epoch $END_EPOCH \
    --checkpoint-dir /work/nvme/bhxl/jgu3/imagenet256_checkpoints/${RUN_NAME} \
    --wandb_runname $RUN_NAME \
    --wandb-project imagenet256_reinforce_learning_rate_ablations
