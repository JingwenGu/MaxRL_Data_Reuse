echo "Script started"
module load python/anaconda3/2.10.0
conda activate maxrl

export HF_HUB_HTTP_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/work/nvme/bhxl/jgu3/hf_cache

# All overridable via `sbatch --export=ALL,LEARNING_RATE=...,ROLLOUT_FILTER=...,LR_SCHEDULE=... rollout_sft_sbatch.sh`
LEARNING_RATE=${LEARNING_RATE:-0.1}
SEED=69

# "all": every sampled rollout is an SFT example.
# "correct": only rollouts where the sampled action matched the true label (see
# rollout_sft.py's module docstring for what this mode collapses to mathematically).
ROLLOUT_FILTER=${ROLLOUT_FILTER:-all}

# "constant": LEARNING_RATE held fixed the whole replay.
# "cosine_with_warmup": matches the online run's own schedule; LEARNING_RATE is the peak.
LR_SCHEDULE=${LR_SCHEDULE:-constant}

# Only affects ROLLOUT_FILTER=correct. "per-rollout" (original): normalizes by this step's
# own correct-rollout count. "fixed-batch": normalizes by the constant B*K instead (see
# rollout_sft.py's --correct-loss-norm help for the full rationale).
CORRECT_LOSS_NORM=${CORRECT_LOSS_NORM:-per-rollout}

# Rollouts saved by the MaxRL run being replayed
ROLLOUT_DIR=/work/nvme/bhxl/jgu3/imagenet256_checkpoints/reinforce_with_p_normalization_1024

# Which of that run's saved epochs to replay, in order
START_EPOCH=1
END_EPOCH=20

RUN_NAME=sft_on_rollouts_${ROLLOUT_FILTER}_reinforce_with_p_normalization_1024_lr_${LEARNING_RATE}_${LR_SCHEDULE}
if [ "$ROLLOUT_FILTER" = "correct" ]; then
    RUN_NAME=${RUN_NAME}_${CORRECT_LOSS_NORM}
fi

python -m verl.cifar10_experiments.rollout_sft \
    --wandb \
    --lr $LEARNING_RATE \
    --lr-schedule $LR_SCHEDULE \
    --seed $SEED \
    --rollout-dir $ROLLOUT_DIR \
    --rollout-filter $ROLLOUT_FILTER \
    --correct-loss-norm $CORRECT_LOSS_NORM \
    --start-epoch $START_EPOCH \
    --end-epoch $END_EPOCH \
    --checkpoint-dir /work/nvme/bhxl/jgu3/imagenet256_checkpoints/${RUN_NAME} \
    --wandb_runname $RUN_NAME \
    --wandb-project imagenet256_reinforce_learning_rate_ablations \
    --wandb-entity paprika_online
