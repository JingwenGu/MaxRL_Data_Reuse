#!/usr/bin/env python3
"""
Supervised fine-tuning of a fresh ResNet-50 directly on the training-time rollouts
saved by sampling_based_rl_objective_experiments.py, replayed in the exact order the
original MaxRL run produced them: source epoch ascending, then step ascending within
an epoch, then image index within a step.

Each saved step recorded one forward pass over B unique images, each with K sampled
rollout actions attached. Rather than materializing every rollout as its own (image,
label) example -- which would mean redundantly forwarding the same image up to K times
-- this script forwards each step's B images once and computes the cross-entropy loss
against all B*K labels via a single gather. This is mathematically identical to
training on every one of the B*K rollouts as an independent SFT example.

--rollout-filter controls which rollouts count as SFT examples:
  all      Every one of the B*K sampled rollouts, correct or not (the default).
  correct  Only rollouts where the sampled action matched the true label. Note this
           is NOT self-distillation on varied model outputs -- a "correct" rollout's
           action is by definition equal to the true label, so this mode collapses to
           standard cross-entropy against the true label, with each image weighted by
           how many of its K rollouts happened to land on the correct class. Images
           the model rarely got right at that point in the original run contribute
           less (or nothing); images it usually got right contribute more.

Usage:
    python -m verl.cifar10_experiments.rollout_sft \
        --rollout-dir /path/to/imagenet256_checkpoints/reinforce_with_p_normalization_1024 \
        --checkpoint-dir /path/to/imagenet256_checkpoints/sft_on_rollouts
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms as T
from datasets import load_dataset

import wandb

from verl.cifar10_experiments.sampling_based_rl_objective_experiments import (
    apply_deterministic_transform,
    set_seed,
    get_device,
    count_params,
    get_checkpoint_state,
    save_checkpoint,
    HFImageNet,
    evaluate,
)


class RolloutReplayDataset(IterableDataset):
    """
    Streams saved training-time rollouts (rollouts_epoch{N}.pt files) in exact original
    order. Each yielded item is one full original training step:
        (images: (B, 3, crop_size, crop_size), labels: (B, L), weights: (B, L),
         epoch: int, step: int)

    labels/weights depend on rollout_filter:
      "all"     labels = all K sampled actions (B, K); weights = all ones. Loss/metrics
                computed over this reduce to an unweighted mean over every rollout.
      "correct" labels = the true target repeated once per image (B, 1); weights = how
                many of that image's K rollouts were correct (B, 1). See module
                docstring for why this collapses to weighted true-label training
                rather than distillation on varied sampled outputs.

    Rollout files are read one epoch at a time and dropped before the next is loaded,
    so at most one ~4GB epoch file is held in memory. Files on disk are only ever read
    -- this dataset never modifies or deletes them.

    Must be used with num_workers=0: multiple DataLoader workers would interleave steps
    non-deterministically, breaking the exact-order guarantee.
    """
    def __init__(self, rollout_dir, hf_ds, start_epoch, end_epoch, crop_size, mean, std,
                 rollout_filter="all"):
        assert rollout_filter in ("all", "correct")
        self.rollout_dir = rollout_dir
        self.ds = hf_ds["train"]
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.crop_size = crop_size
        self.rollout_filter = rollout_filter
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError(
                "RolloutReplayDataset requires num_workers=0 to preserve exact replay "
                "order; multiple workers would interleave steps non-deterministically."
            )

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            path = os.path.join(self.rollout_dir, f"rollouts_epoch{epoch}.pt")
            rollouts = torch.load(path, weights_only=False)
            num_steps = len(rollouts["global_step"])

            for step_idx in range(num_steps):
                img_idx = rollouts["img_idx"][step_idx]
                ci = rollouts["crop_i"][step_idx]
                cj = rollouts["crop_j"][step_idx]
                ch = rollouts["crop_h"][step_idx]
                cw = rollouts["crop_w"][step_idx]
                flip = rollouts["flip"][step_idx]
                actions = rollouts["actions"][step_idx]  # (B, K)

                if self.rollout_filter == "all":
                    labels = actions
                    weights = torch.ones_like(actions, dtype=torch.float32)
                else:
                    target = rollouts["target"][step_idx]              # (B,)
                    rewards = rollouts["rewards"][step_idx]             # (B, K)
                    labels = target.unsqueeze(1)                       # (B, 1)
                    weights = rewards.sum(dim=1, keepdim=True).float()  # (B, 1): correct-rollout count

                B = img_idx.shape[0]
                images = torch.empty(B, 3, self.crop_size, self.crop_size)
                for b in range(B):
                    img = self.ds[int(img_idx[b])]["image"]
                    images[b] = apply_deterministic_transform(
                        img,
                        int(ci[b]), int(cj[b]), int(ch[b]), int(cw[b]), bool(flip[b]),
                        self.crop_size, self.to_tensor, self.normalize,
                    )

                yield images, labels, weights, epoch, int(rollouts["global_step"][step_idx])

            del rollouts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=str, required=True,
                         help="Directory containing rollouts_epoch{N}.pt files from the MaxRL run")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                         help="Where to save SFT checkpoints (one per source epoch replayed)")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=20)
    parser.add_argument("--rollout-filter", type=str, default="all", choices=["all", "correct"],
                         help="'all': every sampled rollout is an SFT example. "
                              "'correct': only rollouts where the sampled action matched the "
                              "true label (see module docstring for what this collapses to)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--max-grad-norm", type=float, default=1e5)
    parser.add_argument("--eval_every_k", type=int, default=1000, help="Run validation every K optimizer steps")
    parser.add_argument("--num_test_rollouts_per_example", type=int, default=1024,
                         help="Highest k for val/pass@k (reported for k=1,2,4,...,up to this value)")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers for the validation loader")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="imagenet256_reinforce_learning_rate_ablations")
    parser.add_argument("--wandb_runname", type=str, default="sft_on_rollouts")
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_runname,
            config=vars(args),
        )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    print("Loading imagenet 256 x 256 (for image reconstruction only; labels come from saved rollouts) ...")
    hf_ds = load_dataset("benjamin-paine/imagenet-1k-256x256")

    # Validation set/transform mirrors get_dataloaders()'s imagenet256 test path exactly,
    # built directly here (rather than via get_dataloaders) to avoid loading the dataset
    # a second time and constructing an unused training-side dataset.
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    test_set = HFImageNet(split="validation", transform=test_transform, hf_ds=hf_ds)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    dataset = RolloutReplayDataset(
        rollout_dir=args.rollout_dir,
        hf_ds=hf_ds,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        crop_size=224,
        mean=mean,
        std=std,
        rollout_filter=args.rollout_filter,
    )
    # batch_size=None: the dataset already yields one complete training batch per item
    # (one original step's worth of images+labels) -- don't let DataLoader re-batch it.
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    model = models.resnet50(num_classes=1000).to(device)
    total, trainable = count_params(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False,
    )
    # Constant LR: this replay doesn't know the total step count up front without a
    # separate pass over the rollout files, so it skips the original run's
    # cosine-with-warmup schedule. Swap in a real schedule here if you want one.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    global_step = 0
    current_epoch = None

    val_loss, val_acc, val_passk = evaluate(
        model, test_loader, criterion, device,
        num_validation_rollouts=args.num_test_rollouts_per_example,
    )
    print(f"Before any training: val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
    if args.wandb:
        wandb.log({
            "val/loss": val_loss,
            "val/acc": val_acc,
            **{f"val/{k}": v for k, v in val_passk.items()},
            "sft/global_step": global_step,
        })

    for images, labels, weights, source_epoch, source_step in loader:
        model.train()
        global_step += 1

        if current_epoch is None:
            current_epoch = source_epoch
        elif source_epoch != current_epoch:
            save_checkpoint(
                state=get_checkpoint_state(
                    model, optimizer, scheduler, epoch=current_epoch, global_step=global_step, best_acc=0.0,
                ),
                is_best=False,
                checkpoint_dir=args.checkpoint_dir,
                steps=f"epoch{current_epoch}",
            )
            print(f"Saved SFT checkpoint for replayed epoch {current_epoch}")
            current_epoch = source_epoch

        images = images.to(device)
        labels = labels.to(device).long()    # (B, L)
        weights = weights.to(device)         # (B, L)

        optimizer.zero_grad(set_to_none=True)
        # Why one forward pass instead of one per label: every column of `labels` for a
        # given image is a label attached to that *same* image (in "all" mode, K distinct
        # sampled actions; in "correct" mode, a single repeated true label), so its logits
        # are identical across all of them. Naively treating each label as its own SFT
        # example would mean forwarding the same image once per label -- for "all" mode
        # over the full run that's ~26 billion redundant forward passes. Instead we
        # forward each of the B unique images once and use a weighted gather to pull out
        # log p(label | image_b) for every (b, l) pair from that single (B, 1000) logits
        # tensor. In "all" mode every weight is 1, so this is a plain mean over all B*K
        # rollouts -- mathematically identical to averaging a per-rollout loss computed
        # one rollout at a time (verified numerically in scratchpad testing). In "correct"
        # mode weights are the per-image correct-rollout counts, so this is a weighted
        # mean against the true label. Either way this is purely a compute/memory
        # optimization, not an approximation.
        logits = model(images)                        # (B, 1000) -- one forward pass
        log_probs = F.log_softmax(logits, dim=1)
        weight_sum = weights.sum().clamp(min=1e-6)     # guards the (extremely unlikely) all-zero-weight step
        loss = -(log_probs.gather(1, labels) * weights).sum() / weight_sum

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # weighted accuracy of the model's *current* prediction against `labels`
            # (rollout-sampled in "all" mode, true label in "correct" mode) -- monitoring only
            pred = logits.argmax(1)
            acc_vs_rollouts = ((pred.unsqueeze(1) == labels).float() * weights).sum().item() / weight_sum.item()

        if global_step % 1000 == 0:
            print(
                f"[source epoch {source_epoch} step {source_step}] "
                f"sft_global_step={global_step} loss={loss.item():.4f} "
                f"acc_vs_rollout_labels={acc_vs_rollouts*100:.2f}%"
            )

        if args.wandb:
            wandb.log({
                "sft/loss": loss.item(),
                "sft/acc_vs_rollout_labels": acc_vs_rollouts,
                "sft/grad_norm": grad_norm.item(),
                "sft/lr": optimizer.param_groups[0]["lr"],
                "sft/global_step": global_step,
                "source_epoch": source_epoch,
                "source_step": source_step,
            })

        if global_step % args.eval_every_k == 0:
            val_loss, val_acc, val_passk = evaluate(
                model, test_loader, criterion, device,
                num_validation_rollouts=args.num_test_rollouts_per_example,
            )

            print(
                f"[sft_global_step {global_step}] "
                f"train_loss={loss.item():.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
            )

            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    **{f"val/{k}": v for k, v in val_passk.items()},
                    "sft/global_step": global_step,
                    "source_epoch": source_epoch,
                })

    if current_epoch is not None:
        save_checkpoint(
            state=get_checkpoint_state(
                model, optimizer, scheduler, epoch=current_epoch, global_step=global_step, best_acc=0.0,
            ),
            is_best=False,
            checkpoint_dir=args.checkpoint_dir,
            steps=f"epoch{current_epoch}",
        )
        print(f"Saved final SFT checkpoint for replayed epoch {current_epoch}")

    if args.wandb:
        wandb.finish()

    print("SFT-on-rollouts replay complete.")


if __name__ == "__main__":
    main()
