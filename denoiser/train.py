import os
import tqdm
import wandb
from datetime import datetime

import random
import numpy as np

import hydra
from omegaconf import OmegaConf

import torch
import torch.optim as optim
import torch.nn as nn

from data import load_data, move_features_to_device
from models import load_model


@torch.no_grad()
def eval_model(model, dataloaders, device):
    print(f"EVALUATING MODEL")
    model.eval()

    losses = []
    metrics = []

    for dataloader in dataloaders:
        running_loss = 0.0
        running_l1 = 0.0
        running_l2 = 0.0

        for features, targets in tqdm.tqdm(dataloader):
            features = move_features_to_device(features, device)
            targets = move_features_to_device(targets, device)

            preds = model.forward_with_preprocess(features)

            running_loss += model.compute_loss(features, targets)[0]

            running_l1 += torch.mean(torch.abs(preds - targets["rgb"]))
            running_l2 += torch.mean((preds - targets["rgb"]) ** 2)

        losses.append(running_loss / len(dataloader))
        metrics.append(
            {
                "l1_error": running_l1 / len(dataloader),
                "l2_error": running_l2 / len(dataloader),
            }
        )

    model.train()
    return list(zip(losses, metrics))


def train(model, train_loader, val_loader, test_loader, config):
    device = config.device
    model.train()

    num_grad_steps = config.training.num_grad_steps
    log_interval = config.logging.log_interval
    eval_interval = config.logging.eval_interval

    save_ckpt = config.logging.save_ckpt
    ckpt_dir = config.logging.ckpt_dir
    ckpt_interval = config.logging.ckpt_interval

    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.lr)

    epoch = 0
    iter_num = 0

    while iter_num < num_grad_steps:
        print(f"Epoch {epoch}")

        for features, targets in tqdm.tqdm(train_loader):
            if iter_num == num_grad_steps:
                break

            features = move_features_to_device(features, device)
            targets = move_features_to_device(targets, device)

            train_loss, metrics = model.compute_loss(features, targets)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            to_log = {}

            if (iter_num + 1) % log_interval == 0:
                to_log["iter"] = iter_num
                to_log["train/train_loss"] = train_loss

                log_str = [f"{iter_num}: train_loss = {train_loss}"]

                for metric_name, metric_val in metrics.items():
                    to_log[f"train/{metric_name}"] = metric_val
                    log_str.append(f"{metric_name} = {metric_val}")

                print(", ".join(log_str))

            if (iter_num + 1) % eval_interval == 0:
                (val_loss, val_metrics), (test_loss, test_metrics) = eval_model(
                    model, [val_loader, test_loader], device
                )

                to_log["iter"] = iter_num
                to_log["val/val_loss"] = val_loss
                to_log["test/test_loss"] = test_loss

                log_str = [f"{iter_num}: val_loss = {val_loss}"]
                for metric_name, metric_val in val_metrics.items():
                    to_log[f"val/{metric_name}"] = metric_val
                    log_str.append(f"{metric_name} = {metric_val}")

                for metric_name, metric_val in test_metrics.items():
                    to_log[f"test/{metric_name}"] = metric_val

                print(", ".join(log_str))

            if len(to_log) > 0:
                wandb.log(to_log)

            if save_ckpt and (iter_num + 1) % ckpt_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, f"iter_{iter_num}.pt"),
                )

            iter_num += 1

        epoch += 1


def get_run_name(config):
    run_name = (
        f"{config.data.name}-{config.model.name}-lr_{config.optimizer.lr}"
    )
    if len(config.wandb.run_name_suffix) > 0:
        run_name = f"{run_name}-{config.wandb.run_name_suffix}"
    return run_name


def setup_logging(config):
    # Set checkpoint dir. Do this first so it saves to wandb config.
    save_ckpt = config.logging.save_ckpt

    if save_ckpt:
        ckpt_subdir = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        ckpt_dir = os.path.join(config.logging.ckpt_dir, ckpt_subdir)
        os.makedirs(ckpt_dir, exist_ok=True)
        config.logging.ckpt_dir = ckpt_dir

    # Setup wandb.
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=get_run_name(config),
        config=OmegaConf.to_container(config),
        mode=config.wandb.mode,
        reinit=True,
    )


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(
    config_path="config", config_name="tiny_imagenet", version_base=None
)
def main(config):
    seed(config.seed)
    setup_logging(config)

    model = load_model(config.model).to(config.device)
    train_loader, val_loader, test_loader = load_data(config.data)

    print(f"Num train batches = {len(train_loader)}")
    print(f"Num val batches = {len(val_loader)}")
    print(f"Num test batches = {len(test_loader)}")

    train(model, train_loader, val_loader, test_loader, config)


if __name__ == "__main__":
    main()
