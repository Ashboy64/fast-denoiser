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

    for dataloader in dataloaders:
        running_loss = 0.0

        for features, targets in tqdm.tqdm(dataloader):
            features = move_features_to_device(features, device)
            targets = move_features_to_device(targets, device)

            running_loss += model.compute_loss(features, targets)

        losses.append(running_loss / len(dataloader))

    model.train()
    return losses


def train(model, train_loader, val_loader, test_loader, config):
    device = config.device
    model.train()

    num_grad_steps = config.training.num_grad_steps
    log_interval = config.logging.log_interval
    eval_interval = config.logging.eval_interval

    save_ckpt = config.logging.save_ckpt
    ckpt_interval = config.logging.ckpt_interval

    if save_ckpt:
        ckpt_subdir = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        ckpt_dir = os.path.join(
            config.logging.ckpt_dir, config.model.name, ckpt_subdir
        )
        os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.lr)

    epoch = 0
    iter_num = 0

    while iter_num < num_grad_steps:
        print(f"Epoch {epoch}")

        for features, targets in train_loader:
            if iter_num == num_grad_steps:
                break

            features = move_features_to_device(features, device)
            targets = move_features_to_device(targets, device)

            loss = model.compute_loss(features, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_log = {}

            if (iter_num + 1) % log_interval == 0:
                to_log["iter"] = iter_num
                to_log["train/train_loss"] = loss
                print(f"{iter_num}: Train loss = {loss}")

            if (iter_num + 1) % eval_interval == 0:
                val_loss, test_loss = eval_model(
                    model, [val_loader, test_loader], device
                )
                print(f"{iter_num}: Val loss = {val_loss}")

                to_log["iter"] = iter_num
                to_log["val/val_loss"] = val_loss
                to_log["test/test_loss"] = test_loss

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
    run_name = f"{config.data.name}-{config.model.name}"
    if len(config.wandb.run_name_suffix) > 0:
        run_name = f"{run_name}-{config.wandb.run_name_suffix}"
    return run_name


def setup_wandb(config):
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=get_run_name(config),
        config=OmegaConf.to_container(config),
        mode=config.wandb.mode,
    )


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path="config", config_name="tiny_imagenet", version_base=None)
def main(config):
    seed(config.seed)
    setup_wandb(config)

    model = load_model(config.model).to(config.device)
    train_loader, val_loader, test_loader = load_data(config.data)

    print(f"Num train batches = {len(train_loader)}")
    print(f"Num val batches = {len(val_loader)}")
    print(f"Num test batches = {len(test_loader)}")

    train(model, train_loader, val_loader, test_loader, config)


if __name__ == "__main__":
    main()
