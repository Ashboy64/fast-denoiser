import os
import tqdm
import hydra
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn

from data import load_data, move_features_to_device
from models import load_model


@torch.no_grad()
def eval_model(model, dataloaders, device):
    print(f"EVALUATING MODEL")
    model.eval()

    loss_fn = nn.MSELoss()
    losses = []

    for dataloader in dataloaders:
        running_loss = 0.0

        for features, targets in tqdm.tqdm(dataloader):
            features = move_features_to_device(features, device)
            targets = targets.to(device)

            outputs = model(features)
            running_loss += loss_fn(outputs, targets)

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

    train_iterator = iter(train_loader)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.lr)

    for iter_num in tqdm.trange(num_grad_steps):
        features, targets = next(train_iterator)
        features = move_features_to_device(features, device)
        targets = targets.to(device)

        outputs = model(features)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_num + 1) % log_interval == 0:
            print(f"{iter_num}: Train loss = {loss}")

        if (iter_num + 1) % eval_interval == 0:
            val_loss, test_loss = eval_model(
                model, [val_loader, test_loader], device
            )
            print(f"{iter_num}: Val loss = {val_loss}")

        if save_ckpt and (iter_num + 1) % ckpt_interval == 0:
            torch.save(model, os.path.join(ckpt_dir, f"iter_{iter_num}.pt"))


@hydra.main(config_path="config", config_name="baseline", version_base=None)
def main(config):
    model = load_model(config.model).to(config.device)
    train_loader, val_loader, test_loader = load_data(config.data)

    print(f"Num train batches = {len(train_loader)}")
    print(f"Num val batches = {len(val_loader)}")
    print(f"Num test batches = {len(test_loader)}")

    train(model, train_loader, val_loader, test_loader, config)


if __name__ == "__main__":
    main()
