from typing import Optional, Tuple
import os
import time

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    loss_actor_fn: Optional[nn.Module] = None,
    loss_critic_fn: Optional[nn.Module] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    model.train()
    model.to(device)

    if loss_critic_fn is None:
        loss_critic_fn = nn.MSELoss()

    total_loss_acc = 0.0
    actor_loss_acc = 0.0
    critic_loss_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        X, y_actor, y_critic = batch
        X = X.to(device)
        y_critic = y_critic.to(device)

        # detect actor target type and move to device if needed
        if isinstance(y_actor, torch.Tensor):
            y_actor_tensor = y_actor.to(device)
        else:
            # if numpy array, convert
            y_actor_tensor = torch.tensor(y_actor, device=device)

        optimizer.zero_grad()

        pred_actor, pred_critic = model(X)

        if loss_actor_fn is None:
            if (
                y_actor_tensor.dtype.is_floating_point
                and y_actor_tensor.dim() == pred_actor.dim()
                and y_actor_tensor.shape[1] == pred_actor.shape[1]
            ):
                log_probs = F.log_softmax(pred_actor, dim=1)
                loss_actor = F.kl_div(log_probs, y_actor_tensor, reduction="batchmean")
            else:
                loss_actor = F.cross_entropy(
                    pred_actor, y_actor_tensor.long().squeeze()
                )
        else:
            loss_actor = loss_actor_fn(pred_actor, y_actor_tensor)

        loss_critic = loss_critic_fn(pred_critic, y_critic)

        loss = loss_actor + loss_critic
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss_acc += loss.item()
        actor_loss_acc += loss_actor.item()
        critic_loss_acc += loss_critic.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss_acc / n_batches,
        actor_loss_acc / n_batches,
        critic_loss_acc / n_batches,
    )


def create_run_dir(base_dir: str = "models", name: Optional[str] = None) -> str:
    """Create a timestamped run directory under base_dir and return its path.

    If name is provided, it will be included in the folder name.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{ts}" if name is None else f"{ts}__{name}"
    run_dir = os.path.join(base_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(run_dir: str, config: dict):
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def save_losses_npz(run_dir: str, losses_actor: list, losses_critic: list):
    path = os.path.join(run_dir, "losses.npz")
    np.savez(
        path, losses_actor=np.array(losses_actor), losses_critic=np.array(losses_critic)
    )


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    epochs: int = 10,
    loss_actor_fn: Optional[nn.Module] = None,
    loss_critic_fn: Optional[nn.Module] = None,
    grad_clip: Optional[float] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_dir: Optional[str] = None,
):
    """High-level training loop without validation.

    Saves checkpoints if checkpoint_dir is provided. If no checkpoint_dir is passed, a
    timestamped folder will be created under `models/` and used to store checkpoints,
    a `config.json` and a `losses.npz` file with numeric loss history.
    """
    created_run_dir = False
    if checkpoint_dir is None:
        checkpoint_dir = create_run_dir("models")
        created_run_dir = True

    losses_actor_history = []
    losses_critic_history = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_total, train_actor, train_critic = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_actor_fn,
            loss_critic_fn,
            grad_clip,
        )
        elapsed = time.time() - start

        if scheduler is not None:
            scheduler.step()

        # store epoch metrics
        losses_actor_history.append(train_actor)
        losses_critic_history.append(train_critic)

        # log training metrics only (no validation loop)
        logger.info(
            f"Epoch {epoch:03d} | train_total={train_total:.4f} actor={train_actor:.4f} critic={train_critic:.4f} | {elapsed:.1f}s"
        )

        if checkpoint_dir is not None:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                path,
            )

    # After training, save losses and a minimal config if we created the run folder
    try:
        if checkpoint_dir is not None:
            # save numeric losses
            save_losses_npz(checkpoint_dir, losses_actor_history, losses_critic_history)

            # save a minimal config
            config = {
                "timestamp": datetime.now().isoformat(),
                "epochs": epochs,
                "device": str(device),
                "model_class": model.__class__.__name__,
            }
            save_config(checkpoint_dir, config)

            # also save a PNG of losses for convenience
            try:
                fig, ax = plt.subplots(1, 1)
                ax.plot(losses_actor_history, label="actor")
                ax.plot(losses_critic_history, label="critic")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                fig_path = os.path.join(
                    checkpoint_dir, f"losses_{model.__class__.__name__}.png"
                )
                fig.savefig(fig_path)
                plt.close(fig)
            except Exception:
                # don't fail training just because plotting failed
                logger.exception("Failed to save loss plot")
    except Exception:
        logger.exception("Failed to save run artifacts")

    logger.info("Training finished. Artifacts saved to: %s", checkpoint_dir)


# small utility to sanity check shapes
def sanity_check_forward(
    model: nn.Module, board_size: int = 6, in_channels: int = 5, batch: int = 2
):
    model.eval()
    X = torch.randn(batch, in_channels, board_size, board_size)
    with torch.no_grad():
        a, c = model(X)
    assert (
        a.shape[0] == batch and a.shape[1] == board_size * board_size
    ), f"actor shape mismatch: {a.shape}"
    assert c.shape == (batch, 1) or (
        c.shape[0] == batch and c.shape[1] == 1
    ), f"critic shape mismatch: {c.shape}"
    logger.info("sanity_check_forward OK %s %s", a.shape, c.shape)
