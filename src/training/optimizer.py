"""
Optimizer utilities for training.
"""

from typing import Dict, Any, Iterator
import torch
import torch.optim as optim


def create_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    name: str = "Adam",
    lr: float = 0.001,
    **kwargs
) -> optim.Optimizer:
    """
    Create an optimizer instance.

    Args:
        parameters: Model parameters to optimize
        name: Optimizer name ('Adam', 'SGD', 'AdamW', 'RMSprop')
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance

    Raises:
        ValueError: If optimizer name is not recognized
    """
    name_lower = name.lower()

    if name_lower == "adam":
        return optim.Adam(
            parameters,
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0)
        )

    elif name_lower == "adamw":
        return optim.AdamW(
            parameters,
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.01)
        )

    elif name_lower == "sgd":
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=kwargs.get("momentum", 0),
            weight_decay=kwargs.get("weight_decay", 0),
            nesterov=kwargs.get("nesterov", False)
        )

    elif name_lower == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=lr,
            alpha=kwargs.get("alpha", 0.99),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0),
            momentum=kwargs.get("momentum", 0)
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "none",
    **kwargs
) -> optim.lr_scheduler.LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_name: Scheduler type
        **kwargs: Scheduler-specific arguments

    Returns:
        Learning rate scheduler
    """
    name_lower = scheduler_name.lower()

    if name_lower == "none" or name_lower == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    elif name_lower == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1)
        )

    elif name_lower == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 0)
        )

    elif name_lower == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10)
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
