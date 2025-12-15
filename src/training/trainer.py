"""
Training engine for neural network models.

Handles the training loop with:
- Batch processing
- Validation
- Early stopping
- Checkpointing
"""

from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn

from models.base import BaseModel
from data.dataset import ResidualDataset, BatchIterator
from portfolio.weights import normalize_weights, compute_portfolio_returns
from training.loss import compute_loss, compute_stats
from training.optimizer import create_optimizer
from dtypes import Objective, TrainingConfig
from utils.logging import TrainingLogger, get_logger
from utils.device import set_random_seeds

logger = get_logger(__name__)


class Trainer:
    """
    Training engine for statistical arbitrage models.

    Handles the complete training loop including:
    - Forward pass through model
    - Weight normalization
    - Return computation
    - Loss computation (Sharpe or mean-variance)
    - Backpropagation
    - Validation
    """

    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        device: torch.device,
        composition_matrix: Optional[torch.Tensor] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            config: Training configuration
            device: Compute device
            composition_matrix: Optional composition matrix for weight transformation
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.composition_matrix = composition_matrix

        # Set random seed
        set_random_seeds(config.random_seed)

        # Create optimizer if model is trainable
        if model.is_trainable:
            self.optimizer = create_optimizer(
                model.parameters(),
                name=config.optimizer_name,
                lr=config.learning_rate
            )
        else:
            self.optimizer = None

        # Training state
        self.current_epoch = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        self.best_state_dict = None

        # For frictions models
        self.old_weights: Optional[torch.Tensor] = None

        # Logging
        self.training_logger = TrainingLogger()

    def train_epoch(
        self,
        train_dataset: ResidualDataset,
        val_dataset: Optional[ResidualDataset] = None
    ) -> Tuple[float, Optional[float]]:
        """
        Train for one epoch.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset

        Returns:
            Tuple of (train_loss, val_loss)
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        batch_iter = BatchIterator(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Keep temporal order
            drop_last=False
        )

        for batch_idx, batch in enumerate(batch_iter):
            batch_loss = self._train_batch(batch, batch_idx)
            epoch_loss += batch_loss
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        self.train_losses.append(avg_train_loss)

        # Validation
        val_loss = None
        if val_dataset is not None and self.current_epoch % self.config.validation_freq == 0:
            val_loss = self.validate(val_dataset)
            self.val_losses.append(val_loss)

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

        self.current_epoch += 1
        return avg_train_loss, val_loss

    def _train_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> float:
        """
        Train on a single batch.

        Args:
            batch: Batch data dictionary
            batch_idx: Index of current batch

        Returns:
            Batch loss value
        """
        # Move data to device
        features = batch["features"].to(self.device)
        residuals = batch["residuals"].to(self.device)
        valid_mask = batch["valid_mask"].to(self.device)
        comp_mtx = batch.get("composition_matrix")
        if comp_mtx is not None:
            comp_mtx = comp_mtx.to(self.device)

        T, N, _ = features.shape

        # Initialize weights tensor
        all_weights = torch.zeros(T, N, device=self.device)

        # Process each timestep
        for t in range(T):
            # Get valid assets at this timestep
            valid_idx = valid_mask[t]
            if valid_idx.sum() == 0:
                continue

            # Get features for valid assets: (N_valid, feature_dim)
            x = features[t, valid_idx]

            # Forward pass
            if self.model.is_frictions_model:
                old_w = self.old_weights[valid_idx] if self.old_weights is not None else None
                w = self.model(x, old_w)
            else:
                w = self.model(x)

            # Store weights
            all_weights[t, valid_idx] = w

        # Normalize weights
        if comp_mtx is not None:
            norm_weights, _ = normalize_weights(all_weights, comp_mtx)
        else:
            norm_weights, _ = normalize_weights(all_weights)

        # Compute portfolio returns
        returns, _, _ = compute_portfolio_returns(
            norm_weights,
            residuals,
            transaction_cost=0,  # Transaction costs handled in backtest
            holding_cost=0
        )

        # Compute loss
        loss = compute_loss(returns, self.config.objective)

        # Backprop if trainable
        if self.model.is_trainable and self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Update old weights for frictions models
        if self.model.is_frictions_model:
            self.old_weights = norm_weights[-1].detach()

        return loss.item()

    def validate(
        self,
        val_dataset: ResidualDataset
    ) -> float:
        """
        Validate model on validation dataset.

        Args:
            val_dataset: Validation dataset

        Returns:
            Validation loss
        """
        self.model.eval()

        with torch.no_grad():
            # Get all data
            all_features = val_dataset.features.to(self.device)
            all_residuals = val_dataset.residuals[val_dataset.lookback:].to(self.device)
            all_valid_mask = val_dataset.valid_mask.to(self.device)

            T, N, _ = all_features.shape
            all_weights = torch.zeros(T, N, device=self.device)

            for t in range(T):
                valid_idx = all_valid_mask[t]
                if valid_idx.sum() == 0:
                    continue

                x = all_features[t, valid_idx]

                if self.model.is_frictions_model:
                    old_w = self.old_weights[valid_idx] if self.old_weights is not None else None
                    w = self.model(x, old_w)
                else:
                    w = self.model(x)

                all_weights[t, valid_idx] = w

            # Get composition matrix if available
            comp_mtx = None
            if val_dataset.composition_matrix is not None:
                comp_mtx = val_dataset.composition_matrix[val_dataset.lookback:].to(self.device)

            norm_weights, _ = normalize_weights(all_weights, comp_mtx)
            returns, _, _ = compute_portfolio_returns(norm_weights, all_residuals)
            loss = compute_loss(returns, self.config.objective)

        return loss.item()

    def train(
        self,
        train_dataset: ResidualDataset,
        val_dataset: Optional[ResidualDataset] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            num_epochs: Override number of epochs

        Returns:
            Training results dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        self.training_logger.start_training(num_epochs, self.model.__class__.__name__)

        for epoch in range(num_epochs):
            train_loss, val_loss = self.train_epoch(train_dataset, val_dataset)

            # Log progress
            train_sharpe = -train_loss  # Loss is negative Sharpe
            val_sharpe = -val_loss if val_loss is not None else None
            self.training_logger.log_epoch(
                epoch,
                train_loss,
                val_loss,
                train_sharpe,
                val_sharpe
            )

        # Compute final Sharpe
        final_sharpe = -self.train_losses[-1] if self.train_losses else 0.0
        self.training_logger.end_training(final_sharpe)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "final_sharpe": final_sharpe,
            "num_epochs": num_epochs,
        }

    def predict(
        self,
        dataset: ResidualDataset
    ) -> torch.Tensor:
        """
        Generate predictions (portfolio weights) for a dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            Portfolio weights tensor (T, N)
        """
        self.model.eval()

        with torch.no_grad():
            features = dataset.features.to(self.device)
            valid_mask = dataset.valid_mask.to(self.device)

            T, N, _ = features.shape
            all_weights = torch.zeros(T, N, device=self.device)

            old_weights = None

            for t in range(T):
                valid_idx = valid_mask[t]
                if valid_idx.sum() == 0:
                    continue

                x = features[t, valid_idx]

                if self.model.is_frictions_model:
                    old_w = old_weights[valid_idx] if old_weights is not None else None
                    w = self.model(x, old_w)
                else:
                    w = self.model(x)

                all_weights[t, valid_idx] = w

                if self.model.is_frictions_model:
                    norm_w, _ = normalize_weights(all_weights[t:t+1])
                    old_weights = norm_w.squeeze(0)

            # Get composition matrix if available
            comp_mtx = None
            if dataset.composition_matrix is not None:
                comp_mtx = dataset.composition_matrix[dataset.lookback:].to(self.device)

            norm_weights, _ = normalize_weights(all_weights, comp_mtx)

        return norm_weights.cpu()

    def save_checkpoint(self, path: Path) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "current_epoch": self.current_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_state_dict": self.best_state_dict,
            "config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "objective": self.config.objective.value,
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["current_epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_state_dict = checkpoint["best_state_dict"]

        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")
