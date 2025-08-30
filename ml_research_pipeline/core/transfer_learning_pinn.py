"""
Transfer Learning Physics-Informed Neural Network (TransferLearningPINN) implementation.

This module implements a transfer learning baseline that pre-trains on multiple tasks
and then fine-tunes on new tasks, serving as a comparison baseline for meta-learning.
"""

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig
from .standard_pinn import StandardPINN


class TransferLearningPINN(StandardPINN):
    """
    Transfer Learning Physics-Informed Neural Network.

    This class implements a transfer learning baseline that:
    1. Pre-trains on multiple tasks to learn general fluid dynamics representations
    2. Fine-tunes on new tasks with reduced learning rates
    3. Serves as a comparison baseline for meta-learning approaches

    Args:
        config: ModelConfig containing model architecture and training parameters
    """

    def __init__(self, config: ModelConfig):
        super(TransferLearningPINN, self).__init__(config)

        # Transfer learning specific parameters
        self.pretrain_history = []
        self.finetune_history = []
        self.is_pretrained = False
        self.pretrain_tasks_seen = 0

        # Store initial parameters for reset functionality
        self.initial_state = None
        self._save_initial_state()

    def _save_initial_state(self):
        """Save the initial model state for reset functionality."""
        self.initial_state = {
            name: param.clone().detach() for name, param in self.named_parameters()
        }

    def pretrain(
        self,
        tasks: List[Dict[str, Any]],
        epochs_per_task: int = 500,
        learning_rate: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Pre-train the model on multiple tasks to learn general representations.

        Args:
            tasks: List of tasks for pre-training, each containing 'coords', 'data', and 'task_info'
            epochs_per_task: Number of epochs to train on each task
            learning_rate: Learning rate for pre-training
            verbose: Whether to print training progress

        Returns:
            Dict[str, List[float]]: Pre-training history aggregated across all tasks
        """
        if verbose:
            print(f"Starting pre-training on {len(tasks)} tasks...")

        # Set up optimizer for pre-training
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler for entire pre-training
        total_steps = len(tasks) * epochs_per_task
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        # Pre-training history
        history = {
            "data_loss": [],
            "physics_loss": [],
            "total_loss": [],
            "task_losses": [],  # Loss per task
            "learning_rates": [],
        }

        self.train()
        step = 0

        # Train on each task sequentially
        for task_idx, task in enumerate(tasks):
            coords = task["coords"]
            data = task["data"]
            task_info = task["task_info"]

            if verbose:
                print(
                    f"Pre-training on task {task_idx + 1}/{len(tasks)}: "
                    f"{task_info.get('viscosity_type', 'unknown')} viscosity"
                )

            task_losses = []

            # Train on current task
            for epoch in range(epochs_per_task):
                optimizer.zero_grad()

                # Forward pass
                predictions = self.forward(coords)

                # Data loss
                data_loss = F.mse_loss(predictions, data)

                # Physics loss
                physics_losses = self.physics_loss(coords, task_info)
                physics_weight = self.compute_adaptive_physics_weight(
                    physics_losses, self.physics_loss_weight
                )
                physics_loss = physics_weight * physics_losses["total"]

                # Total loss
                total_loss = data_loss + physics_loss

                # Check for numerical issues
                if not torch.isfinite(total_loss):
                    if verbose:
                        print(f"Warning: Non-finite loss detected at epoch {epoch}, skipping update")
                    continue

                # Backward pass
                total_loss.backward()

                # Gradient clipping for numerical stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                # Store history
                history["data_loss"].append(data_loss.item())
                history["physics_loss"].append(physics_loss.item())
                history["total_loss"].append(total_loss.item())
                history["learning_rates"].append(scheduler.get_last_lr()[0])

                task_losses.append(total_loss.item())
                step += 1

                # Print progress
                if verbose and (epoch + 1) % 100 == 0:
                    print(
                        f"  Epoch {epoch + 1}/{epochs_per_task}: "
                        f"Loss: {total_loss.item():.6f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

            # Store task-specific loss
            history["task_losses"].append(
                {
                    "task_idx": task_idx,
                    "task_type": task_info.get("viscosity_type", "unknown"),
                    "final_loss": task_losses[-1],
                    "avg_loss": np.mean(task_losses),
                    "loss_reduction": task_losses[0] - task_losses[-1],
                }
            )

        # Mark as pre-trained
        self.is_pretrained = True
        self.pretrain_tasks_seen = len(tasks)
        self.pretrain_history = history

        if verbose:
            avg_final_loss = np.mean(
                [task["final_loss"] for task in history["task_losses"]]
            )
            print(f"Pre-training completed. Average final loss: {avg_final_loss:.6f}")

        return history

    def finetune(
        self,
        task: Dict[str, Any],
        epochs: int = 100,
        learning_rate: float = 0.0001,
        freeze_layers: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the pre-trained model on a new task.

        Args:
            task: Dictionary containing task information with 'coords', 'data', and 'task_info'
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning (typically lower than pre-training)
            freeze_layers: List of layer indices to freeze during fine-tuning (None = no freezing)
            verbose: Whether to print training progress

        Returns:
            Dict[str, List[float]]: Fine-tuning history
        """
        if not self.is_pretrained:
            raise ValueError(
                "Model must be pre-trained before fine-tuning. Call pretrain() first."
            )

        if verbose:
            task_type = task["task_info"].get("viscosity_type", "unknown")
            print(f"Fine-tuning on {task_type} viscosity task...")

        # Extract task data
        coords = task["coords"]
        data = task["data"]
        task_info = task["task_info"]

        # Store current task info
        self.current_task_info = task_info

        # Freeze specified layers if requested
        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)

        # Set up optimizer for fine-tuning (only optimize unfrozen parameters)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params, lr=learning_rate, weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler for fine-tuning
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Fine-tuning history
        history = {
            "data_loss": [],
            "physics_loss": [],
            "total_loss": [],
            "momentum_x_loss": [],
            "momentum_y_loss": [],
            "continuity_loss": [],
            "learning_rates": [],
        }

        # Fine-tuning loop
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            predictions = self.forward(coords)

            # Data loss
            data_loss = F.mse_loss(predictions, data)

            # Physics loss
            physics_losses = self.physics_loss(coords, task_info)
            physics_weight = self.compute_adaptive_physics_weight(
                physics_losses, self.physics_loss_weight
            )
            physics_loss = physics_weight * physics_losses["total"]

            # Total loss
            total_loss = data_loss + physics_loss

            # Check for numerical issues
            if not torch.isfinite(total_loss):
                if verbose:
                    print(f"Warning: Non-finite loss detected at epoch {epoch}, skipping update")
                # Still record the loss for debugging
                history["data_loss"].append(float('nan'))
                history["physics_loss"].append(float('nan'))
                history["total_loss"].append(float('nan'))
                history["momentum_x_loss"].append(float('nan'))
                history["momentum_y_loss"].append(float('nan'))
                history["continuity_loss"].append(float('nan'))
                history["learning_rates"].append(scheduler.get_last_lr()[0])
                continue

            # Backward pass
            total_loss.backward()

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            scheduler.step()

            # Store history
            history["data_loss"].append(data_loss.item())
            history["physics_loss"].append(physics_loss.item())
            history["total_loss"].append(total_loss.item())
            history["momentum_x_loss"].append(physics_losses["momentum_x"].item())
            history["momentum_y_loss"].append(physics_losses["momentum_y"].item())
            history["continuity_loss"].append(physics_losses["continuity"].item())
            history["learning_rates"].append(scheduler.get_last_lr()[0])

            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Data Loss: {data_loss.item():.6f}, "
                    f"Physics Loss: {physics_loss.item():.6f}, "
                    f"Total Loss: {total_loss.item():.6f}"
                )

        # Unfreeze all layers after fine-tuning
        if freeze_layers is not None:
            self._unfreeze_all_layers()

        # Store fine-tuning history
        self.finetune_history = history

        if verbose:
            initial_loss = history["total_loss"][0]
            final_loss = history["total_loss"][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"Fine-tuning completed. Loss improvement: {improvement:.2f}%")

        return history

    def _freeze_layers(self, layer_indices: List[int]):
        """
        Freeze specified layers during fine-tuning.

        Args:
            layer_indices: List of layer indices to freeze
        """
        linear_layer_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if linear_layer_idx in layer_indices:
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
                linear_layer_idx += 1

    def _unfreeze_all_layers(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True

    def pretrain_then_finetune(
        self,
        pretrain_tasks: List[Dict[str, Any]],
        finetune_task: Dict[str, Any],
        pretrain_epochs_per_task: int = 500,
        finetune_epochs: int = 100,
        pretrain_lr: float = 0.001,
        finetune_lr: float = 0.0001,
        freeze_layers: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Complete transfer learning pipeline: pre-train then fine-tune.

        Args:
            pretrain_tasks: List of tasks for pre-training
            finetune_task: Task for fine-tuning
            pretrain_epochs_per_task: Epochs per task during pre-training
            finetune_epochs: Epochs for fine-tuning
            pretrain_lr: Learning rate for pre-training
            finetune_lr: Learning rate for fine-tuning
            freeze_layers: Layers to freeze during fine-tuning
            verbose: Whether to print progress

        Returns:
            Tuple[Dict, Dict]: Pre-training and fine-tuning histories
        """
        # Pre-training phase
        pretrain_history = self.pretrain(
            tasks=pretrain_tasks,
            epochs_per_task=pretrain_epochs_per_task,
            learning_rate=pretrain_lr,
            verbose=verbose,
        )

        # Fine-tuning phase
        finetune_history = self.finetune(
            task=finetune_task,
            epochs=finetune_epochs,
            learning_rate=finetune_lr,
            freeze_layers=freeze_layers,
            verbose=verbose,
        )

        return pretrain_history, finetune_history

    def evaluate_transfer_performance(
        self,
        test_tasks: List[Dict[str, Any]],
        finetune_epochs: int = 50,
        finetune_lr: float = 0.0001,
    ) -> Dict[str, Any]:
        """
        Evaluate transfer learning performance on multiple test tasks.

        Args:
            test_tasks: List of test tasks
            finetune_epochs: Epochs for fine-tuning on each test task
            finetune_lr: Learning rate for fine-tuning

        Returns:
            Dict[str, Any]: Transfer learning evaluation results
        """
        if not self.is_pretrained:
            raise ValueError("Model must be pre-trained before evaluation.")

        results = {
            "task_results": [],
            "avg_final_loss": 0.0,
            "avg_improvement": 0.0,
            "convergence_speeds": [],
            "transfer_effectiveness": [],
        }

        # Store original state to restore after each task
        original_state = {
            name: param.clone() for name, param in self.named_parameters()
        }

        for task_idx, task in enumerate(test_tasks):
            # Restore pre-trained state
            for name, param in self.named_parameters():
                param.data.copy_(original_state[name])

            # Fine-tune on current task
            history = self.finetune(
                task, epochs=finetune_epochs, learning_rate=finetune_lr, verbose=False
            )

            # Evaluate final performance
            final_metrics = self.evaluate_on_task(task)

            # Calculate convergence speed (epochs to reach 90% of final improvement)
            initial_loss = history["total_loss"][0]
            final_loss = history["total_loss"][-1]
            target_loss = initial_loss - 0.9 * (initial_loss - final_loss)

            convergence_epoch = finetune_epochs
            for epoch, loss in enumerate(history["total_loss"]):
                if loss <= target_loss:
                    convergence_epoch = epoch + 1
                    break

            # Calculate transfer effectiveness (improvement vs random initialization)
            # This would require comparison with random initialization baseline
            transfer_effectiveness = (initial_loss - final_loss) / initial_loss

            task_result = {
                "task_idx": task_idx,
                "task_type": task["task_info"].get("viscosity_type", "unknown"),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "improvement": (initial_loss - final_loss) / initial_loss,
                "convergence_epochs": convergence_epoch,
                "transfer_effectiveness": transfer_effectiveness,
                "final_metrics": final_metrics,
            }

            results["task_results"].append(task_result)
            results["convergence_speeds"].append(convergence_epoch)
            results["transfer_effectiveness"].append(transfer_effectiveness)

        # Calculate aggregate statistics
        results["avg_final_loss"] = np.mean(
            [r["final_loss"] for r in results["task_results"]]
        )
        results["avg_improvement"] = np.mean(
            [r["improvement"] for r in results["task_results"]]
        )
        results["avg_convergence_speed"] = np.mean(results["convergence_speeds"])
        results["avg_transfer_effectiveness"] = np.mean(
            results["transfer_effectiveness"]
        )

        # Restore original state
        for name, param in self.named_parameters():
            param.data.copy_(original_state[name])

        return results

    def compare_with_scratch_training(
        self,
        task: Dict[str, Any],
        scratch_epochs: int = 1000,
        finetune_epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Compare transfer learning performance with training from scratch.

        Args:
            task: Task to evaluate on
            scratch_epochs: Epochs for training from scratch
            finetune_epochs: Epochs for fine-tuning
            learning_rate: Learning rate for both approaches

        Returns:
            Dict[str, Any]: Comparison results
        """
        if not self.is_pretrained:
            raise ValueError("Model must be pre-trained before comparison.")

        # Store current pre-trained state
        pretrained_state = {
            name: param.clone() for name, param in self.named_parameters()
        }

        # 1. Fine-tuning approach
        finetune_history = self.finetune(
            task,
            epochs=finetune_epochs,
            learning_rate=learning_rate * 0.1,
            verbose=False,
        )
        finetune_metrics = self.evaluate_on_task(task)

        # 2. Training from scratch approach
        # Reset to initial random state
        for name, param in self.named_parameters():
            param.data.copy_(self.initial_state[name])

        scratch_history = self.train_on_task(
            task, epochs=scratch_epochs, learning_rate=learning_rate, verbose=False
        )
        scratch_metrics = self.evaluate_on_task(task)

        # Restore pre-trained state
        for name, param in self.named_parameters():
            param.data.copy_(pretrained_state[name])

        # Calculate comparison metrics
        comparison = {
            "finetune_final_loss": finetune_history["total_loss"][-1],
            "scratch_final_loss": scratch_history["total_loss"][-1],
            "finetune_epochs": finetune_epochs,
            "scratch_epochs": scratch_epochs,
            "finetune_metrics": finetune_metrics,
            "scratch_metrics": scratch_metrics,
            "transfer_advantage": scratch_history["total_loss"][-1]
            - finetune_history["total_loss"][-1],
            "sample_efficiency": scratch_epochs / finetune_epochs,
            "performance_ratio": finetune_history["total_loss"][-1]
            / scratch_history["total_loss"][-1],
        }

        return comparison

    def get_pretrain_history(self) -> Dict[str, List[float]]:
        """
        Get the pre-training history.

        Returns:
            Dict[str, List[float]]: Pre-training history
        """
        return self.pretrain_history

    def get_finetune_history(self) -> Dict[str, List[float]]:
        """
        Get the fine-tuning history.

        Returns:
            Dict[str, List[float]]: Fine-tuning history
        """
        return self.finetune_history

    def reset_to_pretrained(self):
        """Reset model to pre-trained state (before any fine-tuning)."""
        if not self.is_pretrained:
            raise ValueError("No pre-trained state available.")

        # This would require storing the pre-trained state
        # For now, we'll just clear fine-tuning history
        self.finetune_history = []
        self.current_task_info = None

    def reset_completely(self):
        """Reset model to initial random state."""
        for name, param in self.named_parameters():
            param.data.copy_(self.initial_state[name])

        self.pretrain_history = []
        self.finetune_history = []
        self.is_pretrained = False
        self.pretrain_tasks_seen = 0
        self.current_task_info = None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the transfer learning model.

        Returns:
            Dict[str, Any]: Model information
        """
        base_info = super().get_model_info()
        base_info.update(
            {
                "model_type": "TransferLearningPINN",
                "is_pretrained": self.is_pretrained,
                "pretrain_tasks_seen": self.pretrain_tasks_seen,
                "has_finetune_history": len(self.finetune_history) > 0,
            }
        )
        return base_info
