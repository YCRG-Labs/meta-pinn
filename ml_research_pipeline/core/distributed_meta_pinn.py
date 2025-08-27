"""
Distributed training wrapper for MetaPINN.

This module implements distributed data parallel training for meta-learning
physics-informed neural networks using PyTorch's DistributedDataParallel.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import OrderedDict
import copy

from .meta_pinn import MetaPINN
from ..config.model_config import MetaPINNConfig
from ..utils.distributed_utils import (
    setup_distributed, cleanup_distributed, get_world_size, get_rank,
    is_main_process, barrier, reduce_dict, all_reduce
)


class DistributedMetaPINN(nn.Module):
    """
    Distributed wrapper for MetaPINN using DistributedDataParallel.
    
    Handles multi-GPU task batch processing and gradient synchronization
    for meta-learning across multiple devices.
    
    Args:
        config: MetaPINNConfig containing model architecture and training parameters
        device_ids: List of GPU device IDs to use (None for auto-detection)
        find_unused_parameters: Whether to find unused parameters in DDP
    """
    
    def __init__(
        self,
        config: MetaPINNConfig,
        device_ids: Optional[List[int]] = None,
        find_unused_parameters: bool = True
    ):
        super(DistributedMetaPINN, self).__init__()
        
        self.config = config
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.is_distributed = self.world_size > 1
        
        # Initialize base MetaPINN
        self.meta_pinn = MetaPINN(config)
        
        # Setup distributed training if available
        if self.is_distributed:
            self._setup_distributed_model(device_ids, find_unused_parameters)
        else:
            self.model = self.meta_pinn
            logging.info("Running in single-process mode")
    
    def _setup_distributed_model(
        self,
        device_ids: Optional[List[int]],
        find_unused_parameters: bool
    ):
        """Setup DistributedDataParallel wrapper."""
        # Determine device
        if torch.cuda.is_available():
            if device_ids is None:
                device_id = self.rank % torch.cuda.device_count()
                device_ids = [device_id]
            else:
                device_id = device_ids[0]
            
            self.device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)
        else:
            self.device = torch.device('cpu')
            device_ids = None
        
        # Move model to device
        self.meta_pinn = self.meta_pinn.to(self.device)
        
        # Wrap with DistributedDataParallel
        self.model = DDP(
            self.meta_pinn,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters
        )
        
        logging.info(f"Distributed training setup complete on rank {self.rank}")
    
    @property
    def module(self) -> MetaPINN:
        """Access the underlying MetaPINN module."""
        if self.is_distributed:
            return self.model.module
        return self.model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)
    
    def adapt_to_task(
        self,
        task: Dict[str, Any],
        adaptation_steps: Optional[int] = None
    ) -> OrderedDict:
        """
        Adapt model parameters to a specific task using gradient descent.
        
        Args:
            task: Task dictionary containing support set data
            adaptation_steps: Number of adaptation steps (uses config default if None)
            
        Returns:
            Adapted parameters as OrderedDict
        """
        if adaptation_steps is None:
            adaptation_steps = self.config.adaptation_steps
        
        # Use the underlying module for adaptation
        return self.module.adapt_to_task(task, adaptation_steps)
    
    def meta_update(
        self,
        task_batch: List[Dict[str, Any]],
        meta_optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform meta-learning update across a batch of tasks.
        
        Args:
            task_batch: List of tasks for meta-learning
            meta_optimizer: Meta-optimizer for parameter updates
            
        Returns:
            Dictionary of training metrics
        """
        meta_optimizer.zero_grad()
        
        # Distribute tasks across processes
        local_tasks = self._distribute_tasks(task_batch)
        
        # Compute meta-gradients on local tasks
        local_metrics = self._compute_local_meta_gradients(local_tasks)
        
        # Synchronize gradients across processes
        if self.is_distributed:
            self._synchronize_meta_gradients()
        
        # Update meta-parameters
        meta_optimizer.step()
        
        # Aggregate metrics across processes
        metrics = self._aggregate_metrics(local_metrics)
        
        return metrics
    
    def _distribute_tasks(self, task_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute tasks across processes for parallel processing."""
        if not self.is_distributed:
            return task_batch
        
        # Simple round-robin distribution
        local_tasks = []
        for i, task in enumerate(task_batch):
            if i % self.world_size == self.rank:
                local_tasks.append(task)
        
        return local_tasks
    
    def _compute_local_meta_gradients(
        self,
        local_tasks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute meta-gradients on local subset of tasks."""
        if not local_tasks:
            return {'meta_loss': 0.0, 'num_tasks': 0}
        
        total_meta_loss = 0.0
        num_tasks = len(local_tasks)
        
        for task in local_tasks:
            # Adapt to task
            adapted_params = self.module.adapt_to_task(task)
            
            # Compute meta-loss on query set
            query_loss = self._compute_query_loss(task, adapted_params)
            
            # Accumulate meta-gradients
            meta_gradients = torch.autograd.grad(
                query_loss,
                self.module.parameters(),
                create_graph=False,
                retain_graph=False
            )
            
            # Add gradients to model parameters
            for param, grad in zip(self.module.parameters(), meta_gradients):
                if param.grad is None:
                    param.grad = grad / num_tasks
                else:
                    param.grad += grad / num_tasks
            
            total_meta_loss += query_loss.item()
        
        return {
            'meta_loss': total_meta_loss / num_tasks,
            'num_tasks': num_tasks
        }
    
    def _compute_query_loss(
        self,
        task: Dict[str, Any],
        adapted_params: OrderedDict
    ) -> torch.Tensor:
        """Compute loss on query set using adapted parameters."""
        # Get query set
        query_coords = task['query_set']['coords']
        query_data = task['query_set']['data']
        
        # Compute predictions using functional forward
        predictions = self.module.forward(query_coords, adapted_params)
        
        # Compute data loss
        data_loss = F.mse_loss(predictions, query_data)
        
        # Compute physics loss
        physics_losses = self.module.physics_loss(
            query_coords, task['config'], adapted_params
        )
        
        # Combined loss
        total_loss = data_loss + self.config.physics_loss_weight * physics_losses['total']
        
        return total_loss
    
    def _synchronize_meta_gradients(self):
        """Synchronize meta-gradients across all processes."""
        if not self.is_distributed:
            return
        
        # Average gradients across processes
        for param in self.module.parameters():
            if param.grad is not None:
                all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size
    
    def _aggregate_metrics(self, local_metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate training metrics across all processes."""
        if not self.is_distributed:
            return local_metrics
        
        # Convert to tensors for reduction
        metrics_tensor = torch.tensor([
            local_metrics['meta_loss'],
            local_metrics['num_tasks']
        ], device=self.device if hasattr(self, 'device') else torch.device('cpu'))
        
        # Reduce across processes
        all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Convert back to dictionary
        total_loss = metrics_tensor[0].item()
        total_tasks = int(metrics_tensor[1].item())
        
        return {
            'meta_loss': total_loss / max(total_tasks, 1),
            'num_tasks': total_tasks
        }
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """Save model checkpoint (only on main process)."""
        if is_main_process():
            checkpoint = {
                'model_state_dict': self.module.state_dict(),
                'config': self.config,
                **kwargs
            }
            torch.save(checkpoint, filepath)
            logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device if hasattr(self, 'device') else 'cpu')
        self.module.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self


class DistributedTrainingManager:
    """
    Manager class for distributed meta-learning training.
    
    Handles process group initialization, cleanup, and coordination
    of distributed training across multiple GPUs/nodes.
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
        timeout_minutes: int = 30
    ):
        """
        Initialize distributed training manager.
        
        Args:
            backend: Distributed backend (nccl, gloo, mpi)
            init_method: Process group initialization method
            timeout_minutes: Timeout for distributed operations
        """
        self.backend = backend
        self.init_method = init_method
        import datetime
        self.timeout = torch.distributed.default_pg_timeout
        if timeout_minutes > 0:
            self.timeout = datetime.timedelta(minutes=timeout_minutes)
        
        self.is_initialized = False
    
    def __enter__(self):
        """Context manager entry - setup distributed training."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup distributed training."""
        self.cleanup()
    
    def setup(self) -> bool:
        """Setup distributed training environment."""
        try:
            self.is_initialized = setup_distributed(
                backend=self.backend,
                init_method=self.init_method
            )
            
            if self.is_initialized:
                # Set timeout for distributed operations
                if hasattr(dist, 'default_pg_timeout'):
                    dist.default_pg_timeout = self.timeout
                
                logging.info(f"Distributed training initialized successfully")
                logging.info(f"World size: {get_world_size()}, Rank: {get_rank()}")
            
            return self.is_initialized
            
        except Exception as e:
            logging.error(f"Failed to initialize distributed training: {e}")
            return False
    
    def cleanup(self):
        """Cleanup distributed training environment."""
        if self.is_initialized:
            try:
                cleanup_distributed()
                logging.info("Distributed training cleanup completed")
            except Exception as e:
                logging.error(f"Error during distributed cleanup: {e}")
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            barrier()
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        return is_main_process()
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return get_world_size()
    
    def get_rank(self) -> int:
        """Get rank of current process."""
        return get_rank()


def create_distributed_meta_pinn(
    config: MetaPINNConfig,
    device_ids: Optional[List[int]] = None,
    find_unused_parameters: bool = True
) -> DistributedMetaPINN:
    """
    Factory function to create a distributed MetaPINN.
    
    Args:
        config: MetaPINN configuration
        device_ids: GPU device IDs to use
        find_unused_parameters: Whether to find unused parameters in DDP
        
    Returns:
        DistributedMetaPINN instance
    """
    return DistributedMetaPINN(
        config=config,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters
    )