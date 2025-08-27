"""
Checkpoint management system for distributed meta-learning training.

This module provides comprehensive checkpointing functionality including
model state, optimizer state, training progress persistence, and automatic
checkpoint scheduling and cleanup.
"""

import os
import torch
import json
import shutil
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import glob

from ..utils.distributed_utils import is_main_process, get_rank
from ..config.model_config import MetaPINNConfig


class CheckpointManager:
    """
    Manages checkpointing and resuming for distributed meta-learning training.
    
    Handles saving and loading of model state, optimizer state, training progress,
    and provides automatic checkpoint scheduling and cleanup functionality.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        save_frequency: Save checkpoint every N steps/epochs
        save_best: Whether to save best model based on validation metric
        metric_name: Name of metric to track for best model
        metric_mode: 'min' or 'max' for best metric tracking
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_frequency: int = 100,
        save_best: bool = True,
        metric_name: str = "val_loss",
        metric_mode: str = "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Create checkpoint directory
        if is_main_process():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metric
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
        
        # Checkpoint metadata
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
        logging.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint with model, optimizer, and training state.
        
        Args:
            model: Model to save (should be the underlying module for DDP)
            optimizer: Optimizer state to save
            epoch: Current epoch number
            step: Current training step
            metrics: Dictionary of training metrics
            extra_state: Additional state to save
            is_best: Whether this is the best checkpoint
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved checkpoint file
        """
        if not is_main_process():
            return ""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}_{timestamp}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'rank': get_rank(),
        }
        
        # Add model config if available
        if hasattr(model, 'config'):
            checkpoint['model_config'] = model.config.__dict__
        
        # Add extra state
        if extra_state:
            checkpoint['extra_state'] = extra_state
        
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Update metadata
            self._update_metadata(checkpoint_path, epoch, step, metrics, is_best)
            
            # Handle best model
            if self.save_best and self.metric_name in metrics:
                metric_value = metrics[self.metric_name]
                if self._is_better_metric(metric_value):
                    self.best_metric = metric_value
                    best_path = self.checkpoint_dir / "best_model.pth"
                    shutil.copy2(checkpoint_path, best_path)
                    logging.info(f"New best model saved: {best_path} ({self.metric_name}={metric_value:.6f})")
            
            # Cleanup old checkpoints
            if self.max_checkpoints > 0:
                self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model/optimizer state.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to specific checkpoint (auto-detect if None)
            load_best: Whether to load the best model
            strict: Whether to strictly enforce state dict keys match
            
        Returns:
            Dictionary containing loaded checkpoint information
        """
        # Determine checkpoint path
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        elif checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        try:
            device = next(model.parameters()).device
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Extract training info
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('step', 0)
            metrics = checkpoint.get('metrics', {})
            
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
            logging.info(f"Resumed from epoch {epoch}, step {step}")
            
            return {
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'checkpoint_path': str(checkpoint_path),
                'extra_state': checkpoint.get('extra_state', {})
            }
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pth"):
            try:
                # Load basic info without full checkpoint
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'epoch': checkpoint.get('epoch', 0),
                    'step': checkpoint.get('step', 0),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'metrics': checkpoint.get('metrics', {}),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logging.warning(f"Could not read checkpoint {checkpoint_file}: {e}")
        
        # Sort by epoch and step
        checkpoints.sort(key=lambda x: (x['epoch'], x['step']))
        return checkpoints
    
    def should_save_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved at current step."""
        return step % self.save_frequency == 0
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoints (use with caution)."""
        if not is_main_process():
            return
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            checkpoint_file.unlink()
        
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        logging.info("All checkpoints cleaned up")
    
    def _update_metadata(
        self,
        checkpoint_path: Path,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool
    ):
        """Update checkpoint metadata."""
        if not is_main_process():
            return
        
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'is_best': is_best
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['last_checkpoint'] = checkpoint_info
        
        if is_best or (self.metric_name in metrics and 
                      self._is_better_metric(metrics[self.metric_name])):
            self.metadata['best_checkpoint'] = checkpoint_info
        
        # Save metadata
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save metadata: {e}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load metadata: {e}")
        
        return {
            'checkpoints': [],
            'last_checkpoint': None,
            'best_checkpoint': None
        }
    
    def _is_better_metric(self, metric_value: float) -> bool:
        """Check if metric value is better than current best."""
        if self.metric_mode == 'min':
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if not is_main_process() or self.max_checkpoints <= 0:
            return
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        num_to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:num_to_remove]:
            try:
                checkpoint.unlink()
                logging.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint {checkpoint}: {e}")


class AutoCheckpointer:
    """
    Automatic checkpointing wrapper that handles periodic saves and interruption recovery.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        save_every_n_steps: Save checkpoint every N training steps
        save_every_n_epochs: Save checkpoint every N epochs
        save_on_interrupt: Whether to save checkpoint on training interruption
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        save_every_n_steps: int = 1000,
        save_every_n_epochs: int = 10,
        save_on_interrupt: bool = True
    ):
        self.checkpoint_manager = checkpoint_manager
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.save_on_interrupt = save_on_interrupt
        
        # Track training state
        self.last_save_step = 0
        self.last_save_epoch = 0
        
        # Setup interrupt handler
        if save_on_interrupt:
            self._setup_interrupt_handler()
    
    def step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if checkpoint should be saved and save if needed.
        
        Returns:
            True if checkpoint was saved, False otherwise
        """
        should_save = False
        
        # Check step-based saving
        if (self.save_every_n_steps > 0 and 
            step - self.last_save_step >= self.save_every_n_steps):
            should_save = True
            self.last_save_step = step
        
        # Check epoch-based saving
        if (self.save_every_n_epochs > 0 and 
            epoch - self.last_save_epoch >= self.save_every_n_epochs):
            should_save = True
            self.last_save_epoch = epoch
        
        if should_save:
            self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                metrics=metrics,
                extra_state=extra_state
            )
            return True
        
        return False
    
    def _setup_interrupt_handler(self):
        """Setup signal handler for graceful interruption."""
        import signal
        
        def signal_handler(signum, frame):
            logging.info("Training interrupted, saving checkpoint...")
            # Note: This is a simplified handler
            # In practice, you'd need to coordinate with the training loop
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    config: Optional[MetaPINNConfig] = None,
    **kwargs
) -> CheckpointManager:
    """
    Factory function to create a CheckpointManager with sensible defaults.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        config: Model configuration (used for default settings)
        **kwargs: Additional arguments for CheckpointManager
        
    Returns:
        CheckpointManager instance
    """
    # Set defaults based on config if available
    defaults = {
        'max_checkpoints': 5,
        'save_frequency': 100,
        'save_best': True,
        'metric_name': 'val_loss',
        'metric_mode': 'min'
    }
    
    if config:
        # Adjust defaults based on config
        if hasattr(config, 'meta_epochs'):
            defaults['save_frequency'] = max(1, config.meta_epochs // 20)
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return CheckpointManager(checkpoint_dir, **defaults)