"""
Distributed training utilities.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Any
import logging


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> bool:
    """Setup distributed training.
    
    Args:
        backend: Distributed backend (nccl, gloo, mpi)
        init_method: Initialization method
        world_size: Total number of processes
        rank: Rank of current process
        
    Returns:
        True if distributed training is initialized
    """
    if not dist.is_available():
        logging.warning("Distributed training not available")
        return False
    
    # Get world size and rank from environment if not provided
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    
    # Only initialize if world size > 1
    if world_size <= 1:
        return False
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
    
    # Set device for current process
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    logging.info(f"Distributed training initialized: rank {rank}/{world_size}")
    return True


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_size() -> int:
    """Get world size for distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank for distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is main process."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce operation across processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> list:
    """All-gather operation across processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary of tensors across processes.
    
    Args:
        input_dict: Dictionary of tensors
        average: Whether to average the results
        
    Returns:
        Reduced dictionary
    """
    if not (dist.is_available() and dist.is_initialized()):
        return input_dict
    
    world_size = get_world_size()
    
    with torch.no_grad():
        names = []
        values = []
        
        # Sort the keys for consistent ordering across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


class DistributedSampler:
    """Simple distributed sampler for datasets."""
    
    def __init__(self, dataset_size: int, shuffle: bool = True):
        """Initialize distributed sampler.
        
        Args:
            dataset_size: Size of the dataset
            shuffle: Whether to shuffle the data
        """
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.world_size = get_world_size()
        self.rank = get_rank()
        
        # Calculate samples per process
        self.num_samples = int(dataset_size / self.world_size)
        self.total_size = self.num_samples * self.world_size
    
    def __iter__(self):
        """Generate indices for current process."""
        if self.shuffle:
            # Generate random permutation
            g = torch.Generator()
            g.manual_seed(0)  # Same seed for all processes
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # Subsample for current process
        indices = indices[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        """Get number of samples for current process."""
        return self.num_samples