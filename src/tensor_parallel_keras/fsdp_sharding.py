"""
FSDP-Style Parameter Sharding for Keras

This module implements Fully Sharded Data Parallel (FSDP) style parameter sharding
where the model structure stays intact on all devices, but individual parameter
tensors are sharded across devices to reduce memory footprint.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from keras import Model, layers
from .distributed_backend import create_distributed_backend

logger = logging.getLogger(__name__)

class FSDPParameterShard:
    """Represents a shard of a parameter tensor across devices."""
    
    def __init__(self, parameter_name: str, full_shape: tuple, shard_shape: tuple, 
                 device_rank: int, world_size: int, shard_indices: tuple):
        self.parameter_name = parameter_name
        self.full_shape = full_shape
        self.shard_shape = shard_shape
        self.device_rank = device_rank
        self.world_size = world_size
        self.shard_indices = shard_indices  # (start_idx, end_idx) for the dimension being sharded
        
        # Initialize the shard with zeros
        self.shard_tensor = torch.zeros(shard_shape, dtype=torch.float32, requires_grad=True)
        
    def get_shard_tensor(self) -> torch.Tensor:
        """Get the shard tensor for this device."""
        return self.shard_tensor
        
    def update_shard(self, new_shard: torch.Tensor):
        """Update the shard with new values."""
        if new_shard.shape == self.shard_shape:
            self.shard_tensor.data = new_shard.data
        else:
            logger.warning(f"Shape mismatch: expected {self.shard_shape}, got {new_shard.shape}")
            
    def get_full_parameter(self, all_shards: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct the full parameter from all shards."""
        if len(all_shards) != self.world_size:
            logger.error(f"Expected {self.world_size} shards, got {len(all_shards)}")
            return self.shard_tensor
            
        # Concatenate shards along the sharded dimension
        full_tensor = torch.cat(all_shards, dim=self.shard_indices[0])
        return full_tensor

class FSDPShardingManager:
    """Manages FSDP-style parameter sharding across devices."""
    
    def __init__(self, world_size: int, device_rank: int, distributed_backend_type: str = "multiprocess"):
        self.world_size = world_size
        self.device_rank = device_rank
        self.distributed_backend = create_distributed_backend(
            distributed_backend_type, world_size, device_rank
        )
        self.parameter_shards = {}
        self.model = None
        
        # Initialize the distributed backend
        try:
            self.distributed_backend.initialize()
            logger.info(f"FSDPShardingManager initialized for device {device_rank}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed backend: {e}")
            # Fallback to single device
            self.distributed_backend = create_distributed_backend("fallback", world_size, device_rank)
            self.distributed_backend.initialize()
            
    def shard_model_parameters(self, model: Model) -> Dict[str, FSDPParameterShard]:
        """
        Shard model parameters across devices while keeping model structure intact.
        
        Args:
            model: Keras model to shard
            
        Returns:
            Dictionary mapping parameter names to their shards
        """
        self.model = model
        sharded_parameters = {}
        
        try:
            # Get all trainable parameters
            trainable_vars = model.trainable_variables
            
            for var in trainable_vars:
                param_name = var.name
                param_shape = var.shape
                
                # Determine sharding strategy based on parameter type and shape
                shard_info = self._determine_sharding_strategy(param_name, param_shape)
                
                if shard_info:
                    # Create parameter shard for this device
                    shard = FSDPParameterShard(
                        parameter_name=param_name,
                        full_shape=param_shape,
                        shard_shape=shard_info['shard_shape'],
                        device_rank=self.device_rank,
                        world_size=self.world_size,
                        shard_indices=shard_info['shard_indices']
                    )
                    
                    # Initialize shard with portion of original parameter
                    self._initialize_shard_from_parameter(shard, var)
                    
                    sharded_parameters[param_name] = shard
                    logger.info(f"Device {self.device_rank}: Created shard for {param_name} with shape {shard.shard_shape}")
                    
        except Exception as e:
            logger.error(f"Parameter sharding failed: {e}")
            
        return sharded_parameters
        
    def _determine_sharding_strategy(self, param_name: str, param_shape: tuple) -> Optional[Dict[str, Any]]:
        """
        Determine how to shard a parameter based on its name and shape.
        
        Args:
            param_name: Name of the parameter
            param_shape: Shape of the parameter
            
        Returns:
            Sharding strategy information or None if no sharding
        """
        if len(param_shape) < 2:
            # 1D parameters (bias, etc.) - shard along the single dimension
            return {
                'shard_shape': (param_shape[0] // self.world_size,),
                'shard_indices': (0,),
                'sharding_dim': 0
            }
            
        # 2D parameters (weights) - determine sharding based on layer type
        if 'kernel' in param_name or 'weight' in param_name:
            # Weight matrices - shard along output dimension (column-wise)
            output_dim = param_shape[1]
            shard_size = output_dim // self.world_size
            
            if self.device_rank < self.world_size - 1:
                shard_shape = (param_shape[0], shard_size)
                shard_indices = (1, self.device_rank * shard_size, (self.device_rank + 1) * shard_size)
            else:
                # Last device gets remaining parameters
                remaining = output_dim - (self.world_size - 1) * shard_size
                shard_shape = (param_shape[0], remaining)
                shard_indices = (1, (self.world_size - 1) * shard_size, output_dim)
                
            return {
                'shard_shape': shard_shape,
                'shard_indices': shard_indices,
                'sharding_dim': 1
            }
            
        elif 'bias' in param_name:
            # Bias vectors - shard along the single dimension
            if self.device_rank < self.world_size - 1:
                shard_size = param_shape[0] // self.world_size
                shard_shape = (shard_size,)
                shard_indices = (0, self.device_rank * shard_size, (self.device_rank + 1) * shard_size)
            else:
                remaining = param_shape[0] - (self.world_size - 1) * (param_shape[0] // self.world_size)
                shard_shape = (remaining,)
                shard_indices = (0, (self.world_size - 1) * (param_shape[0] // self.world_size), param_shape[0])
                
            return {
                'shard_shape': shard_shape,
                'shard_indices': shard_indices,
                'sharding_dim': 0
            }
            
        return None
        
    def _initialize_shard_from_parameter(self, shard: FSDPParameterShard, original_param):
        """Initialize shard with portion of the original parameter."""
        try:
            # Convert Keras parameter to numpy
            if hasattr(original_param, 'numpy'):
                param_np = original_param.numpy()
            else:
                param_np = np.array(original_param)
                
            # Extract the portion for this shard
            if len(shard.shard_indices) == 3:
                # 3D indices: (dim, start, end)
                dim, start, end = shard.shard_indices
                if dim == 0:
                    shard_data = param_np[start:end, :]
                elif dim == 1:
                    shard_data = param_np[:, start:end]
                else:
                    shard_data = param_np[start:end]
            else:
                # 1D indices: (start, end)
                start, end = shard.shard_indices
                shard_data = param_np[start:end]
                
            # Update shard tensor
            shard.shard_tensor.data = torch.tensor(shard_data, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Failed to initialize shard {shard.parameter_name}: {e}")
            
    def gather_parameter(self, param_name: str) -> Optional[torch.Tensor]:
        """
        Gather the full parameter from all devices.
        
        Args:
            param_name: Name of the parameter to gather
            
        Returns:
            Full parameter tensor or None if failed
        """
        if param_name not in self.parameter_shards:
            logger.error(f"Parameter {param_name} not found in shards")
            return None
            
        try:
            # Get local shard
            local_shard = self.parameter_shards[param_name]
            
            # Gather shards from all devices
            all_shards = self.distributed_backend.allgather(local_shard.shard_tensor)
            
            if all_shards and len(all_shards) == self.world_size:
                # Reconstruct full parameter
                full_param = local_shard.get_full_parameter(all_shards)
                return full_param
            else:
                logger.warning(f"Failed to gather parameter {param_name}")
                return None
                
        except Exception as e:
            logger.error(f"Parameter gathering failed for {param_name}: {e}")
            return None
            
    def scatter_parameter(self, param_name: str, full_parameter: torch.Tensor) -> bool:
        """
        Scatter a full parameter back to shards across devices.
        
        Args:
            param_name: Name of the parameter
            full_parameter: Full parameter tensor to scatter
            
        Returns:
            True if successful, False otherwise
        """
        if param_name not in self.parameter_shards:
            logger.error(f"Parameter {param_name} not found in shards")
            return False
            
        try:
            local_shard = self.parameter_shards[param_name]
            
            # Extract the portion for this device
            if len(local_shard.shard_indices) == 3:
                dim, start, end = local_shard.shard_indices
                if dim == 0:
                    shard_data = full_parameter[start:end, :]
                elif dim == 1:
                    shard_data = full_parameter[:, start:end]
                else:
                    shard_data = full_parameter[start:end]
            else:
                start, end = local_shard.shard_indices
                shard_data = full_parameter[start:end]
                
            # Update local shard
            local_shard.update_shard(torch.tensor(shard_data, dtype=torch.float32))
            return True
            
        except Exception as e:
            logger.error(f"Parameter scattering failed for {param_name}: {e}")
            return False
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for parameter shards."""
        memory_info = {
            'device_rank': self.device_rank,
            'world_size': self.world_size,
            'total_parameters': len(self.parameter_shards),
            'total_memory': 0,
            'shard_details': {}
        }
        
        for param_name, shard in self.parameter_shards.items():
            shard_memory = shard.shard_tensor.numel() * 4  # 4 bytes per float32
            memory_info['total_memory'] += shard_memory
            memory_info['shard_details'][param_name] = {
                'shard_shape': shard.shard_shape,
                'full_shape': shard.full_shape,
                'memory_bytes': shard_memory
            }
            
        return memory_info
        
    def cleanup(self):
        """Clean up distributed resources."""
        try:
            if hasattr(self.distributed_backend, 'finalize'):
                self.distributed_backend.finalize()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

def create_fsdp_sharding_manager(world_size: int, device_rank: int, 
                                distributed_backend_type: str = "multiprocess") -> FSDPShardingManager:
    """Factory function to create an FSDP sharding manager."""
    return FSDPShardingManager(world_size, device_rank, distributed_backend_type) 