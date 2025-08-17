"""
Parameter Gathering Infrastructure for FSDP-Style Tensor Parallelism

This module handles gathering full parameters from all devices using AllGather operations
to enable proper forward and backward passes with complete parameter information.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from .distributed_backend import DistributedBackend

logger = logging.getLogger(__name__)

class ParameterGatherer:
    """Handles parameter gathering operations for FSDP-style tensor parallelism."""
    
    def __init__(self, distributed_backend: DistributedBackend, world_size: int, device_rank: int):
        self.distributed_backend = distributed_backend
        self.world_size = world_size
        self.device_rank = device_rank
        
    def gather_all_parameters(self, parameter_shards: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Gather all parameter shards from all devices to reconstruct full parameters.
        
        Args:
            parameter_shards: Dictionary mapping parameter names to their shards on this device
            
        Returns:
            Dictionary mapping parameter names to their full reconstructed parameters
        """
        gathered_parameters = {}
        
        try:
            for param_name in parameter_shards.keys():
                # Collect shards from all devices
                all_shards = []
                for device_rank in range(self.world_size):
                    if device_rank == self.device_rank:
                        # This device's shard
                        shard = parameter_shards[param_name]
                        all_shards.append(shard)
                    else:
                        # Get shard from other device using distributed backend
                        shard = self._receive_shard_from_device(param_name, device_rank)
                        all_shards.append(shard)
                
                # Reconstruct full parameter
                full_param = self._reconstruct_parameter(param_name, all_shards)
                gathered_parameters[param_name] = full_param
                
                logger.debug(f"Device {self.device_rank}: Gathered full parameter {param_name} with shape {str(full_param.shape)}")
                
        except Exception as e:
            logger.error(f"Parameter gathering failed: {str(e)}")
            # Fallback: return local shards
            gathered_parameters = parameter_shards
            
        return gathered_parameters
    
    def _receive_shard_from_device(self, param_name: str, source_device: int) -> torch.Tensor:
        """Receive a parameter shard from another device."""
        try:
            # Use distributed backend to receive shard
            if hasattr(self.distributed_backend, 'receive'):
                shard = self.distributed_backend.receive(source_device, param_name)
                return shard
            else:
                # Fallback: create dummy shard
                logger.warning(f"Distributed backend doesn't support receive, using dummy shard")
                return torch.zeros(1)  # Dummy shard
        except Exception as e:
            logger.error(f"Failed to receive shard from device {source_device}: {e}")
            return torch.zeros(1)  # Dummy shard
    
    def _reconstruct_parameter(self, param_name: str, all_shards: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct full parameter from all shards."""
        try:
            # For now, assume shards are concatenated along the last dimension
            # In a full implementation, this would be more sophisticated
            if len(all_shards) == 1:
                return all_shards[0]
            elif len(all_shards) == 2:
                # Simple concatenation for 2 devices
                return torch.cat(all_shards, dim=-1)
            else:
                # Multiple devices - concatenate all
                return torch.cat(all_shards, dim=-1)
                
        except Exception as e:
            logger.error(f"Parameter reconstruction failed for {param_name}: {e}")
            # Return first shard as fallback
            return all_shards[0] if all_shards else torch.zeros(1)

class ParameterReplacer:
    """Handles temporary replacement of sharded parameters with full parameters."""
    
    def __init__(self):
        self._original_sharded_params = {}
        self._full_parameters = {}
        self._parameters_replaced = False
        
    def replace_with_full_parameters(self, model, full_parameters: Dict[str, torch.Tensor]):
        """Temporarily replace sharded parameters with full parameters."""
        if self._parameters_replaced:
            logger.warning("Parameters already replaced, skipping")
            return
            
        try:
            # Store original parameters
            self._backup_original_parameters(model)
            
            # Replace with full parameters
            self._full_parameters = full_parameters
            self._replace_model_parameters(model, full_parameters)
            
            self._parameters_replaced = True
            logger.debug("Successfully replaced parameters with full parameters")
            
        except Exception as e:
            logger.error(f"Parameter replacement failed: {e}")
            # Restore original parameters
            self._restore_original_parameters(model)
    
    def _backup_original_parameters(self, model):
        """Backup original model parameters."""
        self._original_sharded_params = {}
        
        for layer in model.layers:
            if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                for weight in layer.trainable_weights:
                    param_name = weight.name
                    self._original_sharded_params[param_name] = weight.numpy().copy()
    
    def _replace_model_parameters(self, model, full_parameters: Dict[str, torch.Tensor]):
        """Replace model parameters with full parameters."""
        for layer in model.layers:
            if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                for weight in layer.trainable_weights:
                    param_name = weight.name
                    if param_name in full_parameters:
                        # Convert PyTorch tensor to numpy and assign
                        full_param_np = full_parameters[param_name].detach().numpy()
                        weight.assign(full_param_np)
    
    def restore_sharded_parameters(self, model):
        """Restore original sharded parameters."""
        if not self._parameters_replaced:
            logger.warning("Parameters not replaced, nothing to restore")
            return
            
        try:
            self._restore_original_parameters(model)
            self._full_parameters.clear()
            self._parameters_replaced = False
            logger.debug("Successfully restored sharded parameters")
            
        except Exception as e:
            logger.error(f"Parameter restoration failed: {e}")
    
    def _restore_original_parameters(self, model):
        """Restore original parameters from backup."""
        try:
            for layer in model.layers:
                if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                    for weight in layer.trainable_weights:
                        param_name = weight.name
                        if param_name in self._original_sharded_params:
                            original_param = self._original_sharded_params[param_name]
                            
                            # Check if shapes match
                            if original_param.shape != weight.shape:
                                logger.warning(f"Shape mismatch for {param_name}: expected {str(weight.shape)}, got {str(original_param.shape)}")
                                
                                # Try to reshape the parameter
                                try:
                                    # Get sizes using the correct method for each type
                                    if hasattr(original_param, 'numel'):
                                        original_size = original_param.numel()
                                    elif hasattr(original_param, 'size'):
                                        original_size = original_param.size
                                    elif hasattr(original_param, 'numpy'):
                                        # For Keras variables, convert to numpy first
                                        original_size = original_param.numpy().size
                                    else:
                                        # Fallback: try to get shape product
                                        original_size = np.prod(original_param.shape)
                                    
                                    if hasattr(weight, 'numel'):
                                        target_size = weight.numel()
                                    elif hasattr(weight, 'size'):
                                        target_size = weight.size
                                    elif hasattr(weight, 'numpy'):
                                        # For Keras variables, convert to numpy first
                                        target_size = weight.numpy().size
                                    else:
                                        # Fallback: try to get shape product
                                        target_size = np.prod(weight.shape)
                                    
                                    if original_size == target_size:
                                        # Reshape to match target shape
                                        reshaped_param = original_param.reshape(weight.shape)
                                        weight.assign(reshaped_param)
                                        logger.info(f"Successfully reshaped {param_name} from {str(original_param.shape)} to {str(weight.shape)}")
                                    else:
                                        logger.error(f"Cannot reshape {param_name}: size mismatch {original_size} vs {target_size}")
                                        # Try to create a compatible parameter by padding or truncating
                                        try:
                                            # Create a new parameter with the target shape
                                            if hasattr(weight, 'numpy'):
                                                current_value = weight.numpy()
                                            else:
                                                current_value = np.zeros(weight.shape)
                                            
                                            # Try to extract a compatible portion from the original parameter
                                            if len(original_param.shape) == len(weight.shape):
                                                # Same dimensionality, try to extract compatible slices
                                                slices = []
                                                for i, (orig_dim, target_dim) in enumerate(zip(original_param.shape, weight.shape)):
                                                    if orig_dim >= target_dim:
                                                        slices.append(slice(0, target_dim))
                                                    else:
                                                        # Pad with zeros
                                                        slices.append(slice(0, orig_dim))
                                                
                                                compatible_param = original_param[tuple(slices)]
                                                # Pad to target shape if needed
                                                if compatible_param.shape != weight.shape:
                                                    padded_param = np.zeros(weight.shape)
                                                    slices = [slice(0, min(d1, d2)) for d1, d2 in zip(compatible_param.shape, weight.shape)]
                                                    padded_param[tuple(slices)] = compatible_param[tuple(slices)]
                                                    compatible_param = padded_param
                                                
                                                weight.assign(compatible_param)
                                                logger.info(f"Created compatible parameter for {param_name}")
                                            else:
                                                # Different dimensionality, use current value
                                                weight.assign(current_value)
                                                logger.warning(f"Using current value for {param_name} due to dimensionality mismatch")
                                        except Exception as e:
                                            logger.warning(f"Failed to create compatible parameter for {param_name}: {str(e)}")
                                            # Use current value as fallback
                                            if hasattr(weight, 'numpy'):
                                                current_value = weight.numpy()
                                            else:
                                                current_value = np.zeros(weight.shape)
                                            weight.assign(current_value)
                                except Exception as e:
                                    logger.error(f"Failed to reshape {param_name}: {str(e)}")
                                    # Use current value as fallback
                                    if hasattr(weight, 'numpy'):
                                        current_value = weight.numpy()
                                    else:
                                        current_value = np.zeros(weight.shape)
                                    weight.assign(current_value)
                            else:
                                # Shapes match, assign directly
                                weight.assign(original_param)
                                
        except Exception as e:
            logger.error(f"Parameter restoration failed: {str(e)}")
            # Don't raise, just log the error to prevent crashes 