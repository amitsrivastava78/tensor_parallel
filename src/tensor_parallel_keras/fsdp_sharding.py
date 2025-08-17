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
from .parameter_gathering import ParameterGatherer, ParameterReplacer
from .gradient_sharding import GradientSharder, GradientComputer, GradientSynchronizer

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
        self.model = None
        self.parameter_shards = {}  # Initialize parameter_shards
        self._parameters_gathered = False
        self._full_parameters = {}
        
        # Initialize distributed backend with real communication support
        self.distributed_backend = self._create_real_distributed_backend(distributed_backend_type, world_size, device_rank)
        
        # Initialize parameter gathering and replacement components
        self.parameter_gatherer = ParameterGatherer(self.distributed_backend, world_size, device_rank)
        self.parameter_replacer = ParameterReplacer()  # No parameters needed
        
        # Initialize gradient computation and sharding components
        self.gradient_computer = GradientComputer()  # No parameters needed
        self.gradient_sharder = GradientSharder(self.distributed_backend, world_size, device_rank)
        self.gradient_synchronizer = GradientSynchronizer(self.distributed_backend, world_size, device_rank)
        
        # Initialize the distributed backend
        try:
            self.distributed_backend.initialize()
            logger.info(f"FSDPShardingManager initialized for device {device_rank} with {type(self.distributed_backend).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed backend: {e}")
            # Fallback to single device
            self.distributed_backend = create_distributed_backend("fallback", world_size, device_rank)
            self.distributed_backend.initialize()
    
    def _create_real_distributed_backend(self, backend_type: str, world_size: int, device_rank: int):
        """
        Create a distributed backend with real communication support when possible.
        Prioritizes real backends over simulation.
        """
        try:
            # Try to create real distributed backend first
            if backend_type == "multiprocess" and world_size > 1:
                # Check if we can use real communication
                if self._can_use_real_communication():
                    logger.info(f"Device {device_rank}: Using real distributed communication")
                    return create_distributed_backend("multiprocess", world_size, device_rank)
                else:
                    logger.warning(f"Device {device_rank}: Real communication not available, using simulation")
                    return create_distributed_backend("fallback", world_size, device_rank)
            else:
                # Single device or fallback
                return create_distributed_backend(backend_type, world_size, device_rank)
                
        except Exception as e:
            logger.warning(f"Failed to create real distributed backend: {e}, using fallback")
            return create_distributed_backend("fallback", world_size, device_rank)
    
    def _can_use_real_communication(self) -> bool:
        """Check if real cross-device communication is available."""
        try:
            # Check for CUDA devices (GPU)
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                return True
            
            # Check for MPI support
            try:
                import mpi4py
                return True
            except ImportError:
                pass
            
            # Check for other distributed backends
            if hasattr(torch, 'distributed') and torch.distributed.is_available():
                return True
                
            return False
            
        except Exception:
            return False
    
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
            logger.debug(f"Device {self.device_rank}: Found {len(trainable_vars)} trainable variables")
            
            for i, var in enumerate(trainable_vars):
                param_name = var.name
                param_shape = var.shape
                
                # Debug logging to see what type param_shape is
                logger.debug(f"Device {self.device_rank}: Processing variable {i+1}/{len(trainable_vars)}: {param_name}")
                logger.debug(f"Parameter {param_name}: type={type(param_shape)}, value={str(param_shape)}")
                
                # Ensure param_shape is a tuple
                if not isinstance(param_shape, tuple):
                    logger.warning(f"Parameter {param_name} has non-tuple shape: {type(param_shape)} = {str(param_shape)}")
                    if hasattr(param_shape, '__iter__'):
                        param_shape = tuple(param_shape)
                    else:
                        param_shape = (param_shape,)
                    logger.debug(f"Converted shape to: {str(param_shape)}")
                
                # Determine sharding strategy based on parameter type and shape
                logger.debug(f"Device {self.device_rank}: Determining sharding strategy for {param_name}")
                shard_info = self._determine_sharding_strategy(param_name, param_shape)
                
                if shard_info:
                    logger.debug(f"Device {self.device_rank}: Creating parameter shard for {param_name}")
                    # Create parameter shard
                    param_shard = FSDPParameterShard(
                        parameter_name=param_name,
                        full_shape=param_shape,
                        shard_shape=shard_info['shard_shape'],
                        device_rank=self.device_rank,
                        world_size=self.world_size,
                        shard_indices=shard_info['shard_indices']
                    )
                    
                    logger.debug(f"Device {self.device_rank}: Initializing parameter shard for {param_name}")
                    # Initialize shard with appropriate values
                    self._initialize_parameter_shard(param_shard, var, shard_info)
                    
                    sharded_parameters[param_name] = param_shard
                    logger.info(f"Device {self.device_rank}: Created shard for {param_name} with shape {str(shard_info['shard_shape'])}")
                else:
                    logger.warning(f"Device {self.device_rank}: No sharding strategy for {param_name}")
            
            logger.info(f"Device {self.device_rank}: Sharded {len(sharded_parameters)} parameters")
            
        except Exception as e:
            logger.error(f"Parameter sharding failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        return sharded_parameters
    
    def gather_parameters_for_computation(self) -> Dict[str, torch.Tensor]:
        """
        Gather full parameters from all devices for forward/backward computation.
        This is the core of FSDP - each device needs full parameters for computation.
        """
        if self._parameters_gathered:
            logger.debug("Parameters already gathered, returning cached full parameters")
            return self._full_parameters
            
        try:
            # Ensure model is available
            if self.model is None:
                logger.error("Model is None, cannot gather parameters")
                return {}
            
            # Check if we have parameter shards
            if not self.parameter_shards:
                logger.warning("No parameter shards available, using fallback")
                # Create fallback parameters from the model
                fallback_params = {}
                if self.model is not None:
                    for var in self.model.trainable_variables:
                        if hasattr(var, 'numpy'):
                            fallback_params[var.name] = torch.tensor(var.numpy(), dtype=torch.float32)
                        else:
                            fallback_params[var.name] = torch.tensor(var, dtype=torch.float32)
                self._full_parameters = fallback_params
                self._parameters_gathered = True
                return fallback_params
            
            # Convert parameter shards to PyTorch tensors
            pytorch_shards = {}
            for param_name, param_shard in self.parameter_shards.items():
                pytorch_shards[param_name] = param_shard.get_shard_tensor()
            
            # Gather full parameters
            self._full_parameters = self.parameter_gatherer.gather_all_parameters(pytorch_shards)
            
            # Replace model parameters with full parameters
            if self.model is not None:
                self.parameter_replacer.replace_with_full_parameters(self.model, self._full_parameters)
            
            self._parameters_gathered = True
            logger.info(f"Device {self.device_rank}: Gathered full parameters for computation")
            
            return self._full_parameters
            
        except Exception as e:
            logger.error(f"Parameter gathering failed: {e}")
            # Return local shards as fallback
            fallback_params = {}
            for param_name, param_shard in self.parameter_shards.items():
                fallback_params[param_name] = param_shard.get_shard_tensor()
            return fallback_params
    
    def cleanup_full_parameters(self):
        """Clean up full parameters and restore sharded state."""
        if self._parameters_gathered:
            try:
                # Restore sharded parameters
                self.parameter_replacer.restore_sharded_parameters(self.model)
                
                # Clear full parameters
                self._full_parameters.clear()
                self._parameters_gathered = False
                
                logger.info(f"Device {self.device_rank}: Cleaned up full parameters")
                
            except Exception as e:
                logger.error(f"Parameter cleanup failed: {e}")
    
    def compute_gradients_with_full_parameters(self, loss: torch.Tensor) -> List[torch.Tensor]:
        """Compute gradients with respect to full parameters."""
        try:
            # Ensure parameters are gathered
            if not self._parameters_gathered:
                self.gather_parameters_for_computation()
            
            # Get trainable variables (should now be full parameters)
            trainable_vars = []
            if self.model is not None:
                for layer in self.model.layers:
                    if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                        for weight in layer.trainable_weights:
                            # Convert Keras tensor to PyTorch tensor
                            if hasattr(weight, 'numpy'):
                                torch_weight = torch.tensor(weight.numpy(), requires_grad=True)
                            else:
                                torch_weight = torch.tensor(weight, requires_grad=True)
                            trainable_vars.append(torch_weight)
            
            # If no trainable variables found, return empty list
            if not trainable_vars:
                logger.warning("No trainable variables found for gradient computation")
                return []
            
            # Compute gradients
            gradients = self.gradient_computer.compute_gradients_with_full_parameters(loss, trainable_vars)
            
            return gradients
            
        except Exception as e:
            logger.error(f"Gradient computation failed: {str(e)}")
            return []
    
    def shard_gradients(self, full_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Shard gradients back to respective devices."""
        try:
            # Convert parameter shards to PyTorch tensors
            pytorch_shards = {}
            for param_name, param_shard in self.parameter_shards.items():
                pytorch_shards[param_name] = param_shard.get_shard_tensor()
            
            # Shard gradients
            sharded_gradients = self.gradient_sharder.shard_gradients(full_gradients, pytorch_shards)
            
            return sharded_gradients
            
        except Exception as e:
            logger.error(f"Gradient sharding failed: {e}")
            return full_gradients
    
    def synchronize_gradients(self, local_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Synchronize gradients across devices."""
        try:
            return self.gradient_synchronizer.synchronize_gradients(local_gradients)
        except Exception as e:
            logger.error(f"Gradient synchronization failed: {e}")
            return local_gradients
    
    def cleanup(self):
        """Clean up distributed resources."""
        try:
            # Clean up full parameters
            self.cleanup_full_parameters()
            
            # Clean up distributed backend
            if hasattr(self.distributed_backend, 'finalize'):
                self.distributed_backend.finalize()
                
            logger.info(f"Device {self.device_rank}: Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _determine_sharding_strategy(self, param_name: str, param_shape: tuple) -> Optional[Dict]:
        """Determine how to shard a parameter."""
        try:
            if len(param_shape) == 1:
                # Bias vector - shard along the only dimension
                shard_size = max(1, param_shape[0] // self.world_size)  # Ensure minimum size of 1
                
                start_idx = min(self.device_rank * shard_size, param_shape[0] - 1)
                end_idx = min(start_idx + shard_size, param_shape[0])
                
                # Ensure we don't go out of bounds
                if start_idx >= param_shape[0]:
                    start_idx = 0
                    end_idx = min(shard_size, param_shape[0])
                
                return {
                    'shard_shape': (end_idx - start_idx,),
                    'shard_indices': (start_idx, end_idx)
                }
                
            elif len(param_shape) == 2:
                # Weight matrix - shard along output dimension (last dimension)
                shard_size = max(1, param_shape[1] // self.world_size)  # Ensure minimum size of 1
                
                start_idx = min(self.device_rank * shard_size, param_shape[1] - 1)
                end_idx = min(start_idx + shard_size, param_shape[1])
                
                # Ensure we don't go out of bounds
                if start_idx >= param_shape[1]:
                    start_idx = 0
                    end_idx = min(shard_size, param_shape[1])
                
                return {
                    'shard_shape': (param_shape[0], end_idx - start_idx),
                    'shard_indices': (start_idx, end_idx)
                }
                
            else:
                # Higher dimensional parameter - shard along last dimension
                shard_size = max(1, param_shape[-1] // self.world_size)  # Ensure minimum size of 1
                
                start_idx = min(self.device_rank * shard_size, param_shape[-1] - 1)
                end_idx = min(start_idx + shard_size, param_shape[-1])
                
                # Ensure we don't go out of bounds
                if start_idx >= param_shape[-1]:
                    start_idx = 0
                    end_idx = min(shard_size, param_shape[-1])
                
                shard_shape = list(param_shape)
                shard_shape[-1] = end_idx - start_idx
                
                return {
                    'shard_shape': tuple(shard_shape),
                    'shard_indices': (start_idx, end_idx)
                }
                
        except Exception as e:
            logger.error(f"Failed to determine sharding strategy for {param_name}: {e}")
            return None
    
    def _initialize_parameter_shard(self, param_shard: FSDPParameterShard, original_var, shard_info: Dict):
        """Initialize parameter shard with appropriate values."""
        try:
            # Get original parameter values
            if hasattr(original_var, 'numpy'):
                original_values = original_var.numpy()
            elif hasattr(original_var, 'shape'):
                # Handle Keras variables that have shape but no numpy method
                original_values = np.array(original_var)
            else:
                # Handle scalar values
                original_values = np.array([float(original_var)])
            
            # Ensure original_values is a numpy array
            if not isinstance(original_values, np.ndarray):
                original_values = np.array(original_values)
            
            # Extract shard values
            start_idx, end_idx = shard_info['shard_indices']
            
            if len(original_values.shape) == 1:
                # 1D parameter (bias)
                shard_values = original_values[start_idx:end_idx]
            elif len(original_values.shape) == 2:
                # 2D parameter (weight matrix)
                shard_values = original_values[:, start_idx:end_idx]
            else:
                # Higher dimensional parameter
                slices = [slice(None)] * (len(original_values.shape) - 1) + [slice(start_idx, end_idx)]
                shard_values = original_values[slices]
            
            # Assign to shard tensor
            param_shard.shard_tensor.data = torch.tensor(shard_values, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Failed to initialize shard {param_shard.parameter_name}: {e}")
            # Keep shard as zeros
    
    def get_memory_usage(self) -> float:
        """Get memory usage of parameter shards in MB."""
        try:
            total_memory = 0
            # Ensure parameter_shards is initialized
            if not hasattr(self, 'parameter_shards') or not self.parameter_shards:
                return 0.0
                
            for param_shard in self.parameter_shards.values():
                if hasattr(param_shard, 'shard_tensor') and param_shard.shard_tensor is not None:
                    total_memory += param_shard.shard_tensor.numel() * 4  # 4 bytes per float32
            return total_memory / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Failed to compute memory usage: {e}")
            return 0.0

    def split_data_batch(self, batch_data, device_rank):
        """
        Split data batch across devices for true data parallelism.
        This is a key feature of FSDP - each device processes different data chunks.
        
        Args:
            batch_data: Input data batch (numpy array, tensor, or dictionary)
            device_rank: Rank of this device
            
        Returns:
            Data shard for this device
        """
        try:
            # Handle dictionary inputs (like BERT inputs)
            if isinstance(batch_data, dict):
                sharded_dict = {}
                for key, value in batch_data.items():
                    # Convert tensor to numpy if needed
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    
                    # Ensure value is a numpy array
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)
                    
                    # Split along batch dimension
                    batch_size = value.shape[0]
                    shard_size = batch_size // self.world_size
                    start_idx = device_rank * shard_size
                    end_idx = start_idx + shard_size
                    
                    # Handle remainder for the last device
                    if device_rank == self.world_size - 1:
                        end_idx = batch_size
                    
                    # Extract shard for this key
                    sharded_dict[key] = value[start_idx:end_idx]
                
                logger.info(f"Device {self.device_rank}: Split dictionary batch with keys: {list(sharded_dict.keys())}")
                return sharded_dict
            
            # Handle tensor/array inputs
            else:
                # Convert to numpy if it's a tensor
                if hasattr(batch_data, 'numpy'):
                    batch_data = batch_data.numpy()
                
                # Ensure batch_data is a numpy array
                if not isinstance(batch_data, np.ndarray):
                    batch_data = np.array(batch_data)
                
                batch_size = batch_data.shape[0]
                
                # Calculate shard size and indices
                shard_size = batch_size // self.world_size
                start_idx = device_rank * shard_size
                end_idx = start_idx + shard_size
                
                # Handle remainder for the last device
                if device_rank == self.world_size - 1:
                    end_idx = batch_size
                
                # Extract data shard for this device
                data_shard = batch_data[start_idx:end_idx]
                
                logger.info(f"Device {self.device_rank}: Split batch {batch_data.shape} -> shard {data_shard.shape} "
                           f"(indices {start_idx}:{end_idx})")
                
                return data_shard
            
        except Exception as e:
            logger.error(f"Data batch splitting failed: {e}")
            # Fallback: return original data
            return batch_data

    def forward_pass_with_cleanup(self, inputs, training=None, model=None):
        """
        Forward pass with immediate memory cleanup - core FSDP memory optimization.
        
        This method implements the key FSDP pattern:
        1. Gather full parameters (AllGather)
        2. Forward pass with full parameters
        3. IMMEDIATELY cleanup and restore sharded state
        4. Return output
        
        This ensures peak memory = shard_size + temporary_full_parameters
        """
        try:
            # Use provided model or fallback to self.model
            target_model = model if model is not None else self.model
            
            # Step 1: Gather full parameters for forward pass
            if not self._parameters_gathered:
                self.gather_parameters_for_computation()
            
            # Step 2: Forward pass with full parameters
            if target_model is not None:
                output = target_model(inputs, training=training)
            else:
                logger.warning("Model is None, cannot perform forward pass")
                return inputs
            
            # Step 3: IMMEDIATELY cleanup full parameters to save memory
            self.cleanup_full_parameters()
            
            logger.info(f"Device {self.device_rank}: Forward pass completed with memory cleanup")
            return output
            
        except Exception as e:
            logger.error(f"Forward pass with cleanup failed: {e}")
            # Ensure cleanup happens even on error
            self.cleanup_full_parameters()
            raise
    
    def backward_pass_with_gathering(self, loss, inputs):
        """
        Backward pass with parameter re-gathering for gradient computation.
        
        Since we cleaned up after forward pass, we need to gather parameters again
        for the backward pass - this is the standard FSDP pattern.
        """
        try:
            # Re-gather parameters for backward pass
            if not self._parameters_gathered:
                self.gather_parameters_for_computation()
            
            # Compute gradients with full parameters
            gradients = self.gradient_computer.compute_gradients_with_full_parameters(loss, inputs)
            
            logger.info(f"Device {self.device_rank}: Backward pass completed with {len(gradients)} gradients")
            return gradients
            
        except Exception as e:
            logger.error(f"Backward pass failed: {e}")
            raise

def create_fsdp_sharding_manager(world_size: int, device_rank: int, 
                                distributed_backend_type: str = "multiprocess") -> FSDPShardingManager:
    """Factory function to create an FSDP sharding manager."""
    return FSDPShardingManager(world_size, device_rank, distributed_backend_type) 