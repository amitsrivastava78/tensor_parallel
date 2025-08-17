import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from .distributed_backend import create_distributed_backend

logger = logging.getLogger(__name__)

class OptimizerStateShard:
    """Represents a shard of optimizer states for a specific parameter group."""
    
    def __init__(self, parameter_names: List[str], state_shapes: Dict[str, tuple]):
        self.parameter_names = parameter_names
        self.state_shapes = state_shapes
        self.states = {}
        
        # Initialize optimizer states
        for param_name in parameter_names:
            self.states[param_name] = {}
            for state_name, shape in state_shapes.items():
                if state_name in ['exp_avg', 'exp_avg_sq', 'momentum_buffer']:
                    # Initialize with zeros for Adam-like optimizers
                    self.states[param_name][state_name] = torch.zeros(shape, dtype=torch.float32)
                elif state_name in ['v', 'm']:
                    # Initialize with zeros for other optimizers
                    self.states[param_name][state_name] = torch.zeros(shape, dtype=torch.float32)
                else:
                    # Default initialization
                    self.states[param_name][state_name] = torch.zeros(shape, dtype=torch.float32)
                    
    def get_state(self, param_name: str, state_name: str) -> Optional[torch.Tensor]:
        """Get optimizer state for a specific parameter."""
        if param_name in self.states and state_name in self.states[param_name]:
            return self.states[param_name][state_name]
        return None
        
    def set_state(self, param_name: str, state_name: str, value: torch.Tensor):
        """Set optimizer state for a specific parameter."""
        if param_name in self.states:
            self.states[param_name][state_name] = value
            
    def get_all_states(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get all optimizer states for this shard."""
        return self.states
        
    def update_states(self, updates: Dict[str, Dict[str, torch.Tensor]]):
        """Update optimizer states with new values."""
        for param_name, param_states in updates.items():
            if param_name in self.states:
                for state_name, state_value in param_states.items():
                    self.states[param_name][state_name] = state_value

class OptimizerShardingManager:
    """Manages optimizer state sharding across devices."""
    
    def __init__(self, world_size: int, device_rank: int, distributed_backend_type: str = "multiprocess"):
        self.world_size = world_size
        self.device_rank = device_rank
        self.distributed_backend = create_distributed_backend(
            distributed_backend_type, world_size, device_rank
        )
        self.optimizer_states = {}
        self.parameter_mapping = {}
        
        # Initialize the distributed backend
        try:
            self.distributed_backend.initialize()
            logger.info(f"OptimizerShardingManager initialized for device {device_rank}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed backend: {e}")
            # Fallback to single device
            self.distributed_backend = create_distributed_backend("fallback", world_size, device_rank)
            self.distributed_backend.initialize()
            
    def register_parameter_group(self, group_id: int, parameters: List[torch.Tensor], 
                               optimizer_type: str = "adam") -> int:
        """
        Register a parameter group for optimizer state sharding.
        
        Args:
            group_id: Unique identifier for the parameter group
            parameters: List of parameters in this group
            optimizer_type: Type of optimizer (adam, sgd, etc.)
            
        Returns:
            Number of optimizer states created
        """
        try:
            # Determine optimizer state shapes based on optimizer type
            state_shapes = self._get_optimizer_state_shapes(parameters, optimizer_type)
            
            # Create parameter names for this group
            param_names = [f"param_{group_id}_{i}" for i in range(len(parameters))]
            
            # Create optimizer state shard
            state_shard = OptimizerStateShard(param_names, state_shapes)
            self.optimizer_states[group_id] = state_shard
            
            # Store parameter mapping
            self.parameter_mapping[group_id] = {
                'parameters': parameters,
                'param_names': param_names,
                'optimizer_type': optimizer_type
            }
            
            logger.info(f"Device {self.device_rank}: Registered parameter group {group_id} with {len(parameters)} parameters")
            return len(state_shapes)
            
        except Exception as e:
            logger.error(f"Failed to register parameter group {group_id}: {e}")
            return 0
            
    def _get_optimizer_state_shapes(self, parameters: List[torch.Tensor], 
                                   optimizer_type: str) -> Dict[str, tuple]:
        """Get optimizer state shapes based on optimizer type and parameter shapes."""
        state_shapes = {}
        
        for i, param in enumerate(parameters):
            param_name = f"param_{i}"
            param_shape = param.shape
            
            if optimizer_type.lower() == "adam":
                # Adam optimizer states: exp_avg, exp_avg_sq
                state_shapes[f"{param_name}_exp_avg"] = param_shape
                state_shapes[f"{param_name}_exp_avg_sq"] = param_shape
            elif optimizer_type.lower() == "sgd":
                # SGD optimizer states: momentum_buffer
                state_shapes[f"{param_name}_momentum_buffer"] = param_shape
            elif optimizer_type.lower() == "rmsprop":
                # RMSprop optimizer states: v
                state_shapes[f"{param_name}_v"] = param_shape
            else:
                # Default: assume Adam-like states
                state_shapes[f"{param_name}_exp_avg"] = param_shape
                state_shapes[f"{param_name}_exp_avg_sq"] = param_shape
                
        return state_shapes
        
    def get_optimizer_states(self, group_id: int) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Get optimizer states for a specific parameter group."""
        if group_id in self.optimizer_states:
            return self.optimizer_states[group_id].get_all_states()
        return None
        
    def update_optimizer_states(self, group_id: int, 
                               updates: Dict[str, Dict[str, torch.Tensor]]):
        """Update optimizer states for a specific parameter group."""
        if group_id in self.optimizer_states:
            self.optimizer_states[group_id].update_states(updates)
            
    def perform_optimizer_step(self, group_id: int, gradients: List[torch.Tensor], 
                              learning_rate: float = 0.001) -> bool:
        """
        Perform optimizer step for a specific parameter group.
        
        Args:
            group_id: Parameter group identifier
            gradients: Gradients for the parameters in this group
            learning_rate: Learning rate for the update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if group_id not in self.parameter_mapping:
                logger.error(f"Parameter group {group_id} not found")
                return False
                
            group_info = self.parameter_mapping[group_id]
            parameters = group_info['parameters']
            optimizer_type = group_info['optimizer_type']
            
            if len(parameters) != len(gradients):
                logger.error(f"Parameter-gradient mismatch: {len(parameters)} vs {len(gradients)}")
                return False
                
            # Perform optimizer step based on type
            if optimizer_type.lower() == "adam":
                return self._adam_step(group_id, parameters, gradients, learning_rate)
            elif optimizer_type.lower() == "sgd":
                return self._sgd_step(group_id, parameters, gradients, learning_rate)
            else:
                # Default to SGD
                return self._sgd_step(group_id, parameters, gradients, learning_rate)
                
        except Exception as e:
            logger.error(f"Optimizer step failed for group {group_id}: {e}")
            return False
            
    def _adam_step(self, group_id: int, parameters: List[torch.Tensor], 
                   gradients: List[torch.Tensor], learning_rate: float) -> bool:
        """Perform Adam optimizer step."""
        try:
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            
            for i, (param, grad) in enumerate(zip(parameters, gradients)):
                if grad is None:
                    continue
                    
                param_name = f"param_{i}"
                
                # Get optimizer states
                exp_avg = self.optimizer_states[group_id].get_state(param_name, "exp_avg")
                exp_avg_sq = self.optimizer_states[group_id].get_state(param_name, "exp_avg_sq")
                
                if exp_avg is None or exp_avg_sq is None:
                    continue
                    
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** (i + 1)
                bias_correction2 = 1 - beta2 ** (i + 1)
                
                step_size = learning_rate / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                
                # Update parameter
                param.data.addcdiv_(exp_avg, (exp_avg_sq.sqrt() / bias_correction2_sqrt).add(epsilon), value=-step_size)
                
                # Update optimizer states
                self.optimizer_states[group_id].set_state(param_name, "exp_avg", exp_avg)
                self.optimizer_states[group_id].set_state(param_name, "exp_avg_sq", exp_avg_sq)
                
            return True
            
        except Exception as e:
            logger.error(f"Adam step failed: {e}")
            return False
            
    def _sgd_step(self, group_id: int, parameters: List[torch.Tensor], 
                  gradients: List[torch.Tensor], learning_rate: float) -> bool:
        """Perform SGD optimizer step."""
        try:
            for i, (param, grad) in enumerate(zip(parameters, gradients)):
                if grad is None:
                    continue
                    
                # Simple SGD update
                param.data.add_(grad, alpha=-learning_rate)
                
            return True
            
        except Exception as e:
            logger.error(f"SGD step failed: {e}")
            return False
            
    def synchronize_parameters(self, group_id: int) -> bool:
        """
        Synchronize parameters across all devices using all-gather.
        
        Args:
            group_id: Parameter group identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if group_id not in self.parameter_mapping:
                logger.error(f"Parameter group {group_id} not found")
                return False
                
            parameters = self.parameter_mapping[group_id]['parameters']
            
            # Convert parameters to numpy for distributed communication
            param_arrays = []
            for param in parameters:
                if hasattr(param, 'detach'):
                    param_np = param.detach().cpu().numpy()
                else:
                    param_np = np.array(param)
                param_arrays.append(param_np)
                
            # Perform all-gather to synchronize parameters
            # For fallback backend, we need to handle lists differently
            if hasattr(self.distributed_backend, 'allgather'):
                try:
                    synchronized_params = self.distributed_backend.allgather(param_arrays)
                except Exception as e:
                    logger.warning(f"Allgather failed, using fallback: {e}")
                    # Fallback: just return the original parameters
                    synchronized_params = [param_arrays]
            else:
                # Fallback: just return the original parameters
                synchronized_params = [param_arrays]
            
            # Update local parameters with synchronized values
            if synchronized_params:
                # Handle fallback case where synchronized_params might be wrapped differently
                if len(synchronized_params) == 1 and isinstance(synchronized_params[0], list):
                    # Fallback case: synchronized_params[0] contains the actual parameters
                    actual_params = synchronized_params[0]
                    if len(actual_params) == len(parameters):
                        for i, (param, synced_param) in enumerate(zip(parameters, actual_params)):
                            if hasattr(param, 'data'):
                                param.data = torch.tensor(synced_param, dtype=param.dtype)
                            else:
                                # For non-tensor parameters, try to update in place
                                try:
                                    param[:] = synced_param
                                except:
                                    pass
                elif len(synchronized_params) == len(parameters):
                    # Normal case: direct mapping
                    for i, (param, synced_param) in enumerate(zip(parameters, synchronized_params)):
                        if hasattr(param, 'data'):
                            param.data = torch.tensor(synced_param, dtype=param.dtype)
                        else:
                            # For non-tensor parameters, try to update in place
                            try:
                                param[:] = synced_param
                            except:
                                pass
                            
            logger.info(f"Device {self.device_rank}: Synchronized {len(parameters)} parameters for group {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Parameter synchronization failed for group {group_id}: {e}")
            return False
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for optimizer states."""
        memory_info = {
            'device_rank': self.device_rank,
            'world_size': self.world_size,
            'parameter_groups': len(self.parameter_mapping),
            'total_parameters': 0,
            'total_optimizer_states': 0,
            'memory_per_group': {}
        }
        
        for group_id, group_info in self.parameter_mapping.items():
            parameters = group_info['parameters']
            param_count = sum(p.numel() for p in parameters if hasattr(p, 'numel'))
            
            # Estimate optimizer state memory (typically 2x parameter size for Adam)
            optimizer_memory = param_count * 2  # Rough estimate
            
            memory_info['total_parameters'] += param_count
            memory_info['total_optimizer_states'] += optimizer_memory
            memory_info['memory_per_group'][group_id] = {
                'parameters': param_count,
                'optimizer_states': optimizer_memory
            }
            
        return memory_info
        
    def cleanup(self):
        """Clean up distributed resources."""
        try:
            if hasattr(self.distributed_backend, 'finalize'):
                self.distributed_backend.finalize()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

def create_optimizer_sharding_manager(world_size: int, device_rank: int, 
                                    distributed_backend_type: str = "multiprocess") -> OptimizerShardingManager:
    """Factory function to create an optimizer sharding manager."""
    return OptimizerShardingManager(world_size, device_rank, distributed_backend_type) 