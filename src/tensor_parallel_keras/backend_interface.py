#!/usr/bin/env python3
"""
Backend-Agnostic Interface for Tensor Parallelism
Defines abstract interfaces that can be implemented by different backends.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BackendInterface(ABC):
    """Abstract base class for backend implementations."""
    
    @abstractmethod
    def get_device_count(self) -> int:
        """Get the number of available devices."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        pass
    
    @abstractmethod
    def create_sharded_parameters(self, params: List[Any], world_size: int) -> List[List[Any]]:
        """Create sharded parameters across devices."""
        pass
    
    @abstractmethod
    def all_reduce(self, tensors: List[Any], op: str = "sum") -> List[Any]:
        """Perform AllReduce operation across devices."""
        pass
    
    @abstractmethod
    def all_gather(self, tensors: List[Any], dim: int = -1) -> List[Any]:
        """Perform AllGather operation across devices."""
        pass
    
    @abstractmethod
    def broadcast(self, tensor: Any, src_rank: int = 0) -> List[Any]:
        """Broadcast tensor from source rank to all ranks."""
        pass
    
    @abstractmethod
    def scatter(self, tensor: Any, world_size: int, dim: int = -1) -> List[Any]:
        """Scatter tensor across devices."""
        pass

class BackendFactory:
    """Factory for creating backend implementations."""
    
    @staticmethod
    def create_backend(backend_name: str) -> BackendInterface:
        """Create a backend implementation."""
        if backend_name == "jax":
            from .jax_backend import JAXBackend
            return JAXBackend()
        elif backend_name == "pytorch":
            from .pytorch_backend import PyTorchBackend
            return PyTorchBackend()
        elif backend_name == "tensorflow":
            from .tensorflow_backend import TensorFlowBackend
            return TensorFlowBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

# Default backend interface (will be set by specific implementations)
_default_backend: Optional[BackendInterface] = None

def set_default_backend(backend: BackendInterface):
    """Set the default backend."""
    global _default_backend
    _default_backend = backend

def get_default_backend() -> BackendInterface:
    """Get the default backend."""
    if _default_backend is None:
        raise RuntimeError("No default backend set. Call set_default_backend() first.")
    return _default_backend 