"""
Tensor Parallel Keras - Distributed Training with Keras 3.0
"""

# Import main classes
from .tensor_parallel_keras import TensorParallelKeras
from .backend_interface import BackendInterface, BackendFactory, set_default_backend, get_default_backend

# Initialize REAL JAX backend by default if available
try:
    from .real_jax_backend import RealJAXBackend
    # Set REAL JAX as the default backend
    real_jax_backend = RealJAXBackend()
    set_default_backend(real_jax_backend)
    print("🚀 REAL JAX Backend initialized with NO STUBS - actual distributed computation!")
except ImportError as e:
    print(f"❌ JAX not available: {e}")
    print("❌ CANNOT PROCEED - REAL distributed computation required!")
    raise ImportError("JAX is required for real tensor parallelism - no stubs allowed!")
except Exception as e:
    print(f"❌ REAL JAX Backend initialization failed: {e}")
    print("❌ CANNOT PROCEED - REAL distributed computation required!")
    raise RuntimeError(f"REAL JAX backend initialization failed: {e}")

# Version
__version__ = "0.1.0"

__all__ = [
    "TensorParallelKeras",
    "BackendInterface", 
    "BackendFactory",
    "set_default_backend",
    "get_default_backend"
] 