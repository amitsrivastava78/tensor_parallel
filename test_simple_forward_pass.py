#!/usr/bin/env python3
"""
Simple test to verify forward pass numerical correctness.
"""

import numpy as np
import keras
from keras import layers, Model
import os

def setup_jax_backend():
    """Set up JAX backend."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    return True

def create_simple_model():
    """Create a simple model."""
    inputs = keras.Input(shape=(4,))
    x = layers.Dense(8, activation='relu', name="dense_1")(inputs)
    outputs = layers.Dense(2, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_forward_pass_equivalence():
    """Test that forward passes are equivalent."""
    print("🧪 Testing Forward Pass Equivalence")
    print("=" * 50)
    
    setup_jax_backend()
    
    # Create models
    single_cpu_model = create_simple_model()
    
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
    tp_model = TensorParallelKeras(
        model=create_simple_model(),
        world_size=2,
        distributed_backend="jax"
    )
    
    # Synchronize weights
    weights = single_cpu_model.get_weights()
    tp_model.set_weights(weights)
    
    print("✅ Models created and weights synchronized")
    
    # Create test data
    np.random.seed(123)
    x = np.random.randn(8, 4).astype(np.float32)
    
    # Forward pass
    single_output = single_cpu_model(x, training=False)
    tp_output = tp_model(x, training=False)
    
    print(f"Single CPU output shape: {single_output.shape}")
    print(f"Tensor Parallel output shape: {tp_output.shape}")
    
    # Compare outputs
    diff = np.abs(single_output.numpy() - tp_output.numpy())
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Check if they're close enough
    tolerance = 1e-6
    if max_diff < tolerance and mean_diff < tolerance/10:
        print("✅ Forward pass equivalence PASSED!")
        return True
    else:
        print("❌ Forward pass equivalence FAILED!")
        print(f"Single CPU output:\n{single_output.numpy()}")
        print(f"Tensor Parallel output:\n{tp_output.numpy()}")
        return False

if __name__ == "__main__":
    success = test_forward_pass_equivalence()
    exit(0 if success else 1)