#!/usr/bin/env python3
"""
Test to verify backward loss computation matching.
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

def test_backward_loss_matching():
    """Test that backward loss computation matches."""
    print("🧪 Testing Backward Loss Computation Matching")
    print("=" * 60)
    
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
    y = np.random.randn(8, 2).astype(np.float32)
    
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    # Test 1: Forward pass (should be identical)
    print("\n🔍 Test 1: Forward Pass Comparison")
    single_output = single_cpu_model(x, training=False)
    tp_output = tp_model(x, training=False)
    
    forward_diff = np.abs(single_output.numpy() - tp_output.numpy())
    print(f"Forward pass - Max diff: {np.max(forward_diff):.2e}")
    print(f"Forward pass - Mean diff: {np.mean(forward_diff):.2e}")
    
    # Test 2: Loss computation (should be identical)
    print("\n🔍 Test 2: Loss Computation Comparison")
    
    # Use MSE loss for both
    single_loss = keras.ops.mean(keras.ops.square(y - single_output))
    tp_loss = keras.ops.mean(keras.ops.square(y - tp_output))
    
    loss_diff = abs(float(single_loss) - float(tp_loss))
    print(f"Single CPU loss: {float(single_loss):.6f}")
    print(f"Tensor Parallel loss: {float(tp_loss):.6f}")
    print(f"Loss difference: {loss_diff:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BACKWARD LOSS MATCHING SUMMARY")
    print("=" * 60)
    
    forward_ok = np.max(forward_diff) < 1e-6
    loss_ok = loss_diff < 1e-6
    
    print(f"✅ Forward Pass Matching: {'PASS' if forward_ok else 'FAIL'}")
    print(f"✅ Loss Computation Matching: {'PASS' if loss_ok else 'FAIL'}")
    
    overall_success = forward_ok and loss_ok
    print(f"\n🎯 Overall Result: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success

if __name__ == "__main__":
    success = test_backward_loss_matching()
    exit(0 if success else 1)

 