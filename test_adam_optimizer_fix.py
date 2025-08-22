#!/usr/bin/env python3
"""
Test to fix and verify Adam optimizer with tensor parallelism.
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
    """Create a simple test model."""
    inputs = keras.Input(shape=(4,))
    x = layers.Dense(8, activation='relu', name="dense_1")(inputs)
    outputs = layers.Dense(2, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_adam_optimizer_fix():
    """Test that Adam optimizer works correctly with tensor parallelism."""
    print("🧪 Testing Adam Optimizer Fix with Tensor Parallelism")
    print("=" * 60)
    
    setup_jax_backend()
    
    # Create test data
    np.random.seed(42)
    x = np.random.randn(16, 4).astype(np.float32)
    y = np.random.randn(16, 2).astype(np.float32)
    
    # Test 1: Single CPU model with Adam
    print("\n🔍 Test 1: Single CPU Model with Adam")
    single_model = create_simple_model()
    single_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    single_model.compile(optimizer=single_optimizer, loss='mse')
    
    print("✅ Single CPU model compiled successfully")
    
    # Store initial weights
    initial_weights = [w.numpy().copy() for w in single_model.weights]
    
    # Single training step
    single_loss = single_model.train_on_batch(x, y)
    single_updated_weights = [w.numpy().copy() for w in single_model.weights]
    
    print(f"✅ Single CPU training completed - Loss: {single_loss:.6f}")
    
    # Test 2: Tensor parallel model with separate optimizer
    print("\n🔍 Test 2: Tensor Parallel Model with Separate Adam")
    
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
    
    # Create base model for tensor parallelism
    base_model = create_simple_model()
    
    # Set same initial weights
    base_model.set_weights(initial_weights)
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model=base_model,
        world_size=2,
        distributed_backend="jax"
    )
    
    # Create SEPARATE optimizer for tensor parallel model
    tp_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile tensor parallel model with its own optimizer
    tp_model.compile(optimizer=tp_optimizer, loss='mse')
    
    print("✅ Tensor parallel model compiled successfully")
    
    # Verify initial weights are identical
    tp_initial_weights = [w.numpy().copy() for w in tp_model.weights]
    
    initial_weights_match = all(
        np.allclose(single_w, tp_w, atol=1e-6) 
        for single_w, tp_w in zip(initial_weights, tp_initial_weights)
    )
    print(f"✅ Initial weights match: {initial_weights_match}")
    
    # Tensor parallel training step
    try:
        tp_loss = tp_model.train_on_batch(x, y)
        tp_updated_weights = [w.numpy().copy() for w in tp_model.weights]
        
        print(f"✅ Tensor parallel training completed - Loss: {tp_loss:.6f}")
        
        # Compare results
        print("\n🔍 Test 3: Comparing Results")
        loss_diff = abs(single_loss - tp_loss)
        print(f"Loss difference: {loss_diff:.2e}")
        
        # Compare weight updates
        print("\n📊 Weight Update Comparison:")
        max_weight_diffs = []
        
        for i, (single_w, tp_w) in enumerate(zip(single_updated_weights, tp_updated_weights)):
            diff = np.abs(single_w - tp_w)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            max_weight_diffs.append(max_diff)
            print(f"  Weight {i}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        
        overall_max_diff = max(max_weight_diffs)
        
        # Success criteria
        loss_ok = loss_diff < 1e-5
        weights_ok = overall_max_diff < 1e-4  # More relaxed tolerance for optimizer states
        
        print(f"\n📊 Results:")
        print(f"✅ Loss matching: {'PASS' if loss_ok else 'FAIL'} (diff: {loss_diff:.2e})")
        print(f"✅ Weight updates: {'PASS' if weights_ok else 'FAIL'} (max diff: {overall_max_diff:.2e})")
        
        overall_success = loss_ok and weights_ok
        print(f"\n🎯 Overall Adam Optimizer Test: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Tensor parallel training failed: {e}")
        
        # Debug: Let's see what the exact issue is
        print("\n🔧 Debug Information:")
        print(f"  Single model variables: {len(single_model.trainable_weights)}")
        print(f"  TP model variables: {len(tp_model.trainable_weights)}")
        print(f"  Single optimizer class: {type(single_optimizer)}")
        print(f"  TP optimizer class: {type(tp_optimizer)}")
        
        # Check if the models have the same structure
        single_var_shapes = [w.shape for w in single_model.trainable_weights]
        tp_var_shapes = [w.shape for w in tp_model.trainable_weights]
        print(f"  Single model shapes: {single_var_shapes}")
        print(f"  TP model shapes: {tp_var_shapes}")
        
        return False

if __name__ == "__main__":
    success = test_adam_optimizer_fix()
    exit(0 if success else 1)