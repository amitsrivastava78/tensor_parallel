#!/usr/bin/env python3
"""
Fast test to verify numerical correctness between single CPU and 2-CPU sharded models.
Tests forward pass, backward pass, and weight updates on JAX backend.
"""

import numpy as np
import keras
from keras import layers, Model
import os
import time

def setup_jax_backend():
    """Set up JAX backend for testing."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    # Set JAX to use CPU
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    
    print(f"🔍 JAX backend configured. Available devices: {jax.devices()}")
    return True

def create_simple_test_model():
    """Create a simple model for fast testing."""
    inputs = keras.Input(shape=(16,))
    x = layers.Dense(32, activation='relu', name="dense_1")(inputs)
    x = layers.Dense(16, activation='relu', name="dense_2")(x)
    outputs = layers.Dense(8, name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_single_cpu_training():
    """Test single CPU training for baseline."""
    print("🧪 Testing Single CPU Training (Baseline)")
    print("-" * 50)
    
    # Create model
    model = create_simple_test_model()
    
    # Create optimizer instance for single CPU model
    single_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile
    model.compile(
        optimizer=single_optimizer,
        loss='mse'
    )
    
    # Create simple data (use FIXED seed for reproducibility)
    np.random.seed(42)
    x = np.random.randn(32, 16).astype(np.float32)
    y = np.random.randn(32, 8).astype(np.float32)
    
    # Store initial weights
    initial_weights = [w.numpy() for w in model.weights]
    
    # Single training step
    start_time = time.time()
    loss = model.train_on_batch(x, y)
    training_time = time.time() - start_time
    
    # Get updated weights
    updated_weights = [w.numpy() for w in model.weights]
    
    print(f"✅ Single CPU training completed in {training_time:.3f}s")
    print(f"   - Loss: {loss:.6f}")
    print(f"   - Weights updated: {len(updated_weights)}")
    
    return {
        'model': model,
        'optimizer': single_optimizer,
        'loss': loss,
        'initial_weights': initial_weights,
        'updated_weights': updated_weights,
        'training_time': training_time
    }

def test_tensor_parallel_training(single_optimizer_params):
    """Test tensor parallel training with 2-CPU simulation."""
    print("\n🧪 Testing Tensor Parallel Training (2-CPU Simulated)")
    print("-" * 50)
    
    try:
        # Import tensor parallelism components
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        from src.tensor_parallel_keras.config_keras import ConfigKeras
        from src.tensor_parallel_keras.state_actions_keras import SplitKeras
        
        # Create base model
        base_model = create_simple_test_model()
        
        # Create tensor parallel config
        config = ConfigKeras(
            state_rules={
                "dense.*": SplitKeras(world_size=2, dim=-1, sharding_type="column_parallel")
            },
            output_rules={
                "dense.*": "gather"
            }
        )
        
        # Create tensor parallel model
        # The config is auto-generated, we just need to specify world_size
        tp_model = TensorParallelKeras(
            model=base_model,
            world_size=2,  # Simulate 2 CPUs
            distributed_backend="jax"
        )
        
        # Create SEPARATE optimizer instance for tensor parallel model
        tp_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        tp_model.compile(
            optimizer=tp_optimizer,
            loss='mse'
        )
        
        print("✅ Tensor parallel model created and compiled")
        
        # CRITICAL: Ensure both models have identical initial weights
        print("🔧 Synchronizing initial weights...")
        base_weights = base_model.get_weights()
        tp_model.set_weights(base_weights)
        print("✅ Initial weights synchronized")
        
        # Create same data (use FIXED seed for reproducibility)
        np.random.seed(42)
        x = np.random.randn(32, 16).astype(np.float32)
        y = np.random.randn(32, 8).astype(np.float32)
        
        # Store initial weights
        initial_weights = [w.numpy() for w in tp_model.weights]
        
        # Single training step
        start_time = time.time()
        loss = tp_model.train_on_batch(x, y)
        training_time = time.time() - start_time
        
        # Get updated weights
        updated_weights = [w.numpy() for w in tp_model.weights]
        
        print(f"✅ Tensor parallel training completed in {training_time:.3f}s")
        print(f"   - Loss: {loss:.6f}")
        print(f"   - Weights updated: {len(updated_weights)}")
        
        return {
            'model': tp_model,
            'loss': loss,
            'initial_weights': initial_weights,
            'updated_weights': updated_weights,
            'training_time': training_time
        }
        
    except Exception as e:
        print(f"❌ Tensor parallel training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(single_results, tp_results):
    """Compare results between single CPU and tensor parallel."""
    print("\n🔍 Comparing Results: Single CPU vs Tensor Parallel")
    print("=" * 60)
    
    if tp_results is None:
        print("❌ Cannot compare: Tensor parallel training failed")
        return False
    
    # Compare losses
    single_loss = single_results['loss']
    tp_loss = tp_results['loss']
    loss_diff = abs(single_loss - tp_loss)
    
    print(f"📊 Loss Comparison:")
    print(f"   - Single CPU: {single_loss:.6f}")
    print(f"   - Tensor Parallel: {tp_loss:.6f}")
    print(f"   - Difference: {loss_diff:.6f}")
    
    # Compare weights
    single_weights = single_results['updated_weights']
    tp_weights = tp_results['updated_weights']
    
    print(f"\n📊 Weight Comparison:")
    print(f"   - Single CPU weights: {len(single_weights)}")
    print(f"   - TP weights: {len(tp_weights)}")
    
    if len(single_weights) != len(tp_weights):
        print("❌ Weight count mismatch!")
        return False
    
    # Compare each weight tensor
    max_diff = 0.0
    mean_diff = 0.0
    total_elements = 0
    
    for i, (single_w, tp_w) in enumerate(zip(single_weights, tp_weights)):
        if single_w.shape != tp_w.shape:
            print(f"❌ Weight {i} shape mismatch: {single_w.shape} vs {tp_w.shape}")
            return False
        
        # Calculate differences
        diff = np.abs(single_w - tp_w)
        max_diff = max(max_diff, np.max(diff))
        mean_diff += np.mean(diff)
        total_elements += single_w.size
        
        print(f"   - Weight {i} ({single_w.shape}): max_diff={np.max(diff):.2e}, mean_diff={np.mean(diff):.2e}")
    
    mean_diff /= len(single_weights)
    
    print(f"\n📊 Overall Weight Statistics:")
    print(f"   - Max difference: {max_diff:.2e}")
    print(f"   - Mean difference: {mean_diff:.2e}")
    print(f"   - Total elements: {total_elements}")
    
    # Determine if results are acceptable
    # For tensor parallelism, we expect very small differences due to floating-point precision
    acceptable_max_diff = 1e-6
    acceptable_mean_diff = 1e-8
    
    if max_diff < acceptable_max_diff and mean_diff < acceptable_mean_diff:
        print(f"\n✅ Numerical correctness PASSED!")
        print(f"   - Differences are within acceptable tolerance")
        print(f"   - Tensor parallelism is working correctly")
        return True
    else:
        print(f"\n❌ Numerical correctness FAILED!")
        print(f"   - Differences exceed acceptable tolerance")
        print(f"   - Max diff: {max_diff:.2e} > {acceptable_max_diff}")
        print(f"   - Mean diff: {mean_diff:.2e} > {acceptable_mean_diff}")
        return False

def main():
    """Main test function."""
    print("🚀 Fast Numerical Correctness Test")
    print("Testing Single CPU vs 2-CPU Tensor Parallel on JAX Backend")
    print("=" * 70)
    
    # Setup
    setup_jax_backend()
    
    # Test single CPU
    single_results = test_single_cpu_training()
    
    # Test tensor parallel with separate optimizer  
    tp_results = test_tensor_parallel_training(single_results['optimizer'])
    
    # Compare results
    success = compare_results(single_results, tp_results)
    
    # Summary
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED! Tensor parallelism is numerically correct.")
        print("🚀 Ready to test with larger models like OPT-125M.")
    else:
        print("💥 TESTS FAILED! Tensor parallelism has numerical issues.")
        print("🔧 Need to debug the implementation before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 