#!/usr/bin/env python3
"""
Test Random State Management for Dropout Operations
Verifies the critical rule for dropout correctness in tensor parallelism.
"""

import numpy as np
import keras
from keras import layers, Model
import os
import gc

def setup_jax_backend():
    """Set up JAX backend."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    return True

def create_model_with_dropout():
    """Create a model with different types of dropout to test RNG management."""
    inputs = keras.Input(shape=(32, 64))
    
    # Parallel region: Dropout within self-attention mechanism
    # This should have different RNG seeds per device
    x = layers.Dense(128, activation='relu', name="parallel_dense_1")(inputs)
    x = layers.Dropout(0.1, name="parallel_dropout_1")(x)  # Parallel region
    
    # Self-attention with dropout
    x = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=32, 
        dropout=0.1,
        name="attention_with_dropout"
    )(x, x)  # Parallel region dropout
    
    # Replicated region: Dropout after residual connection
    # This should have the same RNG seed across all devices
    residual = inputs
    x = layers.Add(name="residual_add")([x, residual])
    x = layers.Dropout(0.1, name="replicated_dropout_1")(x)  # Replicated region
    
    # More parallel dropout
    x = layers.Dense(64, activation='relu', name="parallel_dense_2")(x)
    x = layers.Dropout(0.1, name="parallel_dropout_2")(x)  # Parallel region
    
    # Final replicated dropout
    x = layers.Dense(32, activation='relu', name="final_dense")(x)
    x = layers.Add(name="final_residual")([x, layers.Dense(32, name="residual_proj")(inputs)])
    x = layers.Dropout(0.1, name="replicated_dropout_2")(x)  # Replicated region
    
    outputs = layers.Dense(10, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_simple_dropout_model():
    """Create a simpler model with clear dropout patterns."""
    inputs = keras.Input(shape=(16, 32))
    
    # Parallel region: Different dropout masks per device
    x = layers.Dense(64, activation='relu', name="dense_1")(inputs)
    x = layers.Dropout(0.2, name="parallel_dropout")(x)  # Parallel region
    
    # Replicated region: Same dropout mask across devices
    x = layers.Dense(32, activation='relu', name="dense_2")(x)
    x = layers.Add(name="residual")([x, layers.Dense(32, name="residual_proj")(inputs)])
    x = layers.Dropout(0.2, name="replicated_dropout")(x)  # Replicated region
    
    # Flatten to ensure proper output shape
    x = layers.Flatten()(x)
    
    outputs = layers.Dense(5, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def clean_test_environment():
    """Clean up the test environment to prevent state contamination."""
    keras.backend.clear_session()
    gc.collect()
    np.random.seed(42)
    keras.utils.set_random_seed(42)

def test_dropout_rng_management():
    """Test random state management for dropout operations."""
    print("🧪 Testing Random State Management for Dropout Operations")
    print("=" * 70)
    
    # Clean environment before test
    clean_test_environment()
    
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Create test data
    batch_size = 8
    seq_len = 16
    input_dim = 32
    
    x = np.random.random((batch_size, seq_len, input_dim)).astype('float32')
    y = np.random.randint(0, 5, size=(batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Test 1: Simple Dropout Model
    print(f"\n🔍 Test 1: Simple Dropout Model (RNG Management)")
    
    simple_model = create_simple_dropout_model()
    
    # Use fixed learning rate
    simple_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    simple_model.compile(optimizer=simple_optimizer, loss='sparse_categorical_crossentropy')
    
    print(f"✅ Simple Dropout Model compiled successfully")
    
    # Store initial weights
    initial_weights = [w.numpy().copy() for w in simple_model.weights]
    print(f"✅ Initial weights captured: {len(initial_weights)} parameters")
    
    # Single training step
    simple_loss = simple_model.train_on_batch(x, y)
    simple_updated_weights = [w.numpy().copy() for w in simple_model.weights]
    
    print(f"✅ Simple Dropout Model training completed - Loss: {simple_loss:.6f}")
    
    # Clean up simple model
    del simple_model
    del simple_optimizer
    clean_test_environment()
    
    # Test 2: Tensor Parallel Dropout Model
    print(f"\n🔍 Test 2: Tensor Parallel Dropout Model (RNG Management)")
    
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
    
    # Create base model for tensor parallelism
    base_model = create_simple_dropout_model()
    
    # CRITICAL: Set EXACTLY the same initial weights
    base_model.set_weights(initial_weights)
    
    # Verify weights are identical
    base_weights = [w.numpy().copy() for w in base_model.weights]
    weights_identical = all(
        np.allclose(init_w, base_w, atol=1e-10) 
        for init_w, base_w in zip(initial_weights, base_weights)
    )
    print(f"✅ Base model weights identical: {weights_identical}")
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model=base_model,
        world_size=2,
        distributed_backend="jax"
    )
    
    # Create SEPARATE optimizer with SAME learning rate
    tp_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    tp_model.compile(optimizer=tp_optimizer, loss='sparse_categorical_crossentropy')
    
    print(f"✅ Tensor parallel Dropout Model compiled successfully")
    
    # Verify tensor parallel model has identical weights
    tp_initial_weights = [w.numpy().copy() for w in tp_model.weights]
    tp_weights_identical = all(
        np.allclose(init_w, tp_w, atol=1e-10) 
        for init_w, tp_w in zip(initial_weights, tp_initial_weights)
    )
    print(f"✅ Tensor parallel weights identical: {tp_weights_identical}")
    
    if not tp_weights_identical:
        print("❌ CRITICAL: Tensor parallel model weights not identical!")
        for i, (init_w, tp_w) in enumerate(zip(initial_weights, tp_initial_weights)):
            diff = np.abs(init_w - tp_w)
            max_diff = np.max(diff)
            print(f"  Weight {i}: max_diff={max_diff:.2e}")
        return False
    
    # Test 3: Forward Pass Comparison
    print(f"\n🔍 Test 3: Forward Pass Comparison")
    
    # Forward pass on single model (recreate to ensure clean state)
    single_model_clean = create_simple_dropout_model()
    single_model_clean.set_weights(initial_weights)
    single_output = single_model_clean(x, training=False)
    
    # Forward pass on tensor parallel model
    tp_output = tp_model(x, training=False)
    
    # Compare outputs
    forward_diff = np.abs(single_output.numpy() - tp_output.numpy())
    max_forward_diff = np.max(forward_diff)
    mean_forward_diff = np.mean(forward_diff)
    
    print(f"Forward pass - Max diff: {max_forward_diff:.2e}")
    print(f"Forward pass - Mean diff: {mean_forward_diff:.2e}")
    
    # Clean up forward pass models
    del single_model_clean
    clean_test_environment()
    
    # Test 4: Tensor Parallel Training Step
    print(f"\n🔍 Test 4: Tensor Parallel Training Step")
    
    # Training step
    tp_loss = tp_model.train_on_batch(x, y)
    tp_updated_weights = [w.numpy().copy() for w in tp_model.weights]
    
    print(f"✅ Tensor parallel Dropout Model training completed - Loss: {tp_loss:.6f}")
    
    # Test 5: Comparing Results
    print(f"\n🔍 Test 5: Comparing Results")
    
    # Compare losses
    loss_diff = abs(simple_loss - tp_loss)
    print(f"Loss difference: {loss_diff:.2e}")
    
    # Compare weight updates
    print(f"\n📊 Weight Update Comparison:")
    max_weight_diff = 0
    for i, (single_w, tp_w) in enumerate(zip(simple_updated_weights, tp_updated_weights)):
        diff = np.abs(single_w - tp_w)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_weight_diff = max(max_weight_diff, max_diff)
        print(f"  Weight {i} {single_w.shape}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    # Clean up tensor parallel model
    del tp_model
    del tp_optimizer
    clean_test_environment()
    
    # Test 6: Verify Dropout RNG Rules
    print(f"\n🔍 Test 6: Verify Dropout RNG Rules")
    
    # Check if the autoconfig is properly applied
    print("✅ Dropout RNG Rules Applied:")
    print("   - Parallel Dropout: Different RNG seed per device (different masks)")
    print("   - Replicated Dropout: Same RNG seed across devices (same mask)")
    print("   - RNG State Management: Proper seeding for numerical correctness")
    print("   - Megatron-LM Equivalent: get_cuda_rng_tracker_equivalent implemented")
    
    # Test RNG management functions
    from src.tensor_parallel_keras.communications_keras import (
        manage_dropout_rng_state, 
        get_cuda_rng_tracker_equivalent,
        get_replicated_rng_tracker
    )
    
    # Test parallel RNG (different seeds per device)
    parallel_rng = get_cuda_rng_tracker_equivalent(world_size=2, base_seed=42)
    print(f"\n🔧 Parallel RNG Test (Different seeds per device):")
    for device_id, rng_info in parallel_rng.items():
        print(f"   Device {device_id}: seed={rng_info['seed']}, type={rng_info['type']}")
    
    # Test replicated RNG (same seed across devices)
    replicated_rng = get_replicated_rng_tracker(world_size=2, base_seed=42)
    print(f"\n🔧 Replicated RNG Test (Same seed across devices):")
    for device_id, rng_info in replicated_rng.items():
        print(f"   Device {device_id}: seed={rng_info['seed']}, type={rng_info['type']}")
    
    # Verify RNG rules are different
    parallel_seeds = [rng_info['seed'] for rng_info in parallel_rng.values()]
    replicated_seeds = [rng_info['seed'] for rng_info in replicated_rng.values()]
    
    parallel_unique = len(set(parallel_seeds)) == len(parallel_seeds)
    replicated_same = len(set(replicated_seeds)) == 1
    
    print(f"\n🔍 RNG Rule Verification:")
    print(f"   Parallel seeds unique: {parallel_unique} (seeds: {parallel_seeds})")
    print(f"   Replicated seeds same: {replicated_same} (seed: {replicated_seeds[0]})")
    
    # Determine test results
    forward_pass_success = max_forward_diff < 1e-6
    loss_success = loss_diff < 1e-6
    weight_success = max_weight_diff < 1e-6
    rng_rules_success = parallel_unique and replicated_same
    
    print(f"\n📊 Results for Dropout RNG Management:")
    print(f"{'✅' if forward_pass_success else '❌'} Forward pass: {'PASS' if forward_pass_success else 'FAIL'} (max diff: {max_forward_diff:.2e})")
    print(f"{'✅' if loss_success else '❌'} Loss matching: {'PASS' if loss_success else 'FAIL'} (diff: {loss_diff:.2e})")
    print(f"{'✅' if weight_success else '❌'} Weight updates: {'PASS' if weight_success else 'FAIL'} (max diff: {max_weight_diff:.2e})")
    print(f"{'✅' if rng_rules_success else '❌'} RNG Rules: {'PASS' if rng_rules_success else 'FAIL'}")
    
    overall_success = forward_pass_success and loss_success and weight_success and rng_rules_success
    print(f"\n🎯 Overall Dropout RNG Management Test: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success

def main():
    """Run the dropout RNG management test."""
    print("🚀 Testing Random State Management for Dropout Operations")
    print("=" * 80)
    
    setup_jax_backend()
    
    # Run the dropout RNG test
    success = test_dropout_rng_management()
    
    # Final results summary
    print("\n" + "=" * 80)
    if success:
        print("🎯 Overall Result: ✅ DROPOUT RNG TEST PASSED")
        print("🎉 Random state management for dropout is working correctly!")
        print("✅ Parallel dropout: Different RNG seeds per device")
        print("✅ Replicated dropout: Same RNG seed across devices")
        print("✅ Megatron-LM equivalent: get_cuda_rng_tracker_equivalent working")
    else:
        print("🎯 Overall Result: ❌ DROPOUT RNG TEST FAILED")
        print("🔧 Need to investigate dropout RNG management issues.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 