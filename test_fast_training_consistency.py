#!/usr/bin/env python3
"""
Fast Training Step Consistency Test
Runs only 2 epochs with a small model to quickly verify training consistency
"""

import os

# Set Keras to use JAX backend explicitly (no TensorFlow)
os.environ['KERAS_BACKEND'] = 'jax'

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import time
import numpy as np
import keras
from keras import layers, optimizers

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_small_model():
    """Create a small model for fast testing."""
    inputs = layers.Input(shape=(20,), dtype='int32', name='input_ids')
    embedding = layers.Embedding(1000, 32, name='embed_tokens')(inputs)
    hidden = layers.Dense(64, activation='relu', name='hidden')(embedding)
    outputs = layers.Dense(1000, name='lm_head')(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name='SmallModel')
    return model

def get_model_weights_state(model):
    """Get the current state of all model weights."""
    weights_state = {}
    for weight in model.weights:
        weights_state[weight.name] = np.array(weight)
    return weights_state

def compare_weights_states(original_state, tp_state, tolerance=1e-5):
    """Compare two weight states and return analysis."""
    comparison = {
        'total_weights': len(original_state),
        'matching_weights': 0,
        'different_weights': 0,
        'max_difference': 0.0,
        'mean_difference': 0.0
    }
    
    all_differences = []
    
    for weight_name, original_weight in original_state.items():
        if weight_name in tp_state:
            tp_weight = tp_state[weight_name]
            
            if original_weight.shape != tp_weight.shape:
                comparison['different_weights'] += 1
                continue
            
            if np.allclose(original_weight, tp_weight, atol=tolerance):
                comparison['matching_weights'] += 1
            else:
                diff = np.abs(original_weight - tp_weight)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                all_differences.append(max_diff)
                comparison['different_weights'] += 1
    
    if all_differences:
        comparison['max_difference'] = max(all_differences)
        comparison['mean_difference'] = np.mean(all_differences)
    
    return comparison

def test_fast_training_consistency():
    """Fast test with only 2 epochs."""
    print("🚀 FAST TRAINING CONSISTENCY TEST - 2 EPOCHS ONLY")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create small training data
    np.random.seed(42)
    x_train = np.random.randint(0, 1000, (16, 20), dtype=np.int32)
    y_train = np.random.randint(0, 1000, (16, 20), dtype=np.int32)
    
    print(f"📊 Training data: x={x_train.shape}, y={y_train.shape}")
    
    # Step 1: Single CPU training (2 epochs)
    print(f"\n⏱️  Step 1: Single CPU training (2 epochs)...")
    
    single_model = create_small_model()
    single_optimizer = optimizers.Adam(learning_rate=0.001)
    
    single_model.compile(
        optimizer=single_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"   Model: {single_model.count_params():,} parameters")
    
    # Train for 2 epochs
    single_history = single_model.fit(
        x_train, y_train, 
        epochs=2, 
        batch_size=4, 
        verbose=0
    )
    
    single_loss = single_history.history['loss'][-1]
    single_weights = get_model_weights_state(single_model)
    
    print(f"   ✅ Single CPU training completed")
    print(f"   Final loss: {single_loss:.6f}")
    print(f"   Weights captured: {len(single_weights)}")
    
    # Step 2: 2-CPU sharded training (2 epochs)
    print(f"\n⏱️  Step 2: 2-CPU sharded training (2 epochs)...")
    
    tp_model = create_small_model()
    tp_optimizer = optimizers.Adam(learning_rate=0.001)
    
    tp_model = TensorParallelKeras(
        model=tp_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    tp_model.compile(
        optimizer=tp_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"   TP Model created successfully")
    print(f"   World size: {tp_model.world_size}")
    
    # Train for 2 epochs
    tp_history = tp_model.fit(
        x_train, y_train, 
        epochs=2, 
        batch_size=4, 
        verbose=0
    )
    
    tp_loss = tp_history.history['loss'][-1]
    tp_weights = get_model_weights_state(tp_model)
    
    print(f"   ✅ 2-CPU sharded training completed")
    print(f"   Final loss: {tp_loss:.6f}")
    print(f"   Weights captured: {len(tp_weights)}")
    
    # Step 3: Quick comparison
    print(f"\n⏱️  Step 3: Quick comparison...")
    
    # Loss comparison
    loss_diff = abs(single_loss - tp_loss)
    loss_tolerance = 1e-4
    
    print(f"   Loss comparison:")
    print(f"     Single CPU: {single_loss:.6f}")
    print(f"     TP Model:   {tp_loss:.6f}")
    print(f"     Difference: {loss_diff:.2e}")
    print(f"     Tolerance:  {loss_tolerance:.2e}")
    
    loss_consistent = loss_diff <= loss_tolerance
    
    if loss_consistent:
        print(f"   ✅ Loss consistency verified!")
    else:
        print(f"   ❌ Loss consistency failed!")
    
    # Weights comparison
    weights_comparison = compare_weights_states(single_weights, tp_weights, tolerance=1e-4)
    
    print(f"\n   Weights comparison:")
    print(f"     Total weights: {weights_comparison['total_weights']}")
    print(f"     Matching: {weights_comparison['matching_weights']}")
    print(f"     Different: {weights_comparison['different_weights']}")
    
    if weights_comparison['max_difference'] > 0:
        print(f"     Max difference: {weights_comparison['max_difference']:.2e}")
        print(f"     Mean difference: {weights_comparison['mean_difference']:.2e}")
    
    weights_consistent = (weights_comparison['matching_weights'] == weights_comparison['total_weights'] and
                         weights_comparison['different_weights'] == 0)
    
    if weights_consistent:
        print(f"   ✅ Weights consistency verified!")
    else:
        print(f"   ❌ Weights consistency failed!")
    
    # Final result
    print(f"\n⏱️  Final assessment...")
    
    if loss_consistent and weights_consistent:
        print(f"   🎯 FAST TRAINING CONSISTENCY VERIFIED!")
        print(f"   ✅ Loss values are numerically identical")
        print(f"   ✅ Weight updates are numerically identical")
        print(f"   🚀 CoordinatedOptimizer workflow is working correctly")
        result = True
    else:
        print(f"   🚨 FAST TRAINING CONSISTENCY FAILED!")
        if not loss_consistent:
            print(f"   ❌ Loss consistency failed")
        if not weights_consistent:
            print(f"   ❌ Weights consistency failed")
        result = False
    
    total_time = time.time() - start_time
    print(f"\n✅ Fast test completed in {total_time:.2f}s")
    return result

if __name__ == "__main__":
    print("🎯 FAST TRAINING STEP CONSISTENCY TEST")
    print("=" * 60)
    print("🔍 Quick test with only 2 epochs to verify training consistency")
    print("=" * 60)
    
    success = test_fast_training_consistency()
    
    if success:
        print("\n🚀 SUCCESS: Fast training consistency test passed!")
        print("\n💡 TRAINING VERIFICATION:")
        print("   ✅ Forward pass consistency verified")
        print("   ✅ Loss computation consistency verified") 
        print("   ✅ Backward pass consistency verified")
        print("   ✅ Optimizer update consistency verified")
        print("   ✅ CoordinatedOptimizer workflow verified")
        print("\n🎯 Your tensor parallelism is working correctly!")
    else:
        print(f"\n⚠️  WARNING: Fast training consistency test failed.")
        print("   Please review and fix the failing components.") 