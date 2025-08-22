#!/usr/bin/env python3
"""
Full Training Identity Test
Tests complete training step identity with shared optimizer
"""

import os

# Set Keras to use JAX backend explicitly (no TensorFlow)
os.environ['KERAS_BACKEND'] = 'jax'

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import numpy as np
import keras
from keras import layers, optimizers

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_tiny_model():
    """Create a tiny model for training comparison."""
    inputs = layers.Input(shape=(5,), dtype='int32', name='input_ids')
    embedding = layers.Embedding(50, 8, name='embed_tokens')(inputs)
    outputs = layers.Dense(50, name='lm_head')(embedding)
    model = keras.Model(inputs=inputs, outputs=outputs, name='TinyModel')
    return model

def copy_weights(source_model, target_model):
    """Copy weights from source model to target model."""
    for source_weight, target_weight in zip(source_model.weights, target_model.weights):
        if source_weight.shape == target_weight.shape:
            target_weight.assign(source_weight.numpy())

def test_full_training_identity():
    """Test full training step identity."""
    print("🎯 FULL TRAINING IDENTITY TEST")
    print("=" * 50)
    
    # Create model with fixed seed
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Create single CPU model
    single_model = create_tiny_model()
    single_optimizer = optimizers.Adam(learning_rate=0.001)
    
    single_model.compile(
        optimizer=single_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"📊 Single CPU model: {single_model.count_params():,} parameters")
    
    # Create tensor parallel model
    tp_model = create_tiny_model()
    
    # CRITICAL FIX: Copy weights from single model to ensure mathematical identity
    copy_weights(single_model, tp_model)
    
    # CRITICAL FIX: Use the EXACT SAME optimizer instance
    tp_optimizer = single_optimizer  # Same instance, not a copy
    
    tp_model.compile(
        optimizer=tp_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"📊 TP model after weight copy: {tp_model.count_params():,} parameters")
    
    # Create TensorParallelKeras
    tp_model = TensorParallelKeras(
        model=tp_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # CRITICAL: Compile the TensorParallelKeras model
    tp_model.compile(
        optimizer=tp_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"📊 TP model after TensorParallelKeras: {tp_model.count_params():,} parameters")
    
    # Create training data
    x_train = np.random.randint(0, 50, (2, 5), dtype=np.int32)
    y_train = np.random.randint(0, 50, (2, 5), dtype=np.int32)
    
    print(f"\n📊 Training data: x={x_train.shape}, y={y_train.shape}")
    
    # Get initial weights from both models
    print(f"\n🔍 Initial weight comparison:")
    single_initial = {}
    tp_initial = {}
    
    for weight in single_model.weights:
        single_initial[weight.name] = np.array(weight)
    
    for weight in tp_model.weights:
        tp_initial[weight.name] = np.array(weight)
    
    weights_identical = True
    for name in single_initial:
        if name in tp_initial:
            single_w = single_initial[name]
            tp_w = tp_initial[name]
            if np.array_equal(single_w, tp_w):
                print(f"   ✅ {name}: IDENTICAL")
            else:
                print(f"   ❌ {name}: DIFFERENT")
                weights_identical = False
        else:
            print(f"   ❌ {name}: Missing in TP model")
            weights_identical = False
    
    if not weights_identical:
        print(f"\n❌ Initial weights are not identical!")
        return False
    
    print(f"\n✅ Initial weights are identical")
    
    # Single training step on single CPU model
    print(f"\n⏱️  Single CPU training step...")
    single_result = single_model.train_on_batch(x_train, y_train)
    try:
        # Try as array first
        single_loss = single_result[0]
    except (IndexError, TypeError):
        # If that fails, try as scalar
        single_loss = float(single_result)
    
    # Get final weights from single CPU model
    single_final = {}
    for weight in single_model.weights:
        single_final[weight.name] = np.array(weight)
    
    print(f"   ✅ Single CPU training completed")
    print(f"   Loss: {single_loss:.6f}")
    
    # Single training step on tensor parallel model
    print(f"\n⏱️  Tensor Parallel training step...")
    tp_result = tp_model.train_on_batch(x_train, y_train)
    try:
        # Try as array first
        tp_loss = tp_result[0]
    except (IndexError, TypeError):
        # If that fails, try as scalar
        tp_loss = float(tp_result)
    
    # Get final weights from tensor parallel model
    tp_final = {}
    for weight in tp_model.weights:
        tp_final[weight.name] = np.array(weight)
    
    print(f"   ✅ TP training completed")
    print(f"   Loss: {tp_loss:.6f}")
    
    # Compare results
    print(f"\n🔍 Training step comparison:")
    
    # Loss comparison
    loss_diff = abs(single_loss - tp_loss)
    loss_tolerance = 1e-6  # Very strict tolerance for mathematical identity
    
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
    
    # Weight comparison
    print(f"\n   Final weight comparison:")
    weights_consistent = True
    
    for name in single_final:
        if name in tp_final:
            single_w = single_final[name]
            tp_w = tp_final[name]
            
            if single_w.shape == tp_w.shape:
                diff = np.abs(single_w - tp_w)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                if max_diff <= 1e-6:  # Very strict tolerance
                    print(f"     ✅ {name}: IDENTICAL (max diff: {max_diff:.2e})")
                else:
                    print(f"     ❌ {name}: DIFFERENT (max diff: {max_diff:.2e}, mean: {mean_diff:.2e})")
                    weights_consistent = False
            else:
                print(f"     ❌ {name}: Shape mismatch {single_w.shape} vs {tp_w.shape}")
                weights_consistent = False
        else:
            print(f"     ❌ {name}: Missing in TP model")
            weights_consistent = False
    
    # Final assessment
    print(f"\n⏱️  Final assessment...")
    
    if loss_consistent and weights_consistent:
        print(f"   🎯 FULL TRAINING IDENTITY VERIFIED!")
        print(f"   ✅ Loss values are numerically identical")
        print(f"   ✅ Weight updates are numerically identical")
        print(f"   🚀 CoordinatedOptimizer workflow is working correctly")
        return True
    else:
        print(f"   🚨 FULL TRAINING IDENTITY FAILED!")
        if not loss_consistent:
            print(f"   ❌ Loss consistency failed")
        if not weights_consistent:
            print(f"   ❌ Weights consistency failed")
        return False

if __name__ == "__main__":
    success = test_full_training_identity()
    
    if success:
        print(f"\n🚀 SUCCESS: Full training identity verified!")
        print(f"\n💡 TRAINING VERIFICATION:")
        print(f"   ✅ Forward pass consistency verified")
        print(f"   ✅ Loss computation consistency verified") 
        print(f"   ✅ Backward pass consistency verified")
        print(f"   ✅ Optimizer update consistency verified")
        print(f"   ✅ CoordinatedOptimizer workflow verified")
        print(f"\n🎯 Your tensor parallelism is working correctly!")
    else:
        print(f"\n⚠️  WARNING: Full training identity test failed.")
        print(f"   Please review and fix the failing components.") 