#!/usr/bin/env python3
"""
Single Training Step Debug Test
Tests just one training step to pinpoint where numerical divergence occurs
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
    """Create a tiny model for single step debugging."""
    inputs = layers.Input(shape=(10,), dtype='int32', name='input_ids')
    embedding = layers.Embedding(100, 16, name='embed_tokens')(inputs)
    outputs = layers.Dense(100, name='lm_head')(embedding)
    model = keras.Model(inputs=inputs, outputs=outputs, name='TinyModel')
    return model

def copy_weights(source_model, target_model):
    """Copy weights from source model to target model."""
    for source_weight, target_weight in zip(source_model.weights, target_model.weights):
        if source_weight.shape == target_weight.shape:
            target_weight.assign(source_weight.numpy())

def test_single_step_debug():
    """Debug single training step to find divergence point."""
    print("🔍 SINGLE TRAINING STEP DEBUG TEST")
    print("=" * 50)
    
    # Create tiny training data
    np.random.seed(42)
    x_train = np.random.randint(0, 100, (2, 10), dtype=np.int32)
    y_train = np.random.randint(0, 100, (2, 10), dtype=np.int32)
    
    print(f"📊 Training data: x={x_train.shape}, y={y_train.shape}")
    print(f"   Input sample: {x_train[0, :5]}...")
    print(f"   Target sample: {y_train[0, :5]}...")
    
    # Step 1: Single CPU single step
    print(f"\n⏱️  Step 1: Single CPU single training step...")
    
    single_model = create_tiny_model()
    single_optimizer = optimizers.Adam(learning_rate=0.001)
    
    single_model.compile(
        optimizer=single_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"   Model: {single_model.count_params():,} parameters")
    
    # Get initial weights
    single_initial_weights = {}
    for weight in single_model.weights:
        single_initial_weights[weight.name] = np.array(weight)
    
    print(f"   Initial weights captured: {len(single_initial_weights)}")
    
    # Single training step
    single_result = single_model.train_on_batch(x_train, y_train)
    try:
        # Try as array first
        single_loss = single_result[0]
    except (IndexError, TypeError):
        # If that fails, try as scalar
        single_loss = float(single_result)
    
    # Get final weights
    single_final_weights = {}
    for weight in single_model.weights:
        single_final_weights[weight.name] = np.array(weight)
    
    print(f"   ✅ Single CPU step completed")
    print(f"   Loss: {single_loss:.6f}")
    print(f"   Final weights: {len(single_final_weights)}")
    
    # Step 2: TP single step
    print(f"\n⏱️  Step 2: Tensor Parallel single training step...")
    
    tp_model = create_tiny_model()
    
    # CRITICAL FIX: Copy weights from single model to ensure mathematical identity
    copy_weights(single_model, tp_model)
    
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
    
    # Get initial weights
    tp_initial_weights = {}
    for weight in tp_model.weights:
        tp_initial_weights[weight.name] = np.array(weight)
    
    print(f"   Initial weights captured: {len(tp_initial_weights)}")
    
    # Single training step
    tp_result = tp_model.train_on_batch(x_train, y_train)
    try:
        # Try as array first
        tp_loss = tp_result[0]
    except (IndexError, TypeError):
        # If that fails, try as scalar
        tp_loss = float(tp_result)
    
    # Get final weights
    tp_final_weights = {}
    for weight in tp_model.weights:
        tp_final_weights[weight.name] = np.array(weight)
    
    print(f"   ✅ TP step completed")
    print(f"   Loss: {tp_loss:.6f}")
    print(f"   Final weights: {len(tp_final_weights)}")
    
    # Step 3: Detailed comparison
    print(f"\n⏱️  Step 3: Detailed comparison...")
    
    # Loss comparison
    loss_diff = abs(single_loss - tp_loss)
    print(f"   Loss comparison:")
    print(f"     Single CPU: {single_loss:.6f}")
    print(f"     TP Model:   {tp_loss:.6f}")
    print(f"     Difference: {loss_diff:.2e}")
    
    # Initial weights comparison
    print(f"\n   Initial weights comparison:")
    for name in single_initial_weights:
        if name in tp_initial_weights:
            single_init = single_initial_weights[name]
            tp_init = tp_initial_weights[name]
            
            if single_init.shape == tp_init.shape:
                diff = np.abs(single_init - tp_init)
                max_diff = np.max(diff)
                print(f"     {name}: shapes match, max diff: {max_diff:.2e}")
            else:
                print(f"     {name}: shape mismatch {single_init.shape} vs {tp_init.shape}")
        else:
            print(f"     {name}: missing in TP model")
    
    # Final weights comparison
    print(f"\n   Final weights comparison:")
    for name in single_final_weights:
        if name in tp_final_weights:
            single_final = single_final_weights[name]
            tp_final = tp_final_weights[name]
            
            if single_final.shape == tp_final.shape:
                diff = np.abs(single_final - tp_final)
                max_diff = np.max(diff)
                print(f"     {name}: shapes match, max diff: {max_diff:.2e}")
            else:
                print(f"     {name}: shape mismatch {single_final.shape} vs {tp_final.shape}")
        else:
            print(f"     {name}: missing in TP model")
    
    # Weight update comparison
    print(f"\n   Weight update comparison:")
    for name in single_final_weights:
        if name in single_initial_weights and name in tp_final_weights:
            single_update = single_final_weights[name] - single_initial_weights[name]
            tp_update = tp_final_weights[name] - tp_initial_weights[name]
            
            if single_update.shape == tp_update.shape:
                diff = np.abs(single_update - tp_update)
                max_diff = np.max(diff)
                print(f"     {name}: update shapes match, max diff: {max_diff:.2e}")
            else:
                print(f"     {name}: update shape mismatch {single_update.shape} vs {tp_update.shape}")
    
    return loss_diff < 1e-4

if __name__ == "__main__":
    success = test_single_step_debug()
    
    if success:
        print(f"\n✅ SUCCESS: Single step consistency verified!")
    else:
        print(f"\n❌ FAILED: Single step shows numerical divergence!")
        print("   This indicates the tensor parallelism implementation needs fixing.") 