#!/usr/bin/env python3
"""
Forward Pass Identity Test
Compares forward pass results between single CPU and tensor parallel models
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
    """Create a tiny model for forward pass comparison."""
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

def test_forward_pass_identity():
    """Test forward pass identity between models."""
    print("🔍 FORWARD PASS IDENTITY TEST")
    print("=" * 40)
    
    # Create model with fixed seed
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Create single CPU model
    single_model = create_tiny_model()
    print(f"📊 Single CPU model: {single_model.count_params():,} parameters")
    
    # Create tensor parallel model
    tp_model = create_tiny_model()
    
    # CRITICAL FIX: Copy weights from single model to ensure mathematical identity
    copy_weights(single_model, tp_model)
    
    print(f"📊 TP model after weight copy: {tp_model.count_params():,} parameters")
    
    # Create TensorParallelKeras
    tp_model = TensorParallelKeras(
        model=tp_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    print(f"📊 TP model after TensorParallelKeras: {tp_model.count_params():,} parameters")
    
    # Create test input
    test_input = np.random.randint(0, 50, (2, 5), dtype=np.int32)
    print(f"\n📊 Test input: {test_input.shape}")
    print(f"   Sample: {test_input[0]}")
    
    # Forward pass on single CPU model
    print(f"\n⏱️  Single CPU forward pass...")
    single_output = single_model(test_input, training=False)
    single_output_np = np.array(single_output)
    print(f"   Output shape: {single_output_np.shape}")
    print(f"   Output sample: {single_output_np[0, 0, :5]}")
    
    # Forward pass on tensor parallel model
    print(f"\n⏱️  Tensor Parallel forward pass...")
    tp_output = tp_model(test_input, training=False)
    tp_output_np = np.array(tp_output)
    print(f"   Output shape: {tp_output_np.shape}")
    print(f"   Output sample: {tp_output_np[0, 0, :5]}")
    
    # Compare outputs
    print(f"\n🔍 Output comparison:")
    
    if single_output_np.shape == tp_output_np.shape:
        print(f"   ✅ Shapes match: {single_output_np.shape}")
        
        # Check if outputs are identical
        if np.array_equal(single_output_np, tp_output_np):
            print(f"   ✅ Outputs are IDENTICAL!")
            print(f"   🎯 Forward pass mathematical identity verified!")
            return True
        else:
            # Check numerical difference
            diff = np.abs(single_output_np - tp_output_np)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"   ❌ Outputs are DIFFERENT!")
            print(f"      Max difference: {max_diff:.2e}")
            print(f"      Mean difference: {mean_diff:.2e}")
            
            # Show specific differences
            print(f"      Single CPU sample: {single_output_np[0, 0, :5]}")
            print(f"      TP sample:         {tp_output_np[0, 0, :5]}")
            print(f"      Sample diff:       {np.abs(single_output_np[0, 0, :5] - tp_output_np[0, 0, :5])}")
            
            return False
    else:
        print(f"   ❌ Shape mismatch: {single_output_np.shape} vs {tp_output_np.shape}")
        return False

if __name__ == "__main__":
    success = test_forward_pass_identity()
    
    if success:
        print(f"\n🎯 SUCCESS: Forward pass identity verified!")
        print(f"   Tensor parallelism produces identical forward pass results.")
    else:
        print(f"\n🚨 FAILED: Forward pass identity not preserved!")
        print(f"   Need to fix the forward pass computation.") 