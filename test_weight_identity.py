#!/usr/bin/env python3
"""
Weight Identity Test
Directly compares weights between single CPU and tensor parallel models
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
    """Create a tiny model for weight comparison."""
    inputs = layers.Input(shape=(5,), dtype='int32', name='input_ids')
    embedding = layers.Embedding(50, 8, name='embed_tokens')(inputs)
    outputs = layers.Dense(50, name='lm_head')(embedding)
    model = keras.Model(inputs=inputs, outputs=outputs, name='TinyModel')
    return model

def test_weight_identity():
    """Test if weights are truly identical between models."""
    print("🔍 WEIGHT IDENTITY TEST")
    print("=" * 40)
    
    # Create model with fixed seed
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Create single CPU model
    single_model = create_tiny_model()
    print(f"📊 Single CPU model: {single_model.count_params():,} parameters")
    
    # Get single CPU weights
    single_weights = {}
    for weight in single_model.weights:
        single_weights[weight.name] = np.array(weight)
        print(f"   {weight.name}: {weight.shape} = {weight.numpy().flatten()[:3]}...")
    
    # Create tensor parallel model
    tp_model = create_tiny_model()
    print(f"\n📊 TP model before TensorParallelKeras: {tp_model.count_params():,} parameters")
    
    # Get TP weights before TensorParallelKeras
    tp_weights_before = {}
    for weight in tp_model.weights:
        tp_weights_before[weight.name] = np.array(weight)
        print(f"   {weight.name}: {weight.shape} = {weight.numpy().flatten()[:3]}...")
    
    # Check if weights are identical before TensorParallelKeras
    print(f"\n🔍 Weight comparison BEFORE TensorParallelKeras:")
    weights_identical_before = True
    for name in single_weights:
        if name in tp_weights_before:
            single_w = single_weights[name]
            tp_w = tp_weights_before[name]
            if np.array_equal(single_w, tp_w):
                print(f"   ✅ {name}: IDENTICAL")
            else:
                print(f"   ❌ {name}: DIFFERENT")
                print(f"      Single: {single_w.flatten()[:3]}...")
                print(f"      TP:     {tp_w.flatten()[:3]}...")
                weights_identical_before = False
        else:
            print(f"   ❌ {name}: Missing in TP model")
            weights_identical_before = False
    
    if weights_identical_before:
        print(f"\n✅ Weights are IDENTICAL before TensorParallelKeras")
    else:
        print(f"\n❌ Weights are DIFFERENT before TensorParallelKeras")
        print(f"   This explains the numerical divergence!")
        return False
    
    # Now create TensorParallelKeras
    print(f"\n🔧 Creating TensorParallelKeras...")
    tp_model = TensorParallelKeras(
        model=tp_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    print(f"📊 TP model after TensorParallelKeras: {tp_model.count_params():,} parameters")
    
    # Get TP weights after TensorParallelKeras
    tp_weights_after = {}
    for weight in tp_model.weights:
        tp_weights_after[weight.name] = np.array(weight)
        print(f"   {weight.name}: {weight.shape} = {weight.numpy().flatten()[:3]}...")
    
    # Check if weights are identical after TensorParallelKeras
    print(f"\n🔍 Weight comparison AFTER TensorParallelKeras:")
    weights_identical_after = True
    for name in single_weights:
        if name in tp_weights_after:
            single_w = single_weights[name]
            tp_w = tp_weights_after[name]
            if np.array_equal(single_w, tp_w):
                print(f"   ✅ {name}: IDENTICAL")
            else:
                print(f"   ❌ {name}: DIFFERENT")
                print(f"      Single: {single_w.flatten()[:3]}...")
                print(f"      TP:     {tp_w.flatten()[:3]}...")
                weights_identical_after = False
        else:
            print(f"   ❌ {name}: Missing in TP model")
            weights_identical_after = False
    
    if weights_identical_after:
        print(f"\n✅ Weights are IDENTICAL after TensorParallelKeras")
        print(f"   Mathematical identity should be preserved!")
        return True
    else:
        print(f"\n❌ Weights are DIFFERENT after TensorParallelKeras")
        print(f"   This is the source of numerical divergence!")
        return False

if __name__ == "__main__":
    success = test_weight_identity()
    
    if success:
        print(f"\n🎯 SUCCESS: Weight identity verified!")
        print(f"   Tensor parallelism should now produce identical results.")
    else:
        print(f"\n🚨 FAILED: Weight identity not preserved!")
        print(f"   Need to fix the weight creation issue.") 