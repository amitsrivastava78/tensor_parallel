#!/usr/bin/env python3
"""
Debug script to investigate parameter count differences across shards.
"""

import os
import numpy as np
import keras
from keras import layers, Model

def setup_real_jax_environment():
    """Set up real JAX multi-device environment."""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    import jax
    device_count = jax.device_count()
    print(f"✅ JAX devices: {device_count}")
    return device_count >= 2

def create_test_model():
    """Create a simple test model."""
    inputs = keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu', name="dense_1")(inputs)
    x = layers.Dense(32, activation='relu', name="dense_2")(x)
    outputs = layers.Dense(10, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def debug_parameter_sharding():
    """Debug the parameter sharding process."""
    print("🔍 Debugging Parameter Sharding Process")
    
    # Create model
    model = create_test_model()
    print(f"✅ Model created with {len(model.weights)} weights")
    
    # Print original weights
    print("\n📊 Original Model Weights:")
    for i, weight in enumerate(model.weights):
        print(f"   Weight {i}: {weight.name} - Shape: {weight.shape} - Params: {np.prod(weight.shape):,}")
    
    # Create tensor parallel model
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
    
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend="jax"
    )
    
    print(f"\n📊 Tensor Parallel Model Analysis:")
    print(f"   Number of shards: {len(tp_model.model_shards)}")
    
    # Analyze each shard
    for i, shard in enumerate(tp_model.model_shards):
        print(f"\n🔍 Shard {i} Analysis:")
        print(f"   Shard type: {type(shard)}")
        print(f"   Has sharding_strategy: {hasattr(shard, 'sharding_strategy')}")
        
        if hasattr(shard, 'sharding_strategy'):
            strategy = shard.sharding_strategy
            print(f"   Sharding strategy type: {type(strategy)}")
            print(f"   World size: {strategy.world_size}")
            print(f"   Rank: {strategy.rank}")
            print(f"   Sharded weights count: {len(strategy.sharded_weights)}")
            
            # Print sharded weights
            print(f"   📊 Sharded Weights:")
            for name, weight in strategy.sharded_weights.items():
                if hasattr(weight, 'shape'):
                    print(f"     {name}: {weight.shape} - Params: {np.prod(weight.shape):,}")
                else:
                    print(f"     {name}: {type(weight)} - No shape")
        
        # Print shard weights
        print(f"   📊 Shard Weights (via .weights property):")
        for j, weight in enumerate(shard.weights):
            if hasattr(weight, 'shape'):
                print(f"     Weight {j}: {weight.name} - Shape: {weight.shape} - Params: {np.prod(weight.shape):,}")
            else:
                print(f"     Weight {j}: {type(weight)} - No shape")
        
        # Count total parameters
        total_params = 0
        for weight in shard.weights:
            if hasattr(weight, 'shape'):
                total_params += np.prod(weight.shape)
        print(f"   📊 Total Parameters: {total_params:,}")

if __name__ == "__main__":
    if setup_real_jax_environment():
        debug_parameter_sharding()
    else:
        print("❌ JAX environment not ready") 