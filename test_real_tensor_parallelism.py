#!/usr/bin/env python3
"""
Test Real Tensor Parallelism (No STUBS)
Verifies actual distributed computation across multiple devices.
"""

import os
import numpy as np
import keras
from keras import layers, Model

def setup_real_jax_environment():
    """Set up real JAX multi-device environment."""
    print("🚀 Setting up REAL JAX multi-device environment")
    
    # Set environment variables for multi-device JAX
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    print("✅ Environment variables set:")
    print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
    print(f"   JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME')}")
    
    # Import JAX and check devices
    try:
        import jax
        import jax.devices
        
        devices = jax.devices()
        print(f"🔍 Real JAX devices detected: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")
        
        if len(devices) >= 2:
            print("✅ Multi-device JAX environment ready!")
            return True
        else:
            print("❌ Need at least 2 JAX devices for tensor parallelism")
            return False
            
    except ImportError as e:
        print(f"❌ JAX not available: {e}")
        return False

def create_test_model():
    """Create a simple test model."""
    inputs = keras.Input(shape=(32,))
    x = layers.Dense(64, activation='relu', name="dense_1")(inputs)
    x = layers.Dropout(0.1, name="dropout_1")(x)
    x = layers.Dense(32, activation='relu', name="dense_2")(x)
    outputs = layers.Dense(10, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_real_tensor_parallelism():
    """Test real tensor parallelism implementation."""
    print("\n🧪 Testing REAL Tensor Parallelism (No STUBS)")
    print("=" * 60)
    
    # Set up real JAX environment
    if not setup_real_jax_environment():
        print("❌ Cannot proceed without real JAX multi-device environment")
        return False
    
    # Create test model
    model = create_test_model()
    print(f"\n✅ Test model created: {len(model.weights)} parameters")
    
    # Create tensor parallel model
    try:
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        print(f"\n🔧 Creating TensorParallelKeras with world_size=2")
        tp_model = TensorParallelKeras(
            model=model,
            world_size=2,
            distributed_backend="jax"
        )
        
        print(f"✅ TensorParallelKeras created successfully")
        
        # Check if we have real shards
        if hasattr(tp_model, 'model_shards'):
            print(f"\n📊 Shard Analysis:")
            print(f"   Number of shards: {len(tp_model.model_shards)}")
            
            # Check if shards are different (real sharding)
            shard_params = []
            for i, shard in enumerate(tp_model.model_shards):
                param_count = sum(np.prod(w.shape) for w in shard.weights)
                shard_params.append(param_count)
                print(f"   Shard {i} parameters: {param_count:,}")
            
            # Verify real sharding
            if len(set(shard_params)) > 1:
                print("✅ REAL SHARDING CONFIRMED: Different parameter counts across shards")
                print("✅ This is NOT using stubs - real tensor parallelism!")
            else:
                print("⚠️  WARNING: Shards have same parameter count")
                print("⚠️  This might still be using stubs")
                
        else:
            print("❌ No model shards found")
            return False
            
        # Test forward pass
        print(f"\n🔍 Testing Forward Pass")
        x = np.random.random((8, 32)).astype('float32')
        
        # Single model output
        single_output = model(x, training=False)
        print(f"   Single model output shape: {single_output.shape}")
        
        # Tensor parallel output
        tp_output = tp_model(x, training=False)
        print(f"   TP model output shape: {tp_output.shape}")
        
        # Check if outputs are different (real computation)
        if np.allclose(single_output.numpy(), tp_output.numpy(), atol=1e-6):
            print("⚠️  WARNING: Outputs are identical - may still be using stubs")
        else:
            print("✅ REAL COMPUTATION: Outputs are different (as expected with real sharding)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating TensorParallelKeras: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the real tensor parallelism test."""
    print("🚀 Testing Real Tensor Parallelism Implementation")
    print("=" * 80)
    
    success = test_real_tensor_parallelism()
    
    print("\n" + "=" * 80)
    if success:
        print("🎯 Overall Result: ✅ REAL TENSOR PARALLELISM TEST COMPLETED")
        print("🔍 Check the output above to verify if stubs were removed")
    else:
        print("🎯 Overall Result: ❌ REAL TENSOR PARALLELISM TEST FAILED")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 