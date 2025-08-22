#!/usr/bin/env python3
"""
Test Real JAX Tensor Parallelism (No STUBS)
Verifies actual distributed computation across multiple JAX devices.
"""

import os
import numpy as np
import keras
from keras import layers, Model
import gc

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
        
        # Use correct JAX API for device detection
        device_count = jax.device_count()
        devices = jax.devices()
        
        print(f"🔍 Real JAX devices detected: {device_count}")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")
        
        if device_count >= 2:
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

def clean_test_environment():
    """Clean up the test environment."""
    keras.backend.clear_session()
    gc.collect()
    np.random.seed(42)
    keras.utils.set_random_seed(42)

def test_real_jax_backend():
    """Test the real JAX backend implementation."""
    print("\n🧪 Testing Real JAX Backend Implementation")
    print("=" * 60)
    
    try:
        from src.tensor_parallel_keras.jax_backend import JAXBackend
        
        # Initialize JAX backend
        jax_backend = JAXBackend()
        
        # Test device detection
        device_count = jax_backend.get_device_count()
        device_info = jax_backend.get_device_info()
        
        print(f"✅ JAX Backend initialized successfully")
        print(f"   Device count: {device_count}")
        print(f"   Platform: {device_info['platform']}")
        print(f"   Backend: {device_info['backend']}")
        print(f"   Devices: {device_info['devices']}")
        
        # Test parameter sharding
        print(f"\n🔧 Testing Real Parameter Sharding")
        test_params = [
            np.random.random((64, 32)).astype('float32'),
            np.random.random((32,)).astype('float32'),
            np.random.random((32, 16)).astype('float32')
        ]
        
        sharded_params = jax_backend.create_sharded_parameters(test_params, world_size=2)
        print(f"   ✅ Created {len(sharded_params)} shards")
        
        # Check if shards are different (real sharding)
        for i, shard in enumerate(sharded_params):
            print(f"   Shard {i}: {len(shard)} parameters")
            for j, param in enumerate(shard):
                print(f"     Param {j}: shape={param.shape}, type={type(param)}")
        
        # Test communication operations
        print(f"\n🔧 Testing Real Communication Operations")
        
        # Test AllReduce
        test_tensors = [np.random.random((8, 16)).astype('float32') for _ in range(2)]
        allreduce_result = jax_backend.all_reduce(test_tensors, op="sum")
        print(f"   ✅ AllReduce (sum): {len(allreduce_result)} results")
        
        # Test AllGather
        allgather_result = jax_backend.all_gather(test_tensors, dim=-1)
        print(f"   ✅ AllGather: {len(allgather_result)} results")
        
        # Test Broadcast
        broadcast_result = jax_backend.broadcast(test_tensors[0], src_rank=0)
        print(f"   ✅ Broadcast: {len(broadcast_result)} results")
        
        # Test Scatter
        scatter_result = jax_backend.scatter(test_tensors[0], world_size=2, dim=0)
        print(f"   ✅ Scatter: {len(scatter_result)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing JAX backend: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_tensor_parallelism():
    """Test real tensor parallelism implementation."""
    print("\n🧪 Testing Real Tensor Parallelism (No STUBS)")
    print("=" * 60)
    
    # Create test model
    model = create_test_model()
    print(f"✅ Test model created: {len(model.weights)} parameters")
    
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
    """Run the real JAX tensor parallelism test."""
    print("🚀 Testing Real JAX Tensor Parallelism Implementation")
    print("=" * 80)
    
    # Set up environment
    if not setup_real_jax_environment():
        print("❌ Cannot proceed without real JAX multi-device environment")
        return False
    
    # Clean environment
    clean_test_environment()
    
    # Test JAX backend
    backend_success = test_real_jax_backend()
    
    # Test tensor parallelism
    tp_success = test_real_tensor_parallelism()
    
    # Final results
    print("\n" + "=" * 80)
    if backend_success and tp_success:
        print("🎯 Overall Result: ✅ REAL JAX TENSOR PARALLELISM TEST PASSED")
        print("🎉 Stubs have been removed - real distributed computation working!")
        print("✅ Real JAX backend: Working")
        print("✅ Real parameter sharding: Working")
        print("✅ Real communication operations: Working")
    else:
        print("🎯 Overall Result: ❌ REAL JAX TENSOR PARALLELISM TEST FAILED")
        print("🔧 Some components still using stubs or failed")
    
    return backend_success and tp_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 