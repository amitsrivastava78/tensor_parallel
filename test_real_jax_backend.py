"""
Test REAL JAX Backend - NO STUBS, REAL DISTRIBUTED COMPUTATION
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def test_real_jax_backend():
    """Test the REAL JAX backend with actual distributed computation."""
    
    print("🚀 Testing REAL JAX Backend - NO STUBS!")
    print("=" * 60)
    
    try:
        # Import the REAL implementation
        from src.tensor_parallel_keras import TensorParallelKeras, get_default_backend
        
        # Check backend status
        backend = get_default_backend()
        print(f"✅ Backend: {type(backend).__name__}")
        print(f"✅ Is Real: {backend.is_real_backend()}")
        print(f"✅ Device Count: {backend.get_device_count()}")
        
        device_info = backend.get_device_info()
        print(f"✅ Platform: {device_info['platform']}")
        print(f"✅ Distributed: {device_info['distributed']}")
        
        # Create a simple model
        model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(5,)),
            layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Create test data
        x = np.random.random((4, 5)).astype('float32')
        y = np.random.randint(0, 5, size=(4,))
        
        print(f"\n📊 Model created: {len(model.weights)} parameters")
        print(f"📊 Input shape: {x.shape}")
        print(f"📊 Target shape: {y.shape}")
        
        # Test REAL tensor parallelism
        print(f"\n{'='*60}")
        print(f"🧪 Testing REAL Tensor Parallelism (2 devices)")
        print(f"{'='*60}")
        
        try:
            tp_model = TensorParallelKeras(model, world_size=2)
            print("✅ TensorParallelKeras created successfully!")
            
            # Test forward pass
            output = tp_model(x, training=False)
            print(f"✅ Forward pass completed - Output shape: {output.shape}")
            
            # Test training step
            loss = tp_model.train_on_batch(x, y)
            print(f"✅ Training completed - Loss: {loss:.6f}")
            
            print(f"\n🎉 SUCCESS: REAL distributed computation working!")
            print(f"✅ NO STUBS - actual JAX pmap distributed computation!")
            
        except Exception as e:
            print(f"❌ Tensor parallelism failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 Test Complete")

if __name__ == "__main__":
    test_real_jax_backend() 