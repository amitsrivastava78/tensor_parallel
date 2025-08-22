"""
Simple test to isolate numerical identity issue
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def test_simple_identity():
    """Test simple numerical identity."""
    
    print("🚀 Simple Numerical Identity Test")
    print("=" * 50)
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create simple model
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(5,)),
        layers.Dense(3, activation='softmax')
    ])
    
    # Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Create input
    x = np.random.randn(4, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=(4,))
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Test 1: Direct model call
    print("\n🧪 Test 1: Direct Model Call")
    output1 = model(x, training=False)
    print(f"   Output 1 shape: {output1.shape}")
    print(f"   Output 1 values: {output1.numpy()[:2, :2]}")
    
    # Test 2: Same model call again
    output2 = model(x, training=False)
    print(f"   Output 2 shape: {output2.shape}")
    print(f"   Output 2 values: {output2.numpy()[:2, :2]}")
    
    # Compare
    diff = np.abs(output1.numpy() - output2.numpy())
    max_diff = np.max(diff)
    print(f"   Max diff: {max_diff:.2e}")
    
    if max_diff < 1e-15:
        print("   ✅ PERFECT IDENTITY")
    else:
        print("   ❌ DIFFERENCE DETECTED")
    
    # Test 3: Training step
    print("\n🧪 Test 3: Training Step")
    loss1 = model.train_on_batch(x, y)
    print(f"   Loss 1: {loss1:.6f}")
    
    # Test 4: Same training step again
    loss2 = model.train_on_batch(x, y)
    print(f"   Loss 2: {loss2:.6f}")
    
    loss_diff = abs(loss1 - loss2)
    print(f"   Loss diff: {loss_diff:.2e}")
    
    if loss_diff < 1e-15:
        print("   ✅ PERFECT IDENTITY")
    else:
        print("   ❌ DIFFERENCE DETECTED")
    
    # Test 5: TensorParallelKeras wrapper
    print("\n🧪 Test 5: TensorParallelKeras Wrapper")
    try:
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        # Create wrapper
        tp_model = TensorParallelKeras(model, world_size=2)
        tp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Forward pass
        tp_output = tp_model(x, training=False)
        print(f"   TP Output shape: {tp_output.shape}")
        print(f"   TP Output values: {tp_output.numpy()[:2, :2]}")
        
        # Compare with original
        tp_diff = np.abs(output1.numpy() - tp_output.numpy())
        tp_max_diff = np.max(tp_diff)
        print(f"   TP vs Original max diff: {tp_max_diff:.2e}")
        
        if tp_max_diff < 1e-15:
            print("   ✅ PERFECT IDENTITY: TensorParallelKeras matches original")
        else:
            print("   ❌ DIFFERENCE: TensorParallelKeras differs from original")
            
            # Debug: Check if it's using the same underlying model
            if hasattr(tp_model, 'original_model'):
                print(f"      TP model.original_model id: {id(tp_model.original_model)}")
                print(f"      Original model id: {id(model)}")
                
                if id(tp_model.original_model) == id(model):
                    print("      ✅ Same underlying model object")
                else:
                    print("      ❌ Different model objects")
    
    except ImportError as e:
        print(f"   ⚠️ Could not import TensorParallelKeras: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Complete")

if __name__ == "__main__":
    test_simple_identity() 