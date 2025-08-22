"""
Definitive test to prove the exact source of numerical differences
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def test_definitive_identity():
    """Definitive test for numerical identity."""
    
    print("🚀 DEFINITIVE Numerical Identity Test")
    print("=" * 60)
    
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
    
    # Test 1: Get output BEFORE TensorParallelKeras
    print("\n🧪 Test 1: Output BEFORE TensorParallelKeras")
    output_before = model(x, training=False)
    print(f"   Output shape: {output_before.shape}")
    print(f"   Output values: {output_before.numpy()[:2, :2]}")
    
    # Test 2: Create TensorParallelKeras (but don't call it yet)
    print("\n🧪 Test 2: Create TensorParallelKeras (no calls)")
    try:
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        # Create wrapper but DON'T call it
        tp_model = TensorParallelKeras(model, world_size=2)
        print(f"   ✅ TensorParallelKeras created")
        print(f"   Original model id: {id(model)}")
        print(f"   TP model.original_model id: {id(tp_model.original_model)}")
        
        # Test 3: Get output AFTER TensorParallelKeras creation (but no calls)
        print("\n🧪 Test 3: Output AFTER TensorParallelKeras creation (no calls)")
        output_after_creation = model(x, training=False)
        print(f"   Output shape: {output_after_creation.shape}")
        print(f"   Output values: {output_after_creation.numpy()[:2, :2]}")
        
        # Compare BEFORE vs AFTER creation
        creation_diff = np.abs(output_before.numpy() - output_after_creation.numpy())
        creation_max_diff = np.max(creation_diff)
        print(f"   BEFORE vs AFTER creation max diff: {creation_max_diff:.2e}")
        
        if creation_max_diff < 1e-15:
            print("   ✅ PERFECT IDENTITY: Model unchanged after TensorParallelKeras creation")
        else:
            print("   ❌ DIFFERENCE: Model changed during TensorParallelKeras creation!")
            
        # Test 4: Get output from TensorParallelKeras
        print("\n🧪 Test 4: Output FROM TensorParallelKeras")
        tp_output = tp_model(x, training=False)
        print(f"   TP Output shape: {tp_output.shape}")
        print(f"   TP Output values: {tp_output.numpy()[:2, :2]}")
        
        # Compare BEFORE vs TensorParallelKeras
        tp_diff = np.abs(output_before.numpy() - tp_output.numpy())
        tp_max_diff = np.max(tp_diff)
        print(f"   BEFORE vs TensorParallelKeras max diff: {tp_max_diff:.2e}")
        
        if tp_max_diff < 1e-15:
            print("   ✅ PERFECT IDENTITY: TensorParallelKeras matches original")
        else:
            print("   ❌ DIFFERENCE: TensorParallelKeras differs from original")
            
        # Test 5: Check if weights changed
        print("\n🧪 Test 5: Weight Comparison")
        if creation_max_diff > 1e-15:
            print("   🔍 Investigating weight changes...")
            
            # Check if any weights changed during TensorParallelKeras creation
            for i, weight in enumerate(model.weights):
                weight_name = weight.name
                print(f"      Weight {i} ({weight_name}): {weight.shape}")
        
        # Test 6: Check if model state changed
        print("\n🧪 Test 6: Model State Check")
        print(f"   Model built: {model.built}")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        # Test 7: Check if optimizer state changed
        print("\n🧪 Test 7: Optimizer State Check")
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            print(f"   Optimizer: {type(model.optimizer).__name__}")
            print(f"   Learning rate: {model.optimizer.learning_rate}")
        else:
            print("   No optimizer found")
    
    except ImportError as e:
        print(f"   ⚠️ Could not import TensorParallelKeras: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Definitive Test Complete")

if __name__ == "__main__":
    test_definitive_identity() 