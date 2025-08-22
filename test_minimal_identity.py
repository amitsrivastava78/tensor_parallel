"""
Minimal test to isolate numerical identity issue
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def test_minimal_identity():
    """Minimal test for numerical identity."""
    
    print("🚀 MINIMAL Numerical Identity Test")
    print("=" * 50)
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create simple model
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(5,)),
        layers.Dense(3, activation='softmax')
    ])
    
    # Create input
    x = np.random.randn(4, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=(4,))
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Test 1: Get output BEFORE compilation
    print("\n🧪 Test 1: Output BEFORE compilation")
    output_before_compile = model(x, training=False)
    print(f"   Output shape: {output_before_compile.shape}")
    print(f"   Output values: {output_before_compile.numpy()[:2, :2]}")
    
    # Test 2: Compile model
    print("\n🧪 Test 2: Compile model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print(f"   ✅ Model compiled")
    
    # Test 3: Get output AFTER compilation
    print("\n🧪 Test 3: Output AFTER compilation")
    output_after_compile = model(x, training=False)
    print(f"   Output shape: {output_after_compile.shape}")
    print(f"   Output values: {output_after_compile.numpy()[:2, :2]}")
    
    # Compare BEFORE vs AFTER compilation
    compile_diff = np.abs(output_before_compile.numpy() - output_after_compile.numpy())
    compile_max_diff = np.max(compile_diff)
    print(f"   BEFORE vs AFTER compilation max diff: {compile_max_diff:.2e}")
    
    if compile_max_diff < 1e-15:
        print("   ✅ PERFECT IDENTITY: Model unchanged after compilation")
    else:
        print("   ❌ DIFFERENCE: Model changed during compilation!")
    
    # Test 4: Create TensorParallelKeras
    print("\n🧪 Test 4: Create TensorParallelKeras")
    try:
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        # Create wrapper
        tp_model = TensorParallelKeras(model, world_size=2)
        print(f"   ✅ TensorParallelKeras created")
        
        # Test 5: Get output from TensorParallelKeras
        print("\n🧪 Test 5: Output FROM TensorParallelKeras")
        tp_output = tp_model(x, training=False)
        print(f"   TP Output shape: {tp_output.shape}")
        print(f"   TP Output values: {tp_output.numpy()[:2, :2]}")
        
        # Compare AFTER compilation vs TensorParallelKeras
        tp_diff = np.abs(output_after_compile.numpy() - tp_output.numpy())
        tp_max_diff = np.max(tp_diff)
        print(f"   AFTER compilation vs TensorParallelKeras max diff: {tp_max_diff:.2e}")
        
        if tp_max_diff < 1e-15:
            print("   ✅ PERFECT IDENTITY: TensorParallelKeras matches compiled model")
        else:
            print("   ❌ DIFFERENCE: TensorParallelKeras differs from compiled model")
            
        # Test 6: Check if model weights changed
        print("\n🧪 Test 6: Weight Comparison")
        if tp_max_diff > 1e-15:
            print("   🔍 Investigating weight changes...")
            
            # Check if any weights changed during TensorParallelKeras creation
            for i, weight in enumerate(model.weights):
                weight_name = weight.name
                print(f"      Weight {i} ({weight_name}): {weight.shape}")
        
        # Test 7: Check if model state changed
        print("\n🧪 Test 7: Model State Check")
        print(f"   Model built: {model.built}")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        # Test 8: Check if optimizer state changed
        print("\n🧪 Test 8: Optimizer State Check")
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            print(f"   Optimizer: {type(model.optimizer).__name__}")
            print(f"   Learning rate: {model.optimizer.learning_rate}")
        else:
            print("   No optimizer found")
    
    except ImportError as e:
        print(f"   ⚠️ Could not import TensorParallelKeras: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Minimal Test Complete")

if __name__ == "__main__":
    test_minimal_identity() 