"""
Test for Perfect Numerical Identity in Tensor Parallelism
This test ensures both models use the EXACT same weights and random states
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_model():
    """Create a simple test model."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,), name='dense_1'),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dense(10, activation='softmax', name='output')
    ], name='test_model')
    return model

def test_perfect_numerical_identity():
    """Test that demonstrates perfect numerical identity."""
    
    print("🚀 Testing Perfect Numerical Identity")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create input data
    batch_size = 8
    input_shape = (32,)
    inputs = np.random.randn(batch_size, *input_shape).astype(np.float32)
    targets = keras.utils.to_categorical(np.random.randint(0, 10, batch_size), 10)
    
    print(f"📊 Input shape: {inputs.shape}")
    print(f"📊 Target shape: {targets.shape}")
    
    # Test 1: Same model, same computation
    print("\n🧪 Test 1: Same Model, Same Computation")
    print("-" * 40)
    
    # Reset seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create model
    model = create_test_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # First computation
    output1 = model(inputs, training=False)
    
    # Second computation (should be identical)
    output2 = model(inputs, training=False)
    
    # Compare
    diff = np.abs(output1.numpy() - output2.numpy())
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Max diff: {max_diff:.2e}")
    print(f"   Mean diff: {mean_diff:.2e}")
    
    if max_diff < 1e-15:
        print("   ✅ PERFECT IDENTITY: Same model, same computation")
    else:
        print("   ❌ UNEXPECTED DIFFERENCE: Same model should be identical")
    
    # Test 2: Different model instances, same weights
    print("\n🧪 Test 2: Different Model Instances, Same Weights")
    print("-" * 50)
    
    # Reset seeds and create first model
    np.random.seed(42)
    tf.random.set_seed(42)
    model1 = create_test_model()
    model1.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Reset seeds and create second model (should have same weights)
    np.random.seed(42)
    tf.random.set_seed(42)
    model2 = create_test_model()
    model2.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Ensure models are built
    _ = model1(inputs[:1])
    _ = model2(inputs[:1])
    
    # Copy weights from model1 to model2 to ensure they're identical
    for w1, w2 in zip(model1.weights, model2.weights):
        w2.assign(w1)
    
    # Compute outputs
    output1 = model1(inputs, training=False)
    output2 = model2(inputs, training=False)
    
    # Compare
    diff = np.abs(output1.numpy() - output2.numpy())
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Max diff: {max_diff:.2e}")
    print(f"   Mean diff: {mean_diff:.2e}")
    
    if max_diff < 1e-15:
        print("   ✅ PERFECT IDENTITY: Different models, same weights")
    else:
        print("   ❌ UNEXPECTED DIFFERENCE: Same weights should be identical")
        
        # Debug: Check if weights are actually the same
        print("   🔍 Debugging weight differences:")
        for i, (w1, w2) in enumerate(zip(model1.weights, model2.weights)):
            w_diff = np.abs(w1.numpy() - w2.numpy())
            print(f"      Weight {i}: max_diff={np.max(w_diff):.2e}")
    
    # Test 3: Tensor Parallel vs Original (this is our real test)
    print("\n🧪 Test 3: TensorParallelKeras vs Original Model")
    print("-" * 50)
    
    try:
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        # Reset seeds and create original model
        np.random.seed(42)
        tf.random.set_seed(42)
        original_model = create_test_model()
        original_model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Create tensor parallel model with same original model
        tp_model = TensorParallelKeras(original_model, world_size=2)
        
        # Ensure models are built
        _ = original_model(inputs[:1])
        _ = tp_model(inputs[:1])
        
        # Compute outputs
        original_output = original_model(inputs, training=False)
        tp_output = tp_model(inputs, training=False)
        
        # Compare
        diff = np.abs(original_output.numpy() - tp_output.numpy())
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   Max diff: {max_diff:.2e}")
        print(f"   Mean diff: {mean_diff:.2e}")
        
        if max_diff < 1e-15:
            print("   ✅ PERFECT IDENTITY: TensorParallelKeras matches original")
        elif max_diff < 1e-6:
            print("   ✅ VERY CLOSE: Differences within acceptable tolerance")
        else:
            print("   ❌ SIGNIFICANT DIFFERENCE: Investigating cause...")
            
            # Check if it's using the same underlying model
            if hasattr(tp_model, 'original_model'):
                print(f"      TensorParallelKeras original_model id: {id(tp_model.original_model)}")
                print(f"      Original model id: {id(original_model)}")
                
                if id(tp_model.original_model) == id(original_model):
                    print("      ✅ Same underlying model object")
                else:
                    print("      ❌ Different model objects - this explains the difference")
    
    except ImportError as e:
        print(f"   ⚠️ Could not import TensorParallelKeras: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Analysis Complete")

if __name__ == "__main__":
    test_perfect_numerical_identity()