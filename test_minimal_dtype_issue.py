#!/usr/bin/env python3
"""
Minimal test to isolate the dtype issue.
"""

import os

# Set Keras to use JAX backend explicitly (no TensorFlow)
os.environ['KERAS_BACKEND'] = 'jax'

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import numpy as np
import keras
from keras import layers, optimizers

print(f"🔍 Keras backend: {keras.config.backend()}")
print(f"🔍 Using Keras 3.0 with JAX backend (no TensorFlow)")

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_minimal_model():
    """Create a minimal model to test dtype handling."""
    inputs = layers.Input(shape=(10,), dtype='int32', name='input_ids')
    embedding = layers.Embedding(1000, 64, name='embed_tokens')(inputs)
    outputs = layers.Dense(1000, name='lm_head')(embedding)
    model = keras.Model(inputs=inputs, outputs=outputs, name='MinimalModel')
    return model

def test_minimal_training():
    """Test minimal training to isolate dtype issue."""
    print("🔧 Testing minimal model training...")
    
    # Create minimal model
    model = create_minimal_model()
    print(f"   Model created with {model.count_params():,} parameters")
    
    # Check model weights
    print("   Model weights:")
    for weight in model.weights:
        print(f"     {weight.name}: shape={weight.shape}, dtype={weight.dtype}")
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy'
        # Removed metrics to avoid JAX issues
    )
    print("   ✅ Model compiled successfully")
    
    # Create training data
    x_train = np.random.randint(0, 1000, (2, 10), dtype=np.int32)
    y_train = np.random.randint(0, 1000, (2, 10), dtype=np.int32)
    
    # Test single training step
    try:
        result = model.train_on_batch(x_train, y_train)
        # Handle different result types
        try:
            # Try as array first
            loss = result[0]
        except (IndexError, TypeError):
            # If that fails, try as scalar
            loss = float(result)
        print(f"   ✅ Single training step successful: loss={loss:.6f}")
    except Exception as e:
        print(f"   ❌ Single training step failed: {e}")
        return False
    
    # Now test with TensorParallelKeras
    print("\n🔧 Testing with TensorParallelKeras...")
    
    try:
        tp_model = TensorParallelKeras(
            model=model,
            world_size=2,
            distributed_backend='jax'
        )
        print("   ✅ TensorParallelKeras created successfully")
        
        # Check TP model weights
        print("   TP Model weights:")
        for weight in tp_model.weights:
            print(f"     {weight.name}: shape={weight.shape}, dtype={weight.dtype}")
        
        # Compile TP model
        tp_model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy'
            # Removed metrics to avoid JAX issues
        )
        print("   ✅ TP model compiled successfully")
        
        # Test TP training step
        tp_result = tp_model.train_on_batch(x_train, y_train)
        # Handle different result types
        try:
            # Try as array first
            tp_loss = tp_result[0]
        except (IndexError, TypeError):
            # If that fails, try as scalar
            tp_loss = float(tp_result)
        print(f"   ✅ TP training step successful: loss={tp_loss:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TensorParallelKeras test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 Minimal Dtype Issue Test")
    print("=" * 40)
    
    success = test_minimal_training()
    
    if success:
        print("\n✅ Minimal test passed!")
    else:
        print("\n❌ Minimal test failed!") 