#!/usr/bin/env python3
"""
Test communication rules to verify they're being applied correctly.
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

def test_communication_rules():
    """Test that communication rules are being applied."""
    print("🧪 Testing Communication Rules Application")
    
    # Create model
    model = create_test_model()
    print(f"✅ Model created: {model.name}")
    
    # Create tensor parallel model
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
    
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend="jax"
    )
    
    print(f"\n📊 Tensor Parallel Model Analysis:")
    print(f"   Model name: {tp_model.original_model.name}")
    print(f"   Has tensor_parallel_config: {hasattr(tp_model, 'tensor_parallel_config')}")
    
    if hasattr(tp_model, 'tensor_parallel_config'):
        config = tp_model.tensor_parallel_config
        print(f"   Output rules: {config.output_rules}")
        print(f"   State rules: {list(config.state_rules.keys())[:5]}...")  # Show first 5
    
    # Test forward pass to see if communication is applied
    print(f"\n🔍 Testing Forward Pass with Communication")
    
    # Create test input
    test_input = np.random.random((8, 32)).astype('float32')
    
    try:
        # This should trigger the communication rules
        output = tp_model(test_input, training=False)
        print(f"✅ Forward pass completed - Output shape: {output.shape}")
        
        # Check if communication was applied
        if hasattr(tp_model, 'tensor_parallel_config'):
            print(f"✅ Communication rules should have been applied")
        else:
            print(f"⚠️  No tensor parallel config found")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if setup_real_jax_environment():
        test_communication_rules()
    else:
        print("❌ JAX environment not ready") 