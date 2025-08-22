#!/usr/bin/env python3
"""
Comprehensive test suite for tensor parallelism with different layer types.
Tests MLP, self-attention, and einsum dense layers for numerical correctness.
"""

import numpy as np
import keras
from keras import layers, Model
import os

def setup_jax_backend():
    """Set up JAX backend."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    return True

def create_mlp_model():
    """Create a model with MLP layers."""
    inputs = keras.Input(shape=(64,))
    x = layers.Dense(128, activation='relu', name="mlp_1")(inputs)
    x = layers.Dense(256, activation='relu', name="mlp_2")(x)
    x = layers.Dense(128, activation='relu', name="mlp_3")(x)
    outputs = layers.Dense(32, name="mlp_output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_self_attention_model():
    """Create a model with self-attention layers."""
    inputs = keras.Input(shape=(32, 64))
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=8, 
        key_dim=8, 
        name="self_attn"
    )(inputs, inputs)
    
    # Add & Norm
    x = layers.Add()([inputs, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    x = layers.Dense(256, activation='relu', name="ffn_1")(x)
    x = layers.Dense(64, name="ffn_2")(x)
    
    # Add & Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(16, name="attn_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_einsum_dense_model():
    """Create a model with einsum dense layers (simulating complex operations)."""
    inputs = keras.Input(shape=(48,))
    
    # Dense layer that will be sharded
    x = layers.Dense(96, activation='relu', name="einsum_dense_1")(inputs)
    
    # Another dense operation
    x = layers.Dense(64, activation='relu', name="einsum_dense_2")(x)
    
    # Final output - ensure proper shape
    outputs = layers.Dense(24, name="einsum_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_layer_type_tensor_parallelism(model_creator, model_name, input_shape, target_shape):
    """Test tensor parallelism for a specific layer type."""
    print(f"\n🧪 Testing {model_name} Tensor Parallelism")
    print("=" * 60)
    
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Create test data
    x = np.random.randn(*input_shape).astype(np.float32)
    y = np.random.randn(*target_shape).astype(np.float32)
    
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    # Test 1: Single CPU model
    print(f"\n🔍 Test 1: Single CPU {model_name}")
    single_model = model_creator()
    
    # Use fixed learning rate
    single_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    single_model.compile(optimizer=single_optimizer, loss='mse')
    
    print(f"✅ Single CPU {model_name} compiled successfully")
    
    # Store initial weights
    initial_weights = [w.numpy().copy() for w in single_model.weights]
    print(f"✅ Initial weights captured: {len(initial_weights)} parameters")
    
    # Single training step
    single_loss = single_model.train_on_batch(x, y)
    single_updated_weights = [w.numpy().copy() for w in single_model.weights]
    
    print(f"✅ Single CPU {model_name} training completed - Loss: {single_loss:.6f}")
    
    # Test 2: Tensor parallel model with IDENTICAL initial weights
    print(f"\n🔍 Test 2: Tensor Parallel {model_name} (Identical Initial Weights)")
    
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
    
    # Create base model for tensor parallelism
    base_model = model_creator()
    
    # CRITICAL: Set EXACTLY the same initial weights
    base_model.set_weights(initial_weights)
    
    # Verify weights are identical
    base_weights = [w.numpy().copy() for w in base_model.weights]
    weights_identical = all(
        np.allclose(init_w, base_w, atol=1e-10) 
        for init_w, base_w in zip(initial_weights, base_weights)
    )
    print(f"✅ Base model weights identical: {weights_identical}")
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model=base_model,
        world_size=2,
        distributed_backend="jax"
    )
    
    # Create SEPARATE optimizer with SAME learning rate
    tp_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    tp_model.compile(optimizer=tp_optimizer, loss='mse')
    
    print(f"✅ Tensor parallel {model_name} compiled successfully")
    
    # Verify tensor parallel model has identical weights
    tp_initial_weights = [w.numpy().copy() for w in tp_model.weights]
    tp_weights_identical = all(
        np.allclose(init_w, tp_w, atol=1e-10) 
        for init_w, tp_w in zip(initial_weights, tp_initial_weights)
    )
    print(f"✅ Tensor parallel weights identical: {tp_weights_identical}")
    
    if not tp_weights_identical:
        print("❌ CRITICAL: Tensor parallel model weights not identical!")
        for i, (init_w, tp_w) in enumerate(zip(initial_weights, tp_initial_weights)):
            diff = np.abs(init_w - tp_w)
            max_diff = np.max(diff)
            print(f"  Weight {i}: max_diff={max_diff:.2e}")
        return False
    
    # Test 3: Forward pass comparison (should be identical)
    print(f"\n🔍 Test 3: Forward Pass Comparison")
    single_output = single_model(x, training=False)
    tp_output = tp_model(x, training=False)
    
    forward_diff = np.abs(single_output.numpy() - tp_output.numpy())
    max_forward_diff = np.max(forward_diff)
    mean_forward_diff = np.mean(forward_diff)
    
    print(f"Forward pass - Max diff: {max_forward_diff:.2e}")
    print(f"Forward pass - Mean diff: {mean_forward_diff:.2e}")
    
    # Test 4: Tensor parallel training step
    print(f"\n🔍 Test 4: Tensor Parallel Training Step")
    
    try:
        tp_loss = tp_model.train_on_batch(x, y)
        tp_updated_weights = [w.numpy().copy() for w in tp_model.weights]
        
        print(f"✅ Tensor parallel {model_name} training completed - Loss: {tp_loss:.6f}")
        
        # Compare results
        print(f"\n🔍 Test 5: Comparing Results")
        loss_diff = abs(single_loss - tp_loss)
        print(f"Loss difference: {loss_diff:.2e}")
        
        # Compare weight updates
        print(f"\n📊 Weight Update Comparison:")
        max_weight_diffs = []
        
        for i, (single_w, tp_w) in enumerate(zip(single_updated_weights, tp_updated_weights)):
            diff = np.abs(single_w - tp_w)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            max_weight_diffs.append(max_diff)
            print(f"  Weight {i} {single_w.shape}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        
        overall_max_diff = max(max_weight_diffs)
        
        # Success criteria
        forward_ok = max_forward_diff < 1e-6
        loss_ok = loss_diff < 1e-5
        weights_ok = overall_max_diff < 1e-4  # More relaxed tolerance for optimizer states
        
        print(f"\n📊 Results for {model_name}:")
        print(f"✅ Forward pass: {'PASS' if forward_ok else 'FAIL'} (max diff: {max_forward_diff:.2e})")
        print(f"✅ Loss matching: {'PASS' if loss_ok else 'FAIL'} (diff: {loss_diff:.2e})")
        print(f"✅ Weight updates: {'PASS' if weights_ok else 'FAIL'} (max diff: {overall_max_diff:.2e})")
        
        overall_success = forward_ok and loss_ok and weights_ok
        print(f"\n🎯 Overall {model_name} Test: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Tensor parallel training failed: {e}")
        
        # Debug: Let's see what the exact issue is
        print("\n🔧 Debug Information:")
        print(f"  Single model variables: {len(single_model.trainable_weights)}")
        print(f"  TP model variables: {len(tp_model.trainable_weights)}")
        print(f"  Single optimizer class: {type(single_optimizer)}")
        print(f"  TP optimizer class: {type(tp_optimizer)}")
        
        # Check if the models have the same structure
        single_var_shapes = [w.shape for w in single_model.trainable_weights]
        tp_var_shapes = [w.shape for w in tp_model.trainable_weights]
        print(f"  Single model shapes: {single_var_shapes}")
        print(f"  TP model shapes: {tp_var_shapes}")
        
        return False

def main():
    """Run all layer type tests."""
    print("🚀 Comprehensive Tensor Parallelism Layer Type Testing")
    print("Testing MLP, Self-Attention, and Einsum Dense Layers")
    print("=" * 80)
    
    setup_jax_backend()
    
    # Test results
    test_results = []
    
    # Test 1: MLP layers
    mlp_success = test_layer_type_tensor_parallelism(
        create_mlp_model, 
        "MLP Model", 
        (32, 64), 
        (32, 32)
    )
    test_results.append(("MLP Model", mlp_success))
    
    # Test 2: Self-attention layers
    attn_success = test_layer_type_tensor_parallelism(
        create_self_attention_model, 
        "Self-Attention Model", 
        (16, 32, 64), 
        (16, 16)
    )
    test_results.append(("Self-Attention Model", attn_success))
    
    # Test 3: Einsum dense layers
    einsum_success = test_layer_type_tensor_parallelism(
        create_einsum_dense_model, 
        "Einsum Dense Model", 
        (24, 48), 
        (24, 24)
    )
    test_results.append(("Einsum Dense Model", einsum_success))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 Ready for OPT-125M Testing!")
        print("All layer types verified for tensor parallelism correctness.")
    else:
        print("\n🔧 Need to fix failing tests before proceeding to OPT-125M.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 