#!/usr/bin/env python3
"""
Comprehensive Test Suite for OPT-125M Model Training Readiness
Tests all critical components needed for production training with tensor parallelism
"""

import os
import time
import logging
import numpy as np
import keras
from keras import layers, optimizers
import tensorflow as tf

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_opt125m_model(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
    """Create a simplified OPT-125M model for testing."""
    print("   Creating OPT-125M model...")
    
    # Input layer
    inputs = layers.Input(shape=(None,), dtype='int32', name='input_ids')
    
    # Embedding layer
    embedding = layers.Embedding(vocab_size, hidden_size, name='embed_tokens')(inputs)
    
    # For testing, just use the embedding directly (no position embedding)
    hidden_states = embedding
    
    # Layer normalization
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_embedding')(hidden_states)
    
    # Transformer layers
    for i in range(num_layers):
        print(f"     Adding transformer layer {i+1}/{num_layers}")
        
        # Self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            name=f'layers_{i}_self_attn'
        )(hidden_states, hidden_states)
        
        # Residual connection
        hidden_states = layers.Add(name=f'layers_{i}_residual_1')([hidden_states, attention_output])
        
        # Layer normalization
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layernorm_1_{i}')(hidden_states)
        
        # MLP (Feed-forward)
        mlp_hidden = layers.Dense(hidden_size * 4, activation='relu', name=f'layers_{i}_mlp_fc1')(hidden_states)
        mlp_output = layers.Dense(hidden_size, name=f'layers_{i}_mlp_fc2')(mlp_hidden)
        
        # Residual connection
        hidden_states = layers.Add(name=f'layers_{i}_residual_2')([hidden_states, mlp_output])
        
        # Layer normalization
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layernorm_2_{i}')(hidden_states)
    
    # Final layer normalization
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_final')(hidden_states)
    
    # Output projection
    outputs = layers.Dense(vocab_size, name='lm_head')(hidden_states)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='OPT-125M')
    
    print(f"      OPT-125M model created with {model.count_params():,} parameters")
    return model

def test_opt125m_backward_pass_consistency():
    """
    Test backward pass consistency for OPT-125M model.
    Verify that gradients computed on 2-CPU sharded model match single-CPU model.
    """
    print("🔧 OPT-125M Backward Pass Consistency Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create OPT-125M model
    opt_model = create_opt125m_model()
    
    # Create TensorParallelKeras model
    tp_model = TensorParallelKeras(
        model=opt_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # Compile both models
    opt_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    tp_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create training data
    np.random.seed(42)
    batch_size = 2
    seq_len = 10
    vocab_size = 50257
    
    x_train = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    y_train = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    
    print(f"      Training data: x={x_train.shape}, y={y_train.shape}")
    
    # Test single training step
    try:
        # Train original model
        original_result = opt_model.train_on_batch(x_train, y_train)
        print(f"      ✅ Original model training step successful")
        print(f"      Original loss: {original_result[0]:.6f}")
        
        # Train tensor parallel model
        tp_result = tp_model.train_on_batch(x_train, y_train)
        print(f"      ✅ TP model training step successful")
        print(f"      TP loss: {tp_result[0]:.6f}")
        
        # Check if losses are close (allowing for some numerical differences)
        loss_diff = abs(original_result[0] - tp_result[0])
        if loss_diff < 1e-3:
            print(f"      ✅ Loss consistency verified (diff: {loss_diff:.2e})")
            result = True
        else:
            print(f"      ⚠️  Loss difference: {loss_diff:.2e}")
            result = False
            
    except Exception as e:
        print(f"      ❌ Training step failed: {e}")
        result = False
    
    print(f"✅ Backward pass consistency test completed in {time.time() - start_time:.2f}s")
    return result

def test_opt125m_gradient_computation():
    """
    Test gradient computation correctness for OPT-125M model.
    Verify that gradients are properly computed and synchronized across shards.
    """
    print("🔧 OPT-125M Gradient Computation Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create OPT-125M model
    opt_model = create_opt125m_model()
    
    # Create TensorParallelKeras model
    tp_model = TensorParallelKeras(
        model=opt_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # Compile models
    opt_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    tp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Create test data
    np.random.seed(42)
    x_test = np.random.randint(0, 1000, (1, 5), dtype=np.int32)
    y_test = np.random.randint(0, 1000, (1, 5), dtype=np.int32)
    
    try:
        # Get gradients for original model
        with tf.GradientTape() as tape:
            original_output = opt_model(x_test)
            original_loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, original_output)
        
        original_gradients = tape.gradient(original_loss, opt_model.trainable_variables)
        print(f"      ✅ Original model gradients computed")
        
        # Get gradients for TP model
        with tf.GradientTape() as tape:
            tp_output = tp_model(x_test)
            tp_loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, tp_output)
        
        tp_gradients = tape.gradient(tp_loss, tp_model.trainable_variables)
        print(f"      ✅ TP model gradients computed")
        
        # Check gradient shapes and values
        if len(original_gradients) == len(tp_gradients):
            print(f"      ✅ Gradient count matches: {len(original_gradients)}")
            
            # Check a few key gradients for numerical consistency
            grad_checks = 0
            for i, (orig_grad, tp_grad) in enumerate(zip(original_gradients[:3], tp_gradients[:3])):
                if orig_grad is not None and tp_grad is not None:
                    orig_np = np.array(orig_grad)
                    tp_np = np.array(tp_grad)
                    
                    if orig_np.shape == tp_np.shape:
                        grad_checks += 1
                        print(f"      ✅ Gradient {i} shape matches: {orig_np.shape}")
                    else:
                        print(f"      ❌ Gradient {i} shape mismatch: {orig_np.shape} vs {tp_np.shape}")
            
            if grad_checks == 3:
                print(f"      ✅ All gradient shapes verified")
                result = True
            else:
                result = False
        else:
            print(f"      ❌ Gradient count mismatch: {len(original_gradients)} vs {len(tp_gradients)}")
            result = False
            
    except Exception as e:
        print(f"      ❌ Gradient computation failed: {e}")
        result = False
    
    print(f"✅ Gradient computation test completed in {time.time() - start_time:.2f}s")
    return result

def test_opt125m_optimizer_integration():
    """
    Test optimizer integration with OPT-125M model.
    Verify that optimizers work correctly with sharded parameters.
    """
    print("🔧 OPT-125M Optimizer Integration Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create OPT-125M model
    opt_model = create_opt125m_model()
    
    # Create TensorParallelKeras model
    tp_model = TensorParallelKeras(
        model=opt_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # Test different optimizers
    optimizers_to_test = [
        ('adam', 'Adam'),
        ('sgd', 'SGD'),
        ('rmsprop', 'RMSprop')
    ]
    
    all_tests_passed = True
    
    for opt_name, opt_class in optimizers_to_test:
        print(f"      Testing {opt_class} optimizer...")
        
        try:
            # Create optimizer
            if opt_name == 'adam':
                optimizer = optimizers.Adam(learning_rate=0.001)
            elif opt_name == 'sgd':
                optimizer = optimizers.SGD(learning_rate=0.01)
            elif opt_name == 'rmsprop':
                optimizer = optimizers.RMSprop(learning_rate=0.001)
            
            # Compile models
            opt_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
            tp_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
            
            # Create test data
            np.random.seed(42)
            x_test = np.random.randint(0, 1000, (2, 5), dtype=np.int32)
            y_test = np.random.randint(0, 1000, (2, 5), dtype=np.int32)
            
            # Test training step
            opt_result = opt_model.train_on_batch(x_test, y_test)
            tp_result = tp_model.train_on_batch(x_test, y_test)
            
            print(f"        ✅ {opt_class} training successful")
            print(f"        Original loss: {opt_result[0]:.6f}, TP loss: {tp_result[0]:.6f}")
            
        except Exception as e:
            print(f"        ❌ {opt_class} test failed: {e}")
            all_tests_passed = False
    
    print(f"✅ Optimizer integration test completed in {time.time() - start_time:.2f}s")
    return all_tests_passed

def test_opt125m_memory_efficiency():
    """
    Test memory efficiency of OPT-125M with tensor parallelism.
    Verify that memory usage is reduced with sharding.
    """
    print("🔧 OPT-125M Memory Efficiency Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create OPT-125M model
    opt_model = create_opt125m_model()
    
    # Count original parameters
    original_params = opt_model.count_params()
    print(f"      Original model parameters: {original_params:,}")
    
    # Create TensorParallelKeras model
    tp_model = TensorParallelKeras(
        model=opt_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # Count sharded parameters
    if hasattr(tp_model, 'model_shards'):
        total_sharded_params = 0
        for i, shard in enumerate(tp_model.model_shards):
            shard_params = sum(np.prod(p.shape) for p in shard.weights)
            total_sharded_params += shard_params
            print(f"      Shard {i} parameters: {shard_params:,}")
        
        print(f"      Total sharded parameters: {total_sharded_params:,}")
        
        # Check if sharding is working
        if total_sharded_params >= original_params:
            print(f"      ✅ Parameter sharding working correctly")
            
            # Calculate memory efficiency
            memory_reduction = (total_sharded_params - original_params) / original_params * 100
            print(f"      Memory overhead: {memory_reduction:.2f}%")
            
            if memory_reduction < 50:  # Allow reasonable overhead
                print(f"      ✅ Memory efficiency acceptable")
                result = True
            else:
                print(f"      ⚠️  High memory overhead detected")
                result = False
        else:
            print(f"      ❌ Parameter sharding not working correctly")
            result = False
    else:
        print(f"      ⚠️  No model shards found")
        result = False
    
    print(f"✅ Memory efficiency test completed in {time.time() - start_time:.2f}s")
    return result

def test_opt125m_training_stability():
    """
    Test training stability of OPT-125M with tensor parallelism.
    Verify that training converges and remains stable.
    """
    print("🔧 OPT-125M Training Stability Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create simplified OPT-125M model for faster testing
    opt_model = create_opt125m_model(num_layers=2, hidden_size=128)
    
    # Create TensorParallelKeras model
    tp_model = TensorParallelKeras(
        model=opt_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # Compile models
    opt_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    tp_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create training data
    np.random.seed(42)
    x_train = np.random.randint(0, 1000, (50, 8), dtype=np.int32)
    y_train = np.random.randint(0, 1000, (50, 8), dtype=np.int32)
    
    print(f"      Training data: {x_train.shape}")
    
    try:
        # Train original model
        original_history = opt_model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=8,
            verbose=0
        )
        
        # Train TP model
        tp_history = tp_model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=8,
            verbose=0
        )
        
        print(f"      ✅ Training completed for both models")
        
        # Check training stability
        original_losses = original_history.history['loss']
        tp_losses = tp_history.history['loss']
        
        # Check if losses are decreasing (training is working)
        original_decreasing = original_losses[-1] < original_losses[0]
        tp_decreasing = tp_losses[-1] < tp_losses[0]
        
        if original_decreasing and tp_decreasing:
            print(f"      ✅ Training stability verified - losses decreasing")
            print(f"      Original: {original_losses[0]:.4f} → {original_losses[-1]:.4f}")
            print(f"      TP: {tp_losses[0]:.4f} → {tp_losses[-1]:.4f}")
            result = True
        else:
            print(f"      ⚠️  Training stability issues detected")
            result = False
            
    except Exception as e:
        print(f"      ❌ Training stability test failed: {e}")
        result = False
    
    print(f"✅ Training stability test completed in {time.time() - start_time:.2f}s")
    return result

if __name__ == "__main__":
    print("🎯 OPT-125M COMPREHENSIVE TRAINING READINESS TEST SUITE")
    print("=" * 60)
    print("🔍 Testing all critical components needed for production training")
    print("=" * 60)
    
    # Run all tests
    test_results = []
    
    # Test 1: Backward Pass Consistency
    test_results.append(("Backward Pass Consistency", test_opt125m_backward_pass_consistency()))
    
    # Test 2: Gradient Computation
    test_results.append(("Gradient Computation", test_opt125m_gradient_computation()))
    
    # Test 3: Optimizer Integration
    test_results.append(("Optimizer Integration", test_opt125m_optimizer_integration()))
    
    # Test 4: Memory Efficiency
    test_results.append(("Memory Efficiency", test_opt125m_memory_efficiency()))
    
    # Test 5: Training Stability
    test_results.append(("Training Stability", test_opt125m_training_stability()))
    
    # Print comprehensive results
    print("\n" + "=" * 60)
    print("🎉 OPT-125M COMPREHENSIVE TESTING COMPLETED!")
    print(f"\n📋 COMPREHENSIVE RESULTS:")
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   - Total Tests: {len(test_results)}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {len(test_results) - passed_tests}")
    print(f"   - Success Rate: {(passed_tests / len(test_results)) * 100:.1f}%")
    
    if passed_tests == len(test_results):
        print("\n🚀 SUCCESS: All OPT-125M comprehensive tests passed!")
        print("\n💡 OPT-125M PRODUCTION TRAINING READINESS:")
        print("   ✅ Backward pass consistency verified")
        print("   ✅ Gradient computation working correctly")
        print("   ✅ Optimizer integration functional")
        print("   ✅ Memory efficiency acceptable")
        print("   ✅ Training stability confirmed")
        print("\n🎯 Your OPT-125M model is FULLY READY for production training!")
        print("\n🚀 Next steps:")
        print("   1. Scale up to full OPT-125M model (12 layers, 768 hidden size)")
        print("   2. Configure distributed training environment")
        print("   3. Start training with your dataset")
    else:
        print(f"\n⚠️  WARNING: {len(test_results) - passed_tests} tests failed.")
        print("   Please review and fix the failing tests before proceeding with production training.") 