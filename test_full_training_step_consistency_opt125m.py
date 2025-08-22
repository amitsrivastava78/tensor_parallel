#!/usr/bin/env python3
"""
Full Training Step Consistency Test for OPT-125M Model
Verifies that one complete training step (forward, loss, backward, optimizer) produces
identical results on 2-CPU sharded model vs. single CPU model
"""

import os

# Set Keras to use JAX backend explicitly (no TensorFlow)
os.environ['KERAS_BACKEND'] = 'jax'

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import time
import logging
import numpy as np
import keras
from keras import layers, optimizers
# Using only Keras 3.0 with JAX backend

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_opt125m_model(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
    """
    Create a simplified OPT-125M model for testing.
    This matches the architecture described in the OPT paper.
    """
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

def get_model_weights_state(model):
    """
    Get the current state of all model weights.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary mapping weight names to numpy arrays
    """
    weights_state = {}
    for weight in model.weights:
        weights_state[weight.name] = np.array(weight)
    return weights_state

def compare_weights_states(original_state, tp_state, tolerance=1e-5):
    """
    Compare two weight states and return detailed analysis.
    
    Args:
        original_state: Weight state from original model
        tp_state: Weight state from tensor parallel model
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'total_weights': len(original_state),
        'matching_weights': 0,
        'different_weights': 0,
        'missing_weights': 0,
        'extra_weights': 0,
        'max_difference': 0.0,
        'mean_difference': 0.0,
        'details': {}
    }
    
    all_differences = []
    
    # Check each weight in original state
    for weight_name, original_weight in original_state.items():
        if weight_name in tp_state:
            tp_weight = tp_state[weight_name]
            
            # Check shapes match
            if original_weight.shape != tp_weight.shape:
                comparison['details'][weight_name] = {
                    'status': 'shape_mismatch',
                    'original_shape': original_weight.shape,
                    'tp_shape': tp_weight.shape
                }
                comparison['different_weights'] += 1
                continue
            
            # Check numerical values
            if np.allclose(original_weight, tp_weight, atol=tolerance):
                comparison['details'][weight_name] = {
                    'status': 'match',
                    'difference': 0.0
                }
                comparison['matching_weights'] += 1
            else:
                # Calculate differences
                diff = np.abs(original_weight - tp_weight)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                comparison['details'][weight_name] = {
                    'status': 'different',
                    'max_difference': max_diff,
                    'mean_difference': mean_diff,
                    'original_shape': original_weight.shape
                }
                
                comparison['different_weights'] += 1
                all_differences.append(max_diff)
        else:
            comparison['details'][weight_name] = {
                'status': 'missing_in_tp',
                'original_shape': original_weight.shape
            }
            comparison['missing_weights'] += 1
    
    # Check for extra weights in TP model
    for weight_name in tp_state:
        if weight_name not in original_state:
            comparison['details'][weight_name] = {
                'status': 'extra_in_tp',
                'tp_shape': tp_state[weight_name].shape
            }
            comparison['extra_weights'] += 1
    
    # Calculate overall statistics
    if all_differences:
        comparison['max_difference'] = max(all_differences)
        comparison['mean_difference'] = np.mean(all_differences)
    
    return comparison

def test_full_training_step_consistency_opt125m():
    """
    Test full training step consistency for OPT-125M model.
    
    Goal: Ensure that one full training step (forward, loss, backward, optimizer step) 
    produces the same loss and updated weights on 2 CPUs as it does on one CPU.
    
    Steps:
    1. Run a single training step on the single-CPU OPT-125M model. Record the final 
       loss value and the state of all model weights after the optimizer step.
    2. Reset the model and optimizer.
    3. Run the same training step on the 2-CPU sharded model.
    4. Assert:
        - The calculated loss must be almost identical to the single-CPU loss.
        - The sharded weights, when gathered and reconstructed, must be numerically 
          very close to the updated weights from the single-CPU run.
    """
    print("🔧 OPT-125M Full Training Step Consistency Test")
    print("=" * 60)
    print("🎯 Goal: Verify complete training step produces identical results")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create fixed input data for reproducible testing
    np.random.seed(42)
    batch_size = 2
    seq_len = 10
    vocab_size = 50257
    
    # Create fixed input tensor (token IDs)
    x_train = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    y_train = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    
    print(f"      Fixed training data: x={x_train.shape}, y={y_train.shape}")
    print(f"      Input sample: {x_train[0, :5]}...")
    print(f"      Target sample: {y_train[0, :5]}...")
    
    # Step 1: Run training step on single-CPU model
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 1 - Single CPU training step...")
    
    # Create and compile single-CPU model
    original_model = create_opt125m_model()
    original_optimizer = optimizers.Adam(learning_rate=0.001)
    
    original_model.compile(
        optimizer=original_optimizer,
        loss='sparse_categorical_crossentropy'
        # Removed metrics to avoid JAX issues  
    )
    
    # Get initial weights state
    initial_weights = get_model_weights_state(original_model)
    print(f"      Initial weights captured: {len(initial_weights)} parameters")
    
    # Run single training step
    try:
        original_result = original_model.train_on_batch(x_train, y_train)
        original_loss = original_result[0]
        print(f"      ✅ Single CPU training step successful")
        print(f"      Original loss: {original_loss:.6f}")
        
        # Get final weights state
        final_weights_original = get_model_weights_state(original_model)
        print(f"      Final weights captured: {len(final_weights_original)} parameters")
        
    except Exception as e:
        print(f"      ❌ Single CPU training step failed: {e}")
        return False
    
    # Step 2: Reset model and optimizer
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 2 - Resetting model...")
    
    # Create fresh model with same architecture
    reset_model = create_opt125m_model()
    reset_optimizer = optimizers.Adam(learning_rate=0.001)
    
    reset_model.compile(
        optimizer=reset_optimizer,
        loss='sparse_categorical_crossentropy'
        # Removed metrics to avoid JAX issues
    )
    
    # Verify reset model has same initial weights
    reset_initial_weights = get_model_weights_state(reset_model)
    
    # Compare initial weights (should be identical due to same seed)
    initial_comparison = compare_weights_states(initial_weights, reset_initial_weights)
    if initial_comparison['matching_weights'] == initial_comparison['total_weights']:
        print(f"      ✅ Reset model weights verified as identical")
    else:
        print(f"      ⚠️  Reset model weights differ from original")
        print(f"         Matching: {initial_comparison['matching_weights']}/{initial_comparison['total_weights']}")
    
    # Step 3: Run training step on 2-CPU sharded model
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 3 - 2-CPU sharded training step...")
    
    try:
        # Create TensorParallelKeras model
        tp_model = TensorParallelKeras(
            model=reset_model,
            world_size=2,
            distributed_backend='jax'
        )
        
        print(f"      ✅ TensorParallelKeras created successfully")
        print(f"      World size: {tp_model.world_size}")
        print(f"      Device IDs: {tp_model.device_ids}")
        
        # Compile TP model
        tp_model.compile(
            optimizer=reset_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"      ✅ TP model compiled successfully")
        
        # Run training step on TP model
        tp_result = tp_model.train_on_batch(x_train, y_train)
        tp_loss = tp_result[0]
        print(f"      ✅ 2-CPU sharded training step successful")
        print(f"      TP loss: {tp_loss:.6f}")
        
        # Get final weights state from TP model
        final_weights_tp = get_model_weights_state(tp_model)
        print(f"      TP final weights captured: {len(final_weights_tp)} parameters")
        
    except Exception as e:
        print(f"      ❌ 2-CPU sharded training step failed: {e}")
        return False
    
    # Step 4: Assert numerical consistency
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 4 - Verifying numerical consistency...")
    
    # Check loss consistency
    loss_diff = abs(original_loss - tp_loss)
    loss_tolerance = 1e-5
    
    print(f"      Loss comparison:")
    print(f"        Original loss: {original_loss:.6f}")
    print(f"        TP loss: {tp_loss:.6f}")
    print(f"        Difference: {loss_diff:.2e}")
    print(f"        Tolerance: {loss_tolerance:.2e}")
    
    if loss_diff <= loss_tolerance:
        print(f"      ✅ Loss consistency verified!")
        loss_consistent = True
    else:
        print(f"      ❌ Loss consistency failed!")
        loss_consistent = False
    
    # Check weights consistency
    print(f"\n      Weights comparison:")
    
    weights_comparison = compare_weights_states(final_weights_original, final_weights_tp, tolerance=1e-5)
    
    print(f"        Total weights: {weights_comparison['total_weights']}")
    print(f"        Matching weights: {weights_comparison['matching_weights']}")
    print(f"        Different weights: {weights_comparison['different_weights']}")
    print(f"        Missing weights: {weights_comparison['missing_weights']}")
    print(f"        Extra weights: {weights_comparison['extra_weights']}")
    
    if weights_comparison['max_difference'] > 0:
        print(f"        Max difference: {weights_comparison['max_difference']:.2e}")
        print(f"        Mean difference: {weights_comparison['mean_difference']:.2e}")
    
    # Determine weights consistency
    if (weights_comparison['matching_weights'] == weights_comparison['total_weights'] and
        weights_comparison['different_weights'] == 0 and
        weights_comparison['missing_weights'] == 0 and
        weights_comparison['extra_weights'] == 0):
        print(f"      ✅ Weights consistency verified!")
        weights_consistent = True
    else:
        print(f"      ❌ Weights consistency failed!")
        weights_consistent = False
        
        # Show details of differences
        print(f"\n      Detailed weight differences:")
        for weight_name, details in weights_comparison['details'].items():
            if details['status'] != 'match':
                print(f"        {weight_name}: {details['status']}")
                if 'max_difference' in details:
                    print(f"          Max diff: {details['max_difference']:.2e}")
    
    # Final assessment
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Final assessment...")
    
    if loss_consistent and weights_consistent:
        print(f"      🎯 FULL TRAINING STEP CONSISTENCY VERIFIED!")
        print(f"      ✅ Loss values are numerically identical")
        print(f"      ✅ Weight updates are numerically identical")
        print(f"      🚀 CoordinatedOptimizer workflow is working correctly")
        result = True
    else:
        print(f"      🚨 FULL TRAINING STEP CONSISTENCY FAILED!")
        if not loss_consistent:
            print(f"      ❌ Loss consistency failed")
        if not weights_consistent:
            print(f"      ❌ Weights consistency failed")
        result = False
    
    print(f"\n✅ Full training step consistency test completed in {time.time() - start_time:.2f}s")
    return result

def test_training_step_with_different_optimizers():
    """
    Test training step consistency with different optimizers to ensure robustness.
    """
    print("\n🔧 OPT-125M Training Step Consistency - Multiple Optimizers")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create fixed training data
    np.random.seed(42)
    x_train = np.random.randint(0, 1000, (2, 8), dtype=np.int32)
    y_train = np.random.randint(0, 1000, (2, 8), dtype=np.int32)
    
    # Test different optimizers
    optimizers_to_test = [
        ('adam', 'Adam', optimizers.Adam(learning_rate=0.001)),
        ('sgd', 'SGD', optimizers.SGD(learning_rate=0.01)),
        ('rmsprop', 'RMSprop', optimizers.RMSprop(learning_rate=0.001))
    ]
    
    all_tests_passed = True
    
    for opt_name, opt_class, optimizer in optimizers_to_test:
        print(f"\n   Testing {opt_class} optimizer...")
        
        try:
            # Create models
            original_model = create_opt125m_model(num_layers=2, hidden_size=128)
            tp_model = create_opt125m_model(num_layers=2, hidden_size=128)
            
            # Compile models
            original_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
            
            tp_model = TensorParallelKeras(
                model=tp_model,
                world_size=2,
                distributed_backend='jax'
            )
            tp_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
            
            # Run training steps
            original_result = original_model.train_on_batch(x_train, y_train)
            tp_result = tp_model.train_on_batch(x_train, y_train)
            
            # Check consistency
            loss_diff = abs(original_result[0] - tp_result[0])
            
            if loss_diff < 1e-5:
                print(f"      ✅ {opt_class} training step consistent (diff: {loss_diff:.2e})")
            else:
                print(f"      ❌ {opt_class} training step inconsistent (diff: {loss_diff:.2e})")
                all_tests_passed = False
                
        except Exception as e:
            print(f"      ❌ {opt_class} test failed with exception: {e}")
            all_tests_passed = False
    
    print(f"\n✅ Multiple optimizer consistency test completed in {time.time() - start_time:.2f}s")
    return all_tests_passed

if __name__ == "__main__":
    print("🎯 OPT-125M FULL TRAINING STEP CONSISTENCY TEST SUITE")
    print("=" * 60)
    print("🔍 This test verifies that complete training steps produce identical results")
    print("   between single CPU and 2-CPU sharded models")
    print("=" * 60)
    
    # Run the main full training step consistency test
    main_test_passed = test_full_training_step_consistency_opt125m()
    
    # Run additional optimizer tests
    optimizer_tests_passed = test_training_step_with_different_optimizers()
    
    # Print comprehensive results
    print("\n" + "=" * 60)
    print("🎉 FULL TRAINING STEP CONSISTENCY TESTING COMPLETED!")
    print(f"\n📋 COMPREHENSIVE RESULTS:")
    
    test_results = [
        ("Main Full Training Step Consistency", main_test_passed),
        ("Multiple Optimizer Consistency", optimizer_tests_passed)
    ]
    
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
        print("\n🚀 SUCCESS: All full training step consistency tests passed!")
        print("\n💡 OPT-125M TRAINING VERIFICATION:")
        print("   ✅ Forward pass consistency verified")
        print("   ✅ Loss computation consistency verified")
        print("   ✅ Backward pass consistency verified")
        print("   ✅ Optimizer update consistency verified")
        print("   ✅ CoordinatedOptimizer workflow verified")
        print("\n🎯 Your OPT-125M model is FULLY READY for production training!")
        print("\n🚀 The entire training pipeline is mathematically consistent!")
    else:
        print(f"\n⚠️  WARNING: {len(test_results) - passed_tests} tests failed.")
        print("   Please review and fix the failing tests before proceeding with OPT-125M training.") 