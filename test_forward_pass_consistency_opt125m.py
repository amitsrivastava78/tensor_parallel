#!/usr/bin/env python3
"""
Forward Pass Consistency Test for OPT-125M Model
The "Golden Test" - Verifies mathematical identity between single CPU and 2-CPU sharded computation
"""

import os
import time
import logging
import numpy as np
import keras
from keras import layers, optimizers

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

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

def test_forward_pass_consistency_opt125m():
    """
    Test forward pass consistency for OPT-125M model.
    
    Goal: Verify that a forward pass of OPT-125M produces the exact same output logits 
    when run on 2 CPUs vs. a single CPU.
    
    Steps:
    1. Load the OPT-125M model on a single CPU and run a forward pass with a fixed input tensor. 
       Store the output logits (the "golden" result).
    2. Instantiate TensorParallelKeras with a 2-CPU mesh and JAX backend.
    3. Run a forward pass with the same fixed input tensor.
    4. Assert: The output logits from the sharded model must be numerically very close 
       (np.allclose) to the "golden" result.
    """
    print("🔧 OPT-125M Forward Pass Consistency Test (Golden Test)")
    print("=" * 60)
    
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting forward pass consistency test...")
    
    # Step 1: Create OPT-125M model on single CPU and get "golden" result
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating OPT-125M model on single CPU...")
    
    # Create OPT-125M model
    opt_model = create_opt125m_model()
    
    # Create fixed input tensor for reproducible testing
    # Use a fixed seed for deterministic results
    np.random.seed(42)
    seq_len = 10
    batch_size = 2
    vocab_size = 50257
    
    # Create fixed input tensor (token IDs)
    test_input = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    print(f"      Fixed input tensor shape: {test_input.shape}")
    print(f"      Input tensor sample: {test_input[0, :5]}...")  # Show first 5 tokens
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Running forward pass on single CPU...")
    
    # Run forward pass on single CPU to get "golden" result
    try:
        golden_output = opt_model(test_input)
        print(f"      ✅ Single CPU forward pass successful")
        print(f"      Golden output shape: {golden_output.shape}")
        print(f"      Golden output dtype: {golden_output.dtype}")
        
        # Store golden result statistics
        golden_mean = np.mean(golden_output)
        golden_std = np.std(golden_output)
        golden_min = np.min(golden_output)
        golden_max = np.max(golden_output)
        
        print(f"      Golden output stats - Mean: {golden_mean:.6f}, Std: {golden_std:.6f}")
        print(f"      Golden output stats - Min: {golden_min:.6f}, Max: {golden_max:.6f}")
        
    except Exception as e:
        print(f"      ❌ Single CPU forward pass failed: {e}")
        return False
    
    # Step 2: Instantiate TensorParallelKeras with 2-CPU mesh and JAX backend
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating TensorParallelKeras with 2-CPU mesh...")
    
    try:
        tp_model = TensorParallelKeras(
            model=opt_model,
            world_size=2,
            distributed_backend='jax'
        )
        print(f"      ✅ TensorParallelKeras created successfully")
        print(f"      World size: {tp_model.world_size}")
        print(f"      Device IDs: {tp_model.device_ids}")
        
    except Exception as e:
        print(f"      ❌ TensorParallelKeras creation failed: {e}")
        return False
    
    # Step 3: Run forward pass with the same fixed input tensor
    print(f"⏱️  {time.time() - start_time:.2f}s: Running forward pass on 2-CPU sharded model...")
    
    try:
        tp_output = tp_model(test_input)
        print(f"      ✅ 2-CPU sharded forward pass successful")
        print(f"      TP output shape: {tp_output.shape}")
        print(f"      TP output dtype: {tp_output.dtype}")
        
        # Store TP result statistics
        tp_mean = np.mean(tp_output)
        tp_std = np.std(tp_output)
        tp_min = np.min(tp_output)
        tp_max = np.max(tp_output)
        
        print(f"      TP output stats - Mean: {tp_mean:.6f}, Std: {tp_std:.6f}")
        print(f"      TP output stats - Min: {tp_min:.6f}, Max: {tp_max:.6f}")
        
    except Exception as e:
        print(f"      ❌ 2-CPU sharded forward pass failed: {e}")
        return False
    
    # Step 4: Assert numerical closeness
    print(f"⏱️  {time.time() - start_time:.2f}s: Verifying numerical consistency...")
    
    # Check output shapes match
    if golden_output.shape != tp_output.shape:
        print(f"      ❌ Shape mismatch: Golden {golden_output.shape} vs TP {tp_output.shape}")
        return False
    else:
        print(f"      ✅ Output shapes match: {golden_output.shape}")
    
    # Check output dtypes match
    if golden_output.dtype != tp_output.dtype:
        print(f"      ❌ Dtype mismatch: Golden {golden_output.dtype} vs TP {tp_output.dtype}")
        return False
    else:
        print(f"      ✅ Output dtypes match: {golden_output.dtype}")
    
    # Convert to numpy arrays for comparison
    golden_np = np.array(golden_output)
    tp_np = np.array(tp_output)
    
    # Check for NaN or Inf values
    if np.any(np.isnan(golden_np)) or np.any(np.isnan(tp_np)):
        print(f"      ❌ NaN values detected in outputs")
        return False
    
    if np.any(np.isinf(golden_np)) or np.any(np.isinf(tp_np)):
        print(f"      ❌ Inf values detected in outputs")
        return False
    
    # Calculate absolute and relative differences
    abs_diff = np.abs(golden_np - tp_np)
    rel_diff = np.abs((golden_np - tp_np) / (np.abs(golden_np) + 1e-8))  # Add small epsilon to avoid division by zero
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    print(f"      Absolute differences - Max: {max_abs_diff:.2e}, Mean: {mean_abs_diff:.2e}")
    print(f"      Relative differences - Max: {max_rel_diff:.2e}, Mean: {mean_rel_diff:.2e}")
    
    # Define tolerance thresholds
    abs_tolerance = 1e-5
    rel_tolerance = 1e-5
    
    # Check if outputs are numerically close
    is_close_abs = np.allclose(golden_np, tp_np, atol=abs_tolerance)
    is_close_rel = np.allclose(golden_np, tp_np, rtol=rel_tolerance)
    
    print(f"      Tolerance thresholds - Absolute: {abs_tolerance:.2e}, Relative: {rel_tolerance:.2e}")
    print(f"      AllClose (absolute): {is_close_abs}")
    print(f"      AllClose (relative): {is_close_rel}")
    
    # Final assertion: outputs must be numerically very close
    if is_close_abs and is_close_rel:
        print(f"      ✅ NUMERICAL CONSISTENCY VERIFIED!")
        print(f"      🎯 Golden Test PASSED: 2-CPU sharded computation is mathematically identical to single-CPU computation")
        result = True
    else:
        print(f"      ❌ NUMERICAL CONSISTENCY FAILED!")
        print(f"      🚨 Golden Test FAILED: 2-CPU sharded computation differs from single-CPU computation")
        
        # Show detailed differences for debugging
        print(f"      Detailed analysis:")
        print(f"        - Max absolute difference: {max_abs_diff:.2e} (threshold: {abs_tolerance:.2e})")
        print(f"        - Max relative difference: {max_rel_diff:.2e} (threshold: {rel_tolerance:.2e})")
        
        # Find indices of largest differences
        max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        max_rel_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        
        print(f"        - Largest absolute diff at {max_abs_idx}: Golden={golden_np[max_abs_idx]:.6f}, TP={tp_np[max_abs_idx]:.6f}")
        print(f"        - Largest relative diff at {max_rel_idx}: Golden={golden_np[max_rel_idx]:.6f}, TP={tp_np[max_rel_idx]:.6f}")
        
        result = False
    
    print(f"✅ Forward pass consistency test completed in {time.time() - start_time:.2f}s")
    return result

def test_forward_pass_consistency_with_different_inputs():
    """
    Test forward pass consistency with different input sequences to ensure robustness.
    """
    print("\n🔧 OPT-125M Forward Pass Consistency Test - Multiple Inputs")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create OPT-125M model
    opt_model = create_opt125m_model()
    
    # Create TensorParallelKeras model
    tp_model = TensorParallelKeras(
        model=opt_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    # Test different input configurations
    test_configs = [
        {"batch_size": 1, "seq_len": 5, "description": "Single batch, short sequence"},
        {"batch_size": 2, "seq_len": 10, "description": "Small batch, medium sequence"},
        {"batch_size": 4, "seq_len": 15, "description": "Medium batch, long sequence"},
        {"batch_size": 1, "seq_len": 20, "description": "Single batch, long sequence"},
    ]
    
    vocab_size = 50257
    np.random.seed(42)  # Fixed seed for reproducibility
    
    all_tests_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\n   Test {i+1}: {config['description']}")
        print(f"      Input shape: ({config['batch_size']}, {config['seq_len']})")
        
        # Create input tensor
        test_input = np.random.randint(0, vocab_size, (config['batch_size'], config['seq_len']), dtype=np.int32)
        
        try:
            # Run forward passes
            golden_output = opt_model(test_input)
            tp_output = tp_model(test_input)
            
            # Check numerical consistency
            golden_np = np.array(golden_output)
            tp_np = np.array(tp_output)
            
            is_close = np.allclose(golden_np, tp_np, atol=1e-5, rtol=1e-5)
            
            if is_close:
                print(f"      ✅ PASSED - Numerical consistency verified")
            else:
                print(f"      ❌ FAILED - Numerical inconsistency detected")
                all_tests_passed = False
                
        except Exception as e:
            print(f"      ❌ ERROR - Test failed with exception: {e}")
            all_tests_passed = False
    
    print(f"\n✅ Multiple input consistency test completed in {time.time() - start_time:.2f}s")
    return all_tests_passed

if __name__ == "__main__":
    print("🎯 OPT-125M FORWARD PASS CONSISTENCY TEST SUITE")
    print("=" * 60)
    print("🔍 This is the 'Golden Test' - verifying mathematical identity between")
    print("   single CPU and 2-CPU sharded computation on JAX backend")
    print("=" * 60)
    
    # Run the main forward pass consistency test
    main_test_passed = test_forward_pass_consistency_opt125m()
    
    # Run additional robustness tests
    robustness_tests_passed = test_forward_pass_consistency_with_different_inputs()
    
    # Print comprehensive results
    print("\n" + "=" * 60)
    print("🎉 FORWARD PASS CONSISTENCY TESTING COMPLETED!")
    print(f"\n📋 COMPREHENSIVE RESULTS:")
    
    test_results = [
        ("Main Forward Pass Consistency (Golden Test)", main_test_passed),
        ("Robustness Tests (Multiple Inputs)", robustness_tests_passed)
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
        print("\n🚀 SUCCESS: All forward pass consistency tests passed!")
        print("\n💡 OPT-125M FORWARD PASS VERIFICATION:")
        print("   ✅ Mathematical identity verified between single CPU and 2-CPU sharded computation")
        print("   ✅ JAX backend integration working correctly")
        print("   ✅ Tensor parallelism implementation is numerically correct")
        print("\n🎯 Your OPT-125M model is READY for distributed training!")
    else:
        print(f"\n⚠️  WARNING: {len(test_results) - passed_tests} tests failed.")
        print("   Please review and fix the failing tests before proceeding with OPT-125M training.") 