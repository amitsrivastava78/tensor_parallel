"""
Test MLP model only to verify optimizer state reset fix
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def test_mlp_only():
    """Test MLP model with optimizer state reset."""
    
    print("🚀 MLP Model Test with Optimizer State Reset")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create MLP model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,), name='dense_1'),
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dense(10, activation='softmax', name='output')
    ], name='mlp_model')
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    print(f"✅ Model compiled for consistent state")
    
    # Create test data
    batch_size = 8
    x = np.random.random((batch_size, 32)).astype('float32')
    y = np.random.randint(0, 10, size=(batch_size,))
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Test 1: Single model test
    print(f"\n{'='*60}")
    print(f"🧪 Testing MLP Model (Single Device)")
    print(f"{'='*60}")
    
    # Store initial weights
    initial_weights = [w.numpy().copy() for w in model.weights]
    print(f"✅ Initial weights captured: {len(initial_weights)} parameters")
    
    # Forward pass
    output = model(x, training=False)
    print(f"✅ Forward pass completed - Output shape: {output.shape}")
    
    # Training step
    loss = model.train_on_batch(x, y)
    print(f"✅ Training completed - Loss: {loss:.6f}")
    
    # Get updated weights
    updated_weights = [w.numpy().copy() for w in model.weights]
    
    single_results = {
        'initial_weights': initial_weights,
        'output': output,
        'loss': loss,
        'updated_weights': updated_weights
    }
    
    # Test 2: Tensor parallel test with optimizer state reset
    print(f"\n{'='*60}")
    print(f"🧪 Testing MLP Model (Tensor Parallel, 2 devices)")
    print(f"{'='*60}")
    
    # CRITICAL: Reset weights AND optimizer state for fair comparison
    print(f"🔧 Resetting weights to initial state for fair comparison")
    for i, weight in enumerate(model.weights):
        weight.assign(initial_weights[i])
    print(f"✅ Weights reset to initial state")
    
    # CRITICAL: Reset optimizer state to ensure identical training conditions
    print(f"🔧 Resetting optimizer state for fair comparison")
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        # Reset Adam optimizer state (momentum, variance)
        if hasattr(model.optimizer, 'momentums'):
            for momentum in model.optimizer.momentums:
                momentum.assign(tf.zeros_like(momentum))
        if hasattr(model.optimizer, 'velocities'):
            for velocity in model.optimizer.velocities:
                velocity.assign(tf.zeros_like(velocity))
        if hasattr(model.optimizer, 'beta_1_power'):
            model.optimizer.beta_1_power.assign(1.0)
        if hasattr(model.optimizer, 'beta_2_power'):
            model.optimizer.beta_2_power.assign(1.0)
        print(f"✅ Optimizer state reset to initial state")
    
    try:
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        # Create wrapper
        tp_model = TensorParallelKeras(model, world_size=2)
        
        # Forward pass
        tp_output = tp_model(x, training=False)
        print(f"✅ Forward pass completed - Output shape: {tp_output.shape}")
        
        # Training step
        tp_loss = tp_model.train_on_batch(x, y)
        print(f"✅ Training completed - Loss: {tp_loss:.6f}")
        
        # Get updated weights from original model
        tp_updated_weights = [w.numpy().copy() for w in model.weights]
        
        tp_results = {
            'output': tp_output,
            'loss': tp_loss,
            'updated_weights': tp_updated_weights
        }
        
        # Compare results
        print(f"\n{'='*60}")
        print(f"🔍 Comparing Results for MLP Model")
        print(f"{'='*60}")
        
        # Compare forward pass
        forward_diff = np.abs(single_results['output'].numpy() - tp_results['output'].numpy())
        max_forward_diff = np.max(forward_diff)
        mean_forward_diff = np.mean(forward_diff)
        
        print(f"📊 Forward Pass Comparison:")
        print(f"   Max diff: {max_forward_diff:.2e}")
        print(f"   Mean diff: {mean_forward_diff:.2e}")
        
        # Compare loss
        loss_diff = abs(single_results['loss'] - tp_results['loss'])
        print(f"📊 Loss Comparison:")
        print(f"   Loss diff: {loss_diff:.2e}")
        
        # Compare weight updates
        print(f"📊 Weight Update Comparison:")
        for i, (single_w, tp_w) in enumerate(zip(single_results['updated_weights'], tp_results['updated_weights'])):
            weight_diff = np.abs(single_w - tp_w)
            max_diff = np.max(weight_diff)
            mean_diff = np.mean(weight_diff)
            print(f"   Weight {i} {single_w.shape}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        
        # Results summary
        print(f"\n📊 Results Summary for MLP:")
        forward_pass = "✅ PASS" if max_forward_diff < 1e-15 else "❌ FAIL"
        loss_matching = "✅ PASS" if loss_diff < 1e-15 else "❌ FAIL"
        weight_updates = "✅ PASS" if max_diff < 1e-6 else "❌ FAIL"
        
        print(f"{forward_pass}: Forward pass (max diff: {max_forward_diff:.2e})")
        print(f"{loss_matching}: Loss matching (diff: {loss_diff:.2e})")
        print(f"{weight_updates}: Weight updates (max diff: {max_diff:.2e})")
        
        if max_forward_diff < 1e-15 and loss_diff < 1e-15:
            print(f"🎯 Overall MLP Test: ✅ PERFECT NUMERICAL IDENTITY ACHIEVED!")
        else:
            print(f"🎯 Overall MLP Test: ❌ Some tests failed")
        
    except ImportError as e:
        print(f"❌ Could not import TensorParallelKeras: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 MLP Test Complete")

if __name__ == "__main__":
    test_mlp_only() 