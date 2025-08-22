"""
Debug script to identify the exact source of embedding weight update differences
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def debug_embedding_issue():
    """Debug embedding weight update differences step by step."""
    
    print("🔍 Debug Embedding Weight Update Differences")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create embedding model
    model = keras.Sequential([
        layers.Embedding(input_dim=1000, output_dim=64, name="embedding"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu', name="dense"),
        layers.Dense(5, activation='softmax', name="output")
    ], name='embedding_debug_model')
    
    # Build the model first
    model.build(input_shape=(None, 16))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    
    # Create test data
    x = np.random.randint(0, 1000, size=(8, 16))
    y = np.random.randint(0, 5, size=(8,))
    
    print(f"📊 Input shape: {x.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Test 1: Single model
    print(f"\n{'='*60}")
    print(f"🧪 Test 1: Single Model")
    print(f"{'='*60}")
    
    # Store initial weights
    initial_weights = [w.numpy().copy() for w in model.weights]
    print(f"✅ Initial weights captured: {len(initial_weights)} parameters")
    
    # Forward pass
    output1 = model(x, training=False)
    print(f"✅ Forward pass completed - Output shape: {output1.shape}")
    
    # Training step
    loss1 = model.train_on_batch(x, y)
    print(f"✅ Training completed - Loss: {loss1:.6f}")
    
    # Get updated weights
    updated_weights1 = [w.numpy().copy() for w in model.weights]
    
    # Test 2: Reset and retrain
    print(f"\n{'='*60}")
    print(f"🧪 Test 2: Reset and Retrain (Same Model)")
    print(f"{'='*60}")
    
    # Reset weights
    print(f"🔧 Resetting weights to initial state")
    for i, weight in enumerate(model.weights):
        weight.assign(initial_weights[i])
    print(f"✅ Weights reset to initial state")
    
    # Reset optimizer state
    print(f"🔧 Resetting optimizer state")
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
    print(f"✅ Optimizer state reset")
    
    # Set seeds again
    np.random.seed(42)
    tf.random.set_seed(42)
    print(f"✅ Seeds reset")
    
    # Forward pass
    output2 = model(x, training=False)
    print(f"✅ Forward pass completed - Output shape: {output2.shape}")
    
    # Training step
    loss2 = model.train_on_batch(x, y)
    print(f"✅ Training completed - Loss: {loss2:.6f}")
    
    # Get updated weights
    updated_weights2 = [w.numpy().copy() for w in model.weights]
    
    # Compare results
    print(f"\n{'='*60}")
    print(f"🔍 Comparing Results")
    print(f"{'='*60}")
    
    # Compare forward pass
    forward_diff = np.abs(output1.numpy() - output2.numpy())
    max_forward_diff = np.max(forward_diff)
    print(f"📊 Forward Pass Comparison:")
    print(f"   Max diff: {max_forward_diff:.2e}")
    
    # Compare loss
    loss_diff = abs(loss1 - loss2)
    print(f"📊 Loss Comparison:")
    print(f"   Loss diff: {loss_diff:.2e}")
    
    # Compare weight updates
    print(f"📊 Weight Update Comparison:")
    for i, (w1, w2) in enumerate(zip(updated_weights1, updated_weights2)):
        weight_diff = np.abs(w1 - w2)
        max_diff = np.max(weight_diff)
        mean_diff = np.mean(weight_diff)
        print(f"   Weight {i} {w1.shape}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    # Test 3: Check if the issue is with embedding specifically
    print(f"\n{'='*60}")
    print(f"🧪 Test 3: Check Embedding Layer Specifically")
    print(f"{'='*60}")
    
    # Reset weights again
    for i, weight in enumerate(model.weights):
        weight.assign(initial_weights[i])
    
    # Reset optimizer state
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
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Test with different input data to see if it's input-dependent
    x_alt = np.random.randint(0, 1000, size=(8, 16))
    np.random.seed(42)  # Reset seed after generating data
    
    # Forward pass with different input
    output3 = model(x_alt, training=False)
    print(f"✅ Forward pass with different input - Output shape: {output3.shape}")
    
    # Training step with different input
    loss3 = model.train_on_batch(x_alt, y)
    print(f"✅ Training with different input - Loss: {loss3:.6f}")
    
    # Get updated weights
    updated_weights3 = [w.numpy().copy() for w in model.weights]
    
    # Compare with original
    print(f"📊 Weight Update Comparison (Different Input):")
    for i, (w1, w3) in enumerate(zip(updated_weights1, updated_weights3)):
        weight_diff = np.abs(w1 - w3)
        max_diff = np.max(weight_diff)
        mean_diff = np.mean(weight_diff)
        print(f"   Weight {i} {w1.shape}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    print("\n" + "=" * 60)
    print("🎯 Debug Complete")

if __name__ == "__main__":
    debug_embedding_issue() 