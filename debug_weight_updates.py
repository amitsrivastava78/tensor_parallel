"""
Debug script to identify exact source of weight update differences
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

def debug_weight_updates():
    """Debug weight update differences step by step."""
    
    print("🔍 Debug Weight Update Differences")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create simple model
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(5,), name='dense_1'),
        layers.Dense(5, activation='softmax', name='output')
    ], name='debug_model')
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    
    # Create test data
    x = np.random.random((4, 5)).astype('float32')
    y = np.random.randint(0, 5, size=(4,))
    
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
    
    # Test 3: Check if gradients are identical
    print(f"\n{'='*60}")
    print(f"🧪 Test 3: Check Gradient Computation")
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
    
    # Compute gradients manually
    with tf.GradientTape() as tape:
        output = model(x, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y, output)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    print(f"✅ Gradients computed manually")
    
    # Check gradient values
    for i, grad in enumerate(gradients):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            print(f"   Gradient {i} norm: {grad_norm:.6f}")
    
    print("\n" + "=" * 60)
    print("🎯 Debug Complete")

if __name__ == "__main__":
    debug_weight_updates() 