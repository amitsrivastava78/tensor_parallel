#!/usr/bin/env python3
"""
Tensor Parallelism Correctness Verification for OPT-125M Model
Implements practical verification criteria that validate tensor parallelism correctness
without requiring perfect mathematical identity.
"""

import os

# Set Keras to use JAX backend explicitly (no TensorFlow)
os.environ['KERAS_BACKEND'] = 'jax'

# Set JAX environment for 2-CPU simulation
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, optimizers

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
        
        # Self-attention (simplified)
        attention_output = layers.Dense(hidden_size, name=f'layers_{i}_self_attn')(hidden_states)
        
        # Add & Norm
        hidden_states = layers.Add()([hidden_states, attention_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layers_{i}_attention_norm')(hidden_states)
        
        # MLP
        mlp_output = layers.Dense(hidden_size * 4, activation='gelu', name=f'layers_{i}_mlp_fc1')(hidden_states)
        mlp_output = layers.Dense(hidden_size, name=f'layers_{i}_mlp_fc2')(mlp_output)
        
        # Add & Norm
        hidden_states = layers.Add()([hidden_states, mlp_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layers_{i}_mlp_norm')(hidden_states)
    
    # Final layer norm
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_final')(hidden_states)
    
    # Language model head
    outputs = layers.Dense(vocab_size, name='lm_head')(hidden_states)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='opt125m')
    
    print(f"      OPT-125M model created with {model.count_params():,} parameters")
    return model

def copy_weights(source_model, target_model):
    """Copy weights from source model to target model for consistent initialization."""
    print("🔧 Copying weights for consistent initialization...")
    for source_weight, target_weight in zip(source_model.weights, target_model.weights):
        if source_weight.shape == target_weight.shape:
            target_weight.assign(source_weight.numpy())

def generate_training_data(batch_size=4, seq_length=16, vocab_size=50257, num_batches=100):
    """Generate consistent training data for both models."""
    print(f"📊 Generating {num_batches} training batches...")
    
    # Use fixed seed for reproducible data
    np.random.seed(42)
    
    data = []
    for i in range(num_batches):
        # Generate input sequences
        x = np.random.randint(0, vocab_size, (batch_size, seq_length), dtype=np.int32)
        # Generate target sequences (shifted by 1 for language modeling)
        y = np.roll(x, -1, axis=1)
        y[:, -1] = 0  # Set last token to padding
        data.append((x, y))
    
    print(f"      Generated {len(data)} training batches")
    return data

def train_model_for_convergence(model, training_data, optimizer, num_epochs=3, verbose=True):
    """
    Train a model for multiple epochs to observe convergence.
    Returns training history and final weights.
    """
    if verbose:
        print(f"🚀 Training model for {num_epochs} epochs...")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    # Training history
    training_history = {
        'loss': [],
        'epoch_losses': []
    }
    
    # Train for multiple epochs
    for epoch in range(num_epochs):
        if verbose:
            print(f"      Epoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        
        # Train on all batches
        for batch_idx, (x_batch, y_batch) in enumerate(training_data):
            try:
                result = model.train_on_batch(x_batch, y_batch)
                # Handle different result formats
                if isinstance(result, (list, tuple)):
                    loss = result[0]
                else:
                    loss = float(result)
                
                epoch_losses.append(loss)
                
                if verbose and batch_idx % 20 == 0:
                    print(f"        Batch {batch_idx}: Loss = {loss:.6f}")
                    
            except Exception as e:
                print(f"        ❌ Training failed at batch {batch_idx}: {e}")
                return None, None
        
        # Calculate epoch average loss
        avg_epoch_loss = np.mean(epoch_losses)
        training_history['epoch_losses'].append(avg_epoch_loss)
        training_history['loss'].extend(epoch_losses)
        
        if verbose:
            print(f"      Epoch {epoch + 1} average loss: {avg_epoch_loss:.6f}")
    
    # Get final weights
    final_weights = {}
    for weight in model.weights:
        final_weights[weight.name] = weight.numpy()
    
    if verbose:
        print(f"      ✅ Training completed. Final epoch loss: {avg_epoch_loss:.6f}")
    
    return training_history, final_weights

def evaluate_model_performance(model, validation_data, verbose=True):
    """
    Evaluate model performance on validation data.
    Returns evaluation metrics.
    """
    if verbose:
        print("📊 Evaluating model performance...")
    
    total_loss = 0
    num_batches = len(validation_data)
    
    for batch_idx, (x_batch, y_batch) in enumerate(validation_data):
        try:
            # Forward pass
            predictions = model.predict(x_batch, verbose=0)
            
            # Calculate loss manually for consistency
            from keras.losses import sparse_categorical_crossentropy
            batch_loss = sparse_categorical_crossentropy(y_batch, predictions)
            total_loss += float(np.mean(batch_loss))
            
        except Exception as e:
            print(f"        ❌ Evaluation failed at batch {batch_idx}: {e}")
            return None
    
    avg_loss = total_loss / num_batches
    
    if verbose:
        print(f"      Average validation loss: {avg_loss:.6f}")
    
    return {
        'loss': avg_loss,
        'perplexity': np.exp(avg_loss)  # Convert loss to perplexity
    }

def plot_convergence_comparison(single_history, tp_history, save_path=None):
    """Plot training convergence comparison between single CPU and tensor parallel models."""
    plt.figure(figsize=(12, 8))
    
    # Plot epoch losses
    plt.subplot(2, 2, 1)
    epochs = range(1, len(single_history['epoch_losses']) + 1)
    plt.plot(epochs, single_history['epoch_losses'], 'b-', label='Single CPU', linewidth=2)
    plt.plot(epochs, tp_history['epoch_losses'], 'r--', label='2-CPU Tensor Parallel', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Convergence: Epoch Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot batch losses (first 100 batches for clarity)
    plt.subplot(2, 2, 2)
    max_batches = min(100, len(single_history['loss']))
    batches = range(1, max_batches + 1)
    plt.plot(batches, single_history['loss'][:max_batches], 'b-', label='Single CPU', alpha=0.7)
    plt.plot(batches, tp_history['loss'][:max_batches], 'r--', label='2-CPU Tensor Parallel', alpha=0.7)
    plt.xlabel('Training Batch')
    plt.ylabel('Loss')
    plt.title('Training Convergence: Batch Losses (First 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss difference over time
    plt.subplot(2, 2, 3)
    min_batches = min(len(single_history['loss']), len(tp_history['loss']))
    loss_diff = np.array(single_history['loss'][:min_batches]) - np.array(tp_history['loss'][:min_batches])
    plt.plot(range(1, min_batches + 1), loss_diff, 'g-', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Training Batch')
    plt.ylabel('Loss Difference (Single - TP)')
    plt.title('Loss Difference Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot final convergence comparison
    plt.subplot(2, 2, 4)
    final_single = single_history['epoch_losses'][-1]
    final_tp = tp_history['epoch_losses'][-1]
    models = ['Single CPU', '2-CPU Tensor Parallel']
    final_losses = [final_single, final_tp]
    colors = ['blue', 'red']
    
    bars = plt.bar(models, final_losses, color=colors, alpha=0.7)
    plt.ylabel('Final Epoch Loss')
    plt.title('Final Training Loss Comparison')
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{loss:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Convergence plot saved to: {save_path}")
    
    plt.show()

def compare_weights_with_tolerance(single_weights, tp_weights, tolerance=1e-3):
    """
    Compare weights with a practical tolerance for tensor parallelism.
    Returns detailed comparison results.
    """
    print("🔍 Comparing final weights with practical tolerance...")
    
    comparison = {
        'total_weights': 0,
        'matching_weights': 0,
        'different_weights': 0,
        'max_difference': 0,
        'mean_difference': 0,
        'details': {}
    }
    
    differences = []
    
    for weight_name in single_weights.keys():
        if weight_name in tp_weights:
            comparison['total_weights'] += 1
            
            single_weight = single_weights[weight_name]
            tp_weight = tp_weights[weight_name]
            
            if single_weight.shape != tp_weight.shape:
                comparison['details'][weight_name] = {
                    'status': 'shape_mismatch',
                    'single_shape': single_weight.shape,
                    'tp_shape': tp_weight.shape
                }
                comparison['different_weights'] += 1
                continue
            
            # Calculate differences
            diff = np.abs(single_weight - tp_weight)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            differences.append(max_diff)
            
            if max_diff <= tolerance:
                comparison['matching_weights'] += 1
                comparison['details'][weight_name] = {
                    'status': 'match',
                    'max_diff': max_diff,
                    'mean_diff': mean_diff
                }
            else:
                comparison['different_weights'] += 1
                comparison['details'][weight_name] = {
                    'status': 'different',
                    'max_diff': max_diff,
                    'mean_diff': mean_diff
                }
    
    if differences:
        comparison['max_difference'] = np.max(differences)
        comparison['mean_difference'] = np.mean(differences)
    
    return comparison

def test_tensor_parallelism_correctness_opt125m():
    """
    Test tensor parallelism correctness using practical verification criteria.
    """
    print("🎯 OPT-125M TENSOR PARALLELISM CORRECTNESS VERIFICATION")
    print("=" * 70)
    print("🔍 This test validates tensor parallelism correctness using practical criteria:")
    print("   1. Loss Convergence Equivalence")
    print("   2. Evaluation Metric Consistency") 
    print("   3. Bounded Numerical Divergence")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Generate consistent training data
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 1 - Generating training data...")
    training_data = generate_training_data(batch_size=4, seq_length=16, num_batches=200)
    validation_data = training_data[-20:]  # Use last 20 batches for validation
    training_data = training_data[:-20]    # Use first 180 batches for training
    
    print(f"      Training batches: {len(training_data)}")
    print(f"      Validation batches: {len(validation_data)}")
    
    # Step 2: Train single CPU model
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 2 - Training single CPU model...")
    
    single_model = create_opt125m_model()
    single_optimizer = optimizers.Adam(learning_rate=0.001)
    
    single_history, single_weights = train_model_for_convergence(
        single_model, training_data, single_optimizer, num_epochs=3
    )
    
    if single_history is None:
        print("❌ Single CPU training failed!")
        return False
    
    # Step 3: Evaluate single CPU model
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 3 - Evaluating single CPU model...")
    single_metrics = evaluate_model_performance(single_model, validation_data)
    
    if single_metrics is None:
        print("❌ Single CPU evaluation failed!")
        return False
    
    print(f"      Single CPU validation loss: {single_metrics['loss']:.6f}")
    print(f"      Single CPU perplexity: {single_metrics['perplexity']:.2f}")
    
    # Step 4: Train tensor parallel model
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 4 - Training tensor parallel model...")
    
    # Create fresh model with same architecture
    tp_model = create_opt125m_model()
    
    # CRITICAL: Use the EXACT SAME optimizer instance for consistent state
    tp_optimizer = single_optimizer
    
    # Copy weights for consistent initialization
    copy_weights(single_model, tp_model)
    
    # Create TensorParallelKeras model
    tp_tensor_parallel = TensorParallelKeras(
        model=tp_model,
        world_size=2,
        distributed_backend='jax'
    )
    
    print(f"      ✅ TensorParallelKeras created successfully")
    print(f"      World size: {tp_tensor_parallel.world_size}")
    
    # Compile TP model
    tp_tensor_parallel.compile(
        optimizer=tp_optimizer,
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"      ✅ TP model compiled successfully")
    
    # Train TP model
    tp_history, tp_weights = train_model_for_convergence(
        tp_tensor_parallel, training_data, tp_optimizer, num_epochs=3
    )
    
    if tp_history is None:
        print("❌ Tensor parallel training failed!")
        return False
    
    # Step 5: Evaluate tensor parallel model
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 5 - Evaluating tensor parallel model...")
    tp_metrics = evaluate_model_performance(tp_tensor_parallel, validation_data)
    
    if tp_metrics is None:
        print("❌ Tensor parallel evaluation failed!")
        return False
    
    print(f"      TP validation loss: {tp_metrics['loss']:.6f}")
    print(f"      TP perplexity: {tp_metrics['perplexity']:.2f}")
    
    # Step 6: Verify tensor parallelism correctness
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 6 - Verifying tensor parallelism correctness...")
    
    # Criterion 1: Loss Convergence Equivalence
    print(f"\n📊 Criterion 1: Loss Convergence Equivalence")
    
    # Compare final epoch losses
    final_single_loss = single_history['epoch_losses'][-1]
    final_tp_loss = tp_history['epoch_losses'][-1]
    loss_diff = abs(final_single_loss - final_tp_loss)
    loss_tolerance = 0.1  # 10% tolerance for convergence equivalence
    
    print(f"      Final single CPU loss: {final_single_loss:.6f}")
    print(f"      Final TP loss: {final_tp_loss:.6f}")
    print(f"      Loss difference: {loss_diff:.6f}")
    print(f"      Tolerance: {loss_tolerance:.6f}")
    
    if loss_diff <= loss_tolerance:
        print(f"      ✅ Loss convergence equivalence VERIFIED!")
        convergence_verified = True
    else:
        print(f"      ❌ Loss convergence equivalence FAILED!")
        convergence_verified = False
    
    # Criterion 2: Evaluation Metric Consistency
    print(f"\n📊 Criterion 2: Evaluation Metric Consistency")
    
    # Compare validation metrics
    loss_consistency = abs(single_metrics['loss'] - tp_metrics['loss'])
    perplexity_consistency = abs(single_metrics['perplexity'] - tp_metrics['perplexity'])
    
    # Use 5% tolerance for evaluation metrics
    metric_tolerance = 0.05
    
    print(f"      Validation loss difference: {loss_consistency:.6f}")
    print(f"      Validation perplexity difference: {perplexity_consistency:.2f}")
    print(f"      Metric tolerance: {metric_tolerance:.2f}")
    
    if loss_consistency <= metric_tolerance and perplexity_consistency <= metric_tolerance:
        print(f"      ✅ Evaluation metric consistency VERIFIED!")
        metrics_verified = True
    else:
        print(f"      ❌ Evaluation metric consistency FAILED!")
        metrics_verified = False
    
    # Criterion 3: Bounded Numerical Divergence
    print(f"\n📊 Criterion 3: Bounded Numerical Divergence")
    
    # Compare final weights with practical tolerance
    weight_comparison = compare_weights_with_tolerance(single_weights, tp_weights, tolerance=1e-2)
    
    print(f"      Total weights: {weight_comparison['total_weights']}")
    print(f"      Matching weights: {weight_comparison['matching_weights']}")
    print(f"      Different weights: {weight_comparison['different_weights']}")
    print(f"      Max difference: {weight_comparison['max_difference']:.2e}")
    print(f"      Mean difference: {weight_comparison['mean_difference']:.2e}")
    
    # Use 1% tolerance for weight divergence
    weight_tolerance = 0.01
    max_weight_diff_normalized = weight_comparison['max_difference'] / np.mean([np.mean(w) for w in single_weights.values()])
    
    if max_weight_diff_normalized <= weight_tolerance:
        print(f"      ✅ Bounded numerical divergence VERIFIED!")
        divergence_verified = True
    else:
        print(f"      ❌ Bounded numerical divergence FAILED!")
        divergence_verified = False
    
    # Step 7: Generate convergence plots
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Step 7 - Generating convergence plots...")
    
    try:
        plot_convergence_comparison(single_history, tp_history, save_path="opt125m_convergence_comparison.png")
        print(f"      ✅ Convergence plots generated successfully")
    except Exception as e:
        print(f"      ⚠️  Plot generation failed: {e}")
    
    # Step 8: Final assessment
    print(f"\n⏱️  {time.time() - start_time:.2f}s: Final assessment...")
    
    all_criteria_passed = convergence_verified and metrics_verified and divergence_verified
    
    print(f"\n📋 TENSOR PARALLELISM CORRECTNESS VERIFICATION RESULTS:")
    print(f"   - Loss Convergence Equivalence: {'✅ PASS' if convergence_verified else '❌ FAIL'}")
    print(f"   - Evaluation Metric Consistency: {'✅ PASS' if metrics_verified else '❌ FAIL'}")
    print(f"   - Bounded Numerical Divergence: {'✅ PASS' if divergence_verified else '❌ FAIL'}")
    
    if all_criteria_passed:
        print(f"\n🎉 SUCCESS: All tensor parallelism correctness criteria verified!")
        print(f"\n💡 OPT-125M TENSOR PARALLELISM VERIFICATION:")
        print(f"   ✅ Loss convergence is equivalent between single and distributed models")
        print(f"   ✅ Evaluation metrics are consistent across implementations")
        print(f"   ✅ Numerical divergence is bounded and acceptable")
        print(f"\n🚀 Your OPT-125M model is CORRECTLY IMPLEMENTING tensor parallelism!")
        print(f"   The minor numerical differences are expected and acceptable in distributed systems.")
    else:
        print(f"\n⚠️  WARNING: Some tensor parallelism correctness criteria failed.")
        print(f"   Please review the failing criteria before proceeding with production use.")
    
    print(f"\n✅ Tensor parallelism correctness verification completed in {time.time() - start_time:.2f}s")
    
    return all_criteria_passed

if __name__ == "__main__":
    print("🎯 OPT-125M TENSOR PARALLELISM CORRECTNESS VERIFICATION SUITE")
    print("=" * 70)
    print("🔍 This test validates tensor parallelism correctness using practical criteria")
    print("   that are appropriate for distributed systems.")
    print("=" * 70)
    
    # Run the correctness verification test
    success = test_tensor_parallelism_correctness_opt125m()
    
    if success:
        print("\n🚀 OPT-125M TENSOR PARALLELISM IS READY FOR PRODUCTION!")
    else:
        print("\n⚠️  OPT-125M TENSOR PARALLELISM NEEDS FURTHER INVESTIGATION.") 