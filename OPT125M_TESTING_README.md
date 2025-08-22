# OPT-125M Tensor Parallel Testing Guide

This guide explains how to test the OPT-125M model with tensor parallelism to ensure it's ready for production training.

## 🎯 Overview

The OPT-125M model is a 125 million parameter language model that requires careful testing to ensure tensor parallelism works correctly. This test suite verifies:

1. **Forward Pass Consistency** - Mathematical identity between single and multi-device computation
2. **Backward Pass Consistency** - Gradient computation correctness
3. **Training Stability** - Model convergence and stability
4. **Memory Efficiency** - Proper parameter sharding
5. **Optimizer Integration** - Training loop functionality

## 🧪 Test Files

### 1. `test_forward_pass_consistency_opt125m.py` - The "Golden Test"
**Purpose**: Verify mathematical identity between single CPU and 2-CPU sharded computation

**What it tests**:
- Creates OPT-125M model on single CPU
- Runs forward pass with fixed input to get "golden" result
- Creates TensorParallelKeras with 2-CPU mesh and JAX backend
- Runs forward pass with same input on sharded model
- Verifies outputs are numerically identical (within 1e-5 tolerance)

**Why it's critical**: This is the foundation test that proves your tensor parallelism implementation is mathematically correct.

### 2. `test_opt125m_comprehensive.py` - Training Readiness Tests
**Purpose**: Comprehensive testing of all components needed for production training

**Tests included**:
- **Backward Pass Consistency**: Gradient computation verification
- **Gradient Computation**: Proper gradient shapes and synchronization
- **Optimizer Integration**: Adam, SGD, RMSprop compatibility
- **Memory Efficiency**: Parameter sharding verification
- **Training Stability**: Multi-epoch training convergence

### 3. `test_opt125m_verification.py` - Legacy Verification Tests
**Purpose**: Additional verification from existing test suite

**Tests included**:
- Parameter sharding verification
- Inference correctness
- Training verification

## 🚀 How to Run Tests

### Option 1: Run All Tests (Recommended)
```bash
python3 run_opt125m_tests.py
```

This will:
- Execute all available tests in sequence
- Provide comprehensive reporting
- Save detailed results to `opt125m_test_report.txt`
- Exit with appropriate status code (0 for success, 1 for failure)

### Option 2: Run Individual Tests
```bash
# Run the critical "Golden Test"
python3 test_forward_pass_consistency_opt125m.py

# Run comprehensive training tests
python3 test_opt125m_comprehensive.py

# Run legacy verification tests
python3 test_opt125m_verification.py
```

### Option 3: Run with JAX Device Simulation
```bash
# Set JAX environment for 2-CPU simulation
export XLA_FLAGS='--xla_force_host_platform_device_count=2'

# Run tests
python3 run_opt125m_tests.py
```

## 🔍 What Each Test Verifies

### Forward Pass Consistency (Golden Test)
- ✅ Model creation and compilation
- ✅ Single CPU forward pass execution
- ✅ TensorParallelKeras instantiation with JAX backend
- ✅ 2-CPU sharded forward pass execution
- ✅ Numerical consistency verification (1e-5 tolerance)
- ✅ Output shape and dtype matching
- ✅ NaN/Inf value detection

### Backward Pass Consistency
- ✅ Training step execution on both models
- ✅ Loss computation consistency
- ✅ Gradient computation and application
- ✅ Parameter update verification

### Gradient Computation
- ✅ GradientTape integration
- ✅ Gradient shape verification
- ✅ Parameter synchronization across shards
- ✅ Numerical gradient consistency

### Optimizer Integration
- ✅ Adam optimizer compatibility
- ✅ SGD optimizer compatibility
- ✅ RMSprop optimizer compatibility
- ✅ Training loop functionality

### Memory Efficiency
- ✅ Parameter count verification
- ✅ Sharding distribution analysis
- ✅ Memory overhead calculation
- ✅ Shard parameter balance

### Training Stability
- ✅ Multi-epoch training execution
- ✅ Loss convergence verification
- ✅ Training stability assessment
- ✅ Model state consistency

## 📊 Expected Results

### Success Criteria
- **All tests must pass** for production readiness
- **Forward pass consistency**: Outputs identical within 1e-5 tolerance
- **Backward pass consistency**: Loss differences < 1e-3
- **Memory efficiency**: Overhead < 50%
- **Training stability**: Losses decreasing over epochs

### Failure Indicators
- ❌ Numerical inconsistencies > tolerance thresholds
- ❌ Shape mismatches between models
- ❌ Training failures or exceptions
- ❌ Memory overhead > 50%
- ❌ Training instability or divergence

## 🛠️ Troubleshooting

### Common Issues

#### 1. JAX Device Detection Failures
```bash
# Ensure JAX environment is set correctly
export XLA_FLAGS='--xla_force_host_platform_device_count=2'
python3 -c "import jax; print(jax.devices())"
```

#### 2. Import Errors
```bash
# Check Python path includes src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 3. Memory Issues
- Reduce model size for testing (use `num_layers=2`, `hidden_size=128`)
- Check available system memory
- Verify JAX memory allocation

#### 4. Numerical Inconsistencies
- Check tensor parallelism implementation
- Verify communication primitives
- Ensure proper gradient synchronization

### Debug Mode
For detailed debugging, modify test files to include:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Production Readiness Checklist

Before proceeding with OPT-125M training, ensure:

- [ ] **Forward Pass Consistency Test** passes with 1e-5 tolerance
- [ ] **Backward Pass Consistency Test** passes with 1e-3 tolerance
- [ ] **Gradient Computation Test** passes
- [ ] **Optimizer Integration Test** passes for all optimizers
- [ ] **Memory Efficiency Test** shows < 50% overhead
- [ ] **Training Stability Test** shows convergence
- [ ] All tests complete without exceptions
- [ ] Detailed test report generated successfully

## 🚀 Next Steps After Testing

Once all tests pass:

1. **Scale up to full OPT-125M**:
   ```python
   model = create_opt125m_model(
       vocab_size=50257,
       hidden_size=768,
       num_layers=12,
       num_heads=12
   )
   ```

2. **Configure production environment**:
   - Set up proper distributed training infrastructure
   - Configure JAX devices (GPUs/TPUs)
   - Set memory and performance optimizations

3. **Prepare training data**:
   - Tokenize your dataset
   - Set up data loading pipeline
   - Configure training parameters

4. **Start training**:
   ```python
   tp_model = TensorParallelKeras(
       model=model,
       world_size=4,  # Use actual device count
       distributed_backend='jax'
   )
   
   tp_model.fit(
       train_dataset,
       epochs=100,
       batch_size=32,
       validation_data=val_dataset
   )
   ```

## 📚 Additional Resources

- [Tensor Parallel Keras Implementation](../src/tensor_parallel_keras/)
- [JAX Backend Integration](../src/tensor_parallel_keras/distributed_backend.py)
- [Communication Primitives](../src/tensor_parallel_keras/communications_keras.py)
- [Parameter Sharding](../src/tensor_parallel_keras/parameter_sharding.py)

## 🆘 Support

If you encounter issues:

1. Check the detailed test report (`opt125m_test_report.txt`)
2. Review error messages and stack traces
3. Verify environment setup and dependencies
4. Check tensor parallelism implementation
5. Ensure JAX backend is properly configured

---

**Remember**: These tests are your "safety net" for OPT-125M training. Only proceed with production training after all tests pass successfully! 🎯 