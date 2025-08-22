#!/usr/bin/env python3
"""
Test the _apply_real_sharding method to verify it's working correctly.
"""

import torch
import numpy as np

def test_apply_real_sharding():
    """Test the _apply_real_sharding method."""
    print("🧪 Testing _apply_real_sharding method")
    
    from src.tensor_parallel_keras.parameter_sharding import ParameterShardingStrategy
    from src.tensor_parallel_keras.state_actions_keras import SplitKeras
    
    # Create a test tensor
    test_tensor = torch.randn(64, 32)  # Shape: [64, 32]
    print(f"✅ Test tensor shape: {test_tensor.shape}")
    
    # Create SplitKeras action
    split_action = SplitKeras(world_size=2, dim=1, sharding_type="column")
    print(f"✅ SplitKeras action created: world_size=2, dim=1")
    
    # Create ParameterShardingStrategy for rank 0
    strategy_rank0 = ParameterShardingStrategy(world_size=2, rank=0)
    print(f"✅ Strategy rank 0 created")
    
    # Create ParameterShardingStrategy for rank 1
    strategy_rank1 = ParameterShardingStrategy(world_size=2, rank=1)
    print(f"✅ Strategy rank 1 created")
    
    # Test _apply_real_sharding for rank 0
    sharded_param_rank0 = strategy_rank0._apply_real_sharding(test_tensor, split_action, "test_param")
    print(f"✅ Rank 0 sharded param shape: {sharded_param_rank0.shape}")
    
    # Test _apply_real_sharding for rank 1
    sharded_param_rank1 = strategy_rank1._apply_real_sharding(test_tensor, split_action, "test_param")
    print(f"✅ Rank 1 sharded param shape: {sharded_param_rank1.shape}")
    
    # Check if they're different
    if sharded_param_rank0.shape != sharded_param_rank1.shape:
        print("✅ SUCCESS: Different shapes for different ranks")
        print(f"   Rank 0: {sharded_param_rank0.shape}")
        print(f"   Rank 1: {sharded_param_rank1.shape}")
        
        # Check parameter counts
        rank0_params = np.prod(sharded_param_rank0.shape)
        rank1_params = np.prod(sharded_param_rank1.shape)
        print(f"   Rank 0 parameters: {rank0_params}")
        print(f"   Rank 1 parameters: {rank1_params}")
        
        if rank0_params != rank1_params:
            print("✅ SUCCESS: Different parameter counts for different ranks")
        else:
            print("⚠️  WARNING: Same parameter counts despite different shapes")
    else:
        print("❌ FAILURE: Same shapes for different ranks")
    
    # Test with a different tensor
    test_tensor2 = torch.randn(32, 10)  # Shape: [32, 10]
    print(f"\n✅ Test tensor 2 shape: {test_tensor2.shape}")
    
    sharded_param2_rank0 = strategy_rank0._apply_real_sharding(test_tensor2, split_action, "test_param2")
    sharded_param2_rank1 = strategy_rank1._apply_real_sharding(test_tensor2, split_action, "test_param2")
    
    print(f"   Rank 0 tensor 2 shape: {sharded_param2_rank0.shape}")
    print(f"   Rank 1 tensor 2 shape: {sharded_param2_rank1.shape}")
    
    if sharded_param2_rank0.shape != sharded_param2_rank1.shape:
        print("✅ SUCCESS: Different shapes for tensor 2")
    else:
        print("❌ FAILURE: Same shapes for tensor 2")

if __name__ == "__main__":
    test_apply_real_sharding() 