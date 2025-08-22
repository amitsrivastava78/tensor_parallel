#!/usr/bin/env python3
"""
Test SplitKeras to verify it's working correctly with different ranks.
"""

import torch
import numpy as np

def test_split_keras():
    """Test SplitKeras with different ranks."""
    print("🧪 Testing SplitKeras with different ranks")
    
    from src.tensor_parallel_keras.state_actions_keras import SplitKeras
    
    # Create a test tensor
    test_tensor = torch.randn(64, 32)  # Shape: [64, 32]
    print(f"✅ Test tensor shape: {test_tensor.shape}")
    
    # Create SplitKeras instance for column-wise sharding
    split_op = SplitKeras(world_size=2, dim=1, sharding_type="column")
    print(f"✅ SplitKeras created: world_size=2, dim=1")
    
    # Test with rank 0
    rank0_tensor = split_op(test_tensor, rank=0)
    print(f"✅ Rank 0 tensor shape: {rank0_tensor.shape}")
    
    # Test with rank 1
    rank1_tensor = split_op(test_tensor, rank=1)
    print(f"✅ Rank 1 tensor shape: {rank1_tensor.shape}")
    
    # Check if they're different
    if rank0_tensor.shape != rank1_tensor.shape:
        print("✅ SUCCESS: Different shapes for different ranks")
        print(f"   Rank 0: {rank0_tensor.shape}")
        print(f"   Rank 1: {rank1_tensor.shape}")
        
        # Check parameter counts
        rank0_params = np.prod(rank0_tensor.shape)
        rank1_params = np.prod(rank1_tensor.shape)
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
    
    rank0_tensor2 = split_op(test_tensor2, rank=0)
    rank1_tensor2 = split_op(test_tensor2, rank=1)
    
    print(f"   Rank 0 tensor 2 shape: {rank0_tensor2.shape}")
    print(f"   Rank 1 tensor 2 shape: {rank1_tensor2.shape}")
    
    if rank0_tensor2.shape != rank1_tensor2.shape:
        print("✅ SUCCESS: Different shapes for tensor 2")
    else:
        print("❌ FAILURE: Same shapes for tensor 2")

if __name__ == "__main__":
    test_split_keras() 