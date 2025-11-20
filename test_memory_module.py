#!/usr/bin/env python3
"""
Test script to verify the RMT memory module implementation.

This script performs basic sanity checks on the memory-enhanced SmolVLA model.
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def test_memory_initialization():
    """Test 1: Memory tokens initialization"""
    print("\n" + "="*80)
    print("TEST 1: Memory Tokens Initialization")
    print("="*80)
    
    # Test with memory disabled
    config = SmolVLAConfig()
    config.num_mem_tokens = 0
    policy = SmolVLAPolicy(config)
    
    assert policy.model.mem_tokens is None, "Memory tokens should be None when disabled"
    print("✅ Memory disabled: PASSED")
    
    # Test with memory enabled
    config.num_mem_tokens = 4
    policy = SmolVLAPolicy(config)
    
    assert policy.model.mem_tokens is not None, "Memory tokens should be initialized"
    assert policy.model.mem_tokens.shape[0] == 4, f"Expected 4 memory tokens, got {policy.model.mem_tokens.shape[0]}"
    assert policy.model.mem_tokens.requires_grad, "Memory tokens should be trainable"
    print(f"✅ Memory enabled: PASSED (shape={policy.model.mem_tokens.shape})")
    
    # Check parameter count
    mem_params = policy.model.mem_tokens.numel()
    total_params = sum(p.numel() for p in policy.parameters())
    mem_ratio = mem_params / total_params * 100
    print(f"   Memory parameters: {mem_params:,} ({mem_ratio:.4f}% of total)")


def test_forward_pass():
    """Test 2: Forward pass with memory"""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass with Memory")
    print("="*80)
    
    config = SmolVLAConfig()
    config.num_mem_tokens = 4
    config.device = "cpu"  # Use CPU for testing
    
    policy = SmolVLAPolicy(config)
    policy.eval()
    
    # Create dummy batch
    batch_size = 2
    batch = {
        "observation.images.top": torch.randn(batch_size, 3, 480, 640),
        "observation.state": torch.randn(batch_size, 14),
        "observation.language_tokens": torch.randint(0, 1000, (batch_size, 20)),
        "observation.language_attention_mask": torch.ones(batch_size, 20, dtype=torch.bool),
        "action": torch.randn(batch_size, 50, 14),
    }
    
    try:
        with torch.no_grad():
            loss, loss_dict = policy.forward(batch)
        
        assert loss is not None, "Loss should not be None"
        assert loss.item() >= 0, "Loss should be non-negative"
        print(f"✅ Forward pass: PASSED (loss={loss.item():.4f})")
        
    except Exception as e:
        print(f"❌ Forward pass: FAILED")
        print(f"   Error: {e}")
        raise


def test_inference():
    """Test 3: Inference with memory state persistence"""
    print("\n" + "="*80)
    print("TEST 3: Inference with Memory State Persistence")
    print("="*80)
    
    config = SmolVLAConfig()
    config.num_mem_tokens = 4
    config.device = "cpu"
    
    policy = SmolVLAPolicy(config)
    policy.eval()
    
    # Create dummy batch
    batch = {
        "observation.images.top": torch.randn(1, 3, 480, 640),
        "observation.state": torch.randn(1, 14),
        "observation.language_tokens": torch.randint(0, 1000, (1, 20)),
        "observation.language_attention_mask": torch.ones(1, 20, dtype=torch.bool),
    }
    
    try:
        # Reset and check initial state
        policy.reset()
        assert policy._mem_tokens_state is None, "Memory state should be None after reset"
        print("✅ Reset: PASSED")
        
        # First inference
        with torch.no_grad():
            action1 = policy.select_action(batch)
        
        assert action1 is not None, "Action should not be None"
        assert action1.shape[-1] == 14, f"Expected action dim 14, got {action1.shape[-1]}"
        
        # Check memory state is updated
        if config.num_mem_tokens > 0:
            assert policy._mem_tokens_state is not None, "Memory state should be updated after inference"
            mem_state_1 = policy._mem_tokens_state.clone()
            print(f"✅ First inference: PASSED (memory state shape={mem_state_1.shape})")
        
        # Second inference (memory should persist)
        with torch.no_grad():
            action2 = policy.select_action(batch)
        
        if config.num_mem_tokens > 0:
            mem_state_2 = policy._mem_tokens_state
            assert mem_state_2 is not None, "Memory state should persist"
            # Memory should be different (updated)
            assert not torch.allclose(mem_state_1, mem_state_2), "Memory state should be updated"
            print(f"✅ Second inference: PASSED (memory state updated)")
        
        # Reset and check memory is cleared
        policy.reset()
        assert policy._mem_tokens_state is None, "Memory state should be cleared after reset"
        print("✅ Memory persistence: PASSED")
        
    except Exception as e:
        print(f"❌ Inference: FAILED")
        print(f"   Error: {e}")
        raise


def test_parameter_count():
    """Test 4: Verify parameter count increase"""
    print("\n" + "="*80)
    print("TEST 4: Parameter Count Comparison")
    print("="*80)
    
    # Baseline (no memory)
    config_baseline = SmolVLAConfig()
    config_baseline.num_mem_tokens = 0
    config_baseline.device = "cpu"
    policy_baseline = SmolVLAPolicy(config_baseline)
    params_baseline = sum(p.numel() for p in policy_baseline.parameters())
    
    # With memory
    config_memory = SmolVLAConfig()
    config_memory.num_mem_tokens = 4
    config_memory.device = "cpu"
    policy_memory = SmolVLAPolicy(config_memory)
    params_memory = sum(p.numel() for p in policy_memory.parameters())
    
    # Calculate difference
    param_diff = params_memory - params_baseline
    param_ratio = param_diff / params_baseline * 100
    
    # Expected: 4 tokens × 960 hidden_size = 3,840 parameters
    expected_diff = 4 * 960
    
    print(f"   Baseline parameters:  {params_baseline:,}")
    print(f"   Memory parameters:    {params_memory:,}")
    print(f"   Difference:           {param_diff:,} ({param_ratio:.4f}%)")
    print(f"   Expected difference:  {expected_diff:,}")
    
    assert abs(param_diff - expected_diff) < 100, f"Parameter difference mismatch: {param_diff} vs {expected_diff}"
    assert param_ratio < 0.01, f"Parameter increase should be < 0.01%, got {param_ratio:.4f}%"
    print("✅ Parameter count: PASSED")


def test_backward_compatibility():
    """Test 5: Backward compatibility (num_mem_tokens=0)"""
    print("\n" + "="*80)
    print("TEST 5: Backward Compatibility")
    print("="*80)
    
    config = SmolVLAConfig()
    config.num_mem_tokens = 0  # Disabled
    config.device = "cpu"
    
    policy = SmolVLAPolicy(config)
    policy.eval()
    
    batch = {
        "observation.images.top": torch.randn(1, 3, 480, 640),
        "observation.state": torch.randn(1, 14),
        "observation.language_tokens": torch.randint(0, 1000, (1, 20)),
        "observation.language_attention_mask": torch.ones(1, 20, dtype=torch.bool),
        "action": torch.randn(1, 50, 14),
    }
    
    try:
        # Should work exactly like original SmolVLA
        with torch.no_grad():
            loss, _ = policy.forward(batch)
            action = policy.select_action(batch)
        
        assert policy.model.mem_tokens is None
        assert policy._mem_tokens_state is None
        print("✅ Backward compatibility: PASSED")
        
    except Exception as e:
        print(f"❌ Backward compatibility: FAILED")
        print(f"   Error: {e}")
        raise


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SMOLVLA MEMORY MODULE TEST SUITE")
    print("="*80)
    
    try:
        test_memory_initialization()
        test_forward_pass()
        test_inference()
        test_parameter_count()
        test_backward_compatibility()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe memory module is working correctly. You can now:")
        print("1. Train a baseline model: bash train_baseline_smolvla.sh")
        print("2. Train with memory: bash train_smolvla_with_memory.sh")
        print("3. Compare models: python evaluate_models.py")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TESTS FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease check the implementation and try again.")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    main()
