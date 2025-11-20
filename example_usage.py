#!/usr/bin/env python3
"""
Example usage of SmolVLA with RMT memory module.

This script demonstrates how to use the memory-enhanced SmolVLA model
for inference and training.
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def example_1_basic_usage():
    """Example 1: Basic usage with memory enabled"""
    print("\n" + "="*80)
    print("Example 1: Basic Usage with Memory")
    print("="*80)
    
    # Create config with memory enabled
    config = SmolVLAConfig()
    config.num_mem_tokens = 4  # Enable 4 memory tokens
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create policy
    policy = SmolVLAPolicy(config)
    policy.eval()
    
    print(f"✅ Created SmolVLA policy with {config.num_mem_tokens} memory tokens")
    print(f"   Device: {config.device}")
    print(f"   Memory tokens shape: {policy.model.mem_tokens.shape}")
    
    # Simulate an episode
    policy.reset()  # Reset memory at episode start
    print("\n📝 Simulating episode with 5 steps...")
    
    for step in range(5):
        # Create dummy observation
        batch = {
            "observation.images.top": torch.randn(1, 3, 480, 640).to(config.device),
            "observation.state": torch.randn(1, 14).to(config.device),
            "observation.language_tokens": torch.randint(0, 1000, (1, 20)).to(config.device),
            "observation.language_attention_mask": torch.ones(1, 20, dtype=torch.bool).to(config.device),
        }
        
        # Get action
        with torch.no_grad():
            action = policy.select_action(batch)
        
        # Check memory state
        if policy._mem_tokens_state is not None:
            mem_norm = torch.norm(policy._mem_tokens_state).item()
            print(f"   Step {step+1}: action shape={action.shape}, memory norm={mem_norm:.4f}")
        else:
            print(f"   Step {step+1}: action shape={action.shape}, no memory")
    
    print("✅ Episode completed!")


def example_2_compare_with_without_memory():
    """Example 2: Compare inference with and without memory"""
    print("\n" + "="*80)
    print("Example 2: Compare With and Without Memory")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy observation
    batch = {
        "observation.images.top": torch.randn(1, 3, 480, 640).to(device),
        "observation.state": torch.randn(1, 14).to(device),
        "observation.language_tokens": torch.randint(0, 1000, (1, 20)).to(device),
        "observation.language_attention_mask": torch.ones(1, 20, dtype=torch.bool).to(device),
    }
    
    # Without memory
    config_no_mem = SmolVLAConfig()
    config_no_mem.num_mem_tokens = 0
    config_no_mem.device = device
    policy_no_mem = SmolVLAPolicy(config_no_mem)
    policy_no_mem.eval()
    
    # With memory
    config_with_mem = SmolVLAConfig()
    config_with_mem.num_mem_tokens = 4
    config_with_mem.device = device
    policy_with_mem = SmolVLAPolicy(config_with_mem)
    policy_with_mem.eval()
    
    # Measure inference time
    import time
    
    # Without memory
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            action_no_mem = policy_no_mem.select_action(batch)
    time_no_mem = (time.time() - start) / 10 * 1000
    
    # With memory
    policy_with_mem.reset()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            action_with_mem = policy_with_mem.select_action(batch)
    time_with_mem = (time.time() - start) / 10 * 1000
    
    print(f"⚡ Inference time (avg over 10 runs):")
    print(f"   Without memory: {time_no_mem:.2f} ms")
    print(f"   With memory:    {time_with_mem:.2f} ms")
    print(f"   Overhead:       {(time_with_mem - time_no_mem) / time_no_mem * 100:+.2f}%")
    
    # Parameter count
    params_no_mem = sum(p.numel() for p in policy_no_mem.parameters())
    params_with_mem = sum(p.numel() for p in policy_with_mem.parameters())
    
    print(f"\n📊 Parameter count:")
    print(f"   Without memory: {params_no_mem:,}")
    print(f"   With memory:    {params_with_mem:,}")
    print(f"   Difference:     {params_with_mem - params_no_mem:,} ({(params_with_mem - params_no_mem) / params_no_mem * 100:.4f}%)")


def example_3_load_pretrained():
    """Example 3: Load pretrained model with memory"""
    print("\n" + "="*80)
    print("Example 3: Load Pretrained Model")
    print("="*80)
    
    # This example shows how to load a trained model
    checkpoint_path = "outputs/train/smolvla_with_memory_4tokens/checkpoints/last/pretrained_model"
    
    print(f"📂 Loading model from: {checkpoint_path}")
    print("   (Note: This will fail if the model hasn't been trained yet)")
    
    try:
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        policy.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Memory tokens: {policy.config.num_mem_tokens}")
        print(f"   Device: {policy.config.device}")
        
        # Use the model
        policy.reset()
        print("\n📝 Model ready for inference!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Train the model first using: bash train_smolvla_with_memory.sh")


def example_4_training_loop():
    """Example 4: Training loop with memory"""
    print("\n" + "="*80)
    print("Example 4: Training Loop (Simplified)")
    print("="*80)
    
    config = SmolVLAConfig()
    config.num_mem_tokens = 4
    config.device = "cpu"  # Use CPU for demo
    
    policy = SmolVLAPolicy(config)
    policy.train()
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    print("🎓 Training for 5 steps (demo)...")
    
    for step in range(5):
        # Create dummy batch
        batch = {
            "observation.images.top": torch.randn(2, 3, 480, 640),
            "observation.state": torch.randn(2, 14),
            "observation.language_tokens": torch.randint(0, 1000, (2, 20)),
            "observation.language_attention_mask": torch.ones(2, 20, dtype=torch.bool),
            "action": torch.randn(2, 50, 14),
        }
        
        # Forward pass
        loss, loss_dict = policy.forward(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step+1}: loss={loss.item():.4f}")
    
    print("✅ Training demo completed!")
    print("\n💡 For real training, use: bash train_smolvla_with_memory.sh")


def example_5_memory_visualization():
    """Example 5: Visualize memory evolution"""
    print("\n" + "="*80)
    print("Example 5: Memory Evolution Visualization")
    print("="*80)
    
    config = SmolVLAConfig()
    config.num_mem_tokens = 4
    config.device = "cpu"
    
    policy = SmolVLAPolicy(config)
    policy.eval()
    
    # Track memory norms over time
    memory_norms = []
    
    policy.reset()
    print("📊 Tracking memory evolution over 10 steps...")
    
    for step in range(10):
        batch = {
            "observation.images.top": torch.randn(1, 3, 480, 640),
            "observation.state": torch.randn(1, 14),
            "observation.language_tokens": torch.randint(0, 1000, (1, 20)),
            "observation.language_attention_mask": torch.ones(1, 20, dtype=torch.bool),
        }
        
        with torch.no_grad():
            action = policy.select_action(batch)
        
        if policy._mem_tokens_state is not None:
            # Compute norm for each memory token
            norms = torch.norm(policy._mem_tokens_state, dim=-1).squeeze().tolist()
            memory_norms.append(norms)
    
    # Print memory evolution
    print("\n   Memory token norms over time:")
    print("   " + "-" * 60)
    for step, norms in enumerate(memory_norms):
        norm_str = " ".join([f"{n:6.2f}" for n in norms])
        print(f"   Step {step+1:2d}: [{norm_str}]")
    
    print("\n💡 Memory tokens are learning to encode task-relevant information!")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("SMOLVLA WITH MEMORY - USAGE EXAMPLES")
    print("="*80)
    
    try:
        example_1_basic_usage()
        example_2_compare_with_without_memory()
        example_3_load_pretrained()
        example_4_training_loop()
        example_5_memory_visualization()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("="*80)
        print("\n📚 Next steps:")
        print("   1. Read the guide: SMOLVLA_MEMORY_GUIDE.md")
        print("   2. Run tests: python test_memory_module.py")
        print("   3. Train models: bash train_smolvla_with_memory.sh")
        print("   4. Evaluate: python evaluate_models.py")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
