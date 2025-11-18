"""
Test script for the block stacking environment with multiple cameras.
This script visualizes the environment and displays all three camera views.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("Test SO101 block stacking environment")
    parser.add_argument("--no_render", action="store_true", help="Disable window rendering")
    parser.add_argument("--randomize", action="store_true", help="Randomize block positions")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create environment with multiple cameras
    env = So101MujocoEnv(
        So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode="window" if not args.no_render else None,
            use_camera=True,
            camera_width=256,
            camera_height=256,
            camera_names=["top_cam", "wrist_cam", "right_cam"],  # Three camera views
            randomize_blocks=args.randomize,
            num_blocks=3,
        )
    )
    
    print("=" * 60)
    print("Block Stacking Environment Test")
    print("=" * 60)
    print(f"XML Path: {BLOCK_STACKING_XML}")
    print(f"Number of joints (nq): {env.nq}")
    print(f"Number of actuators (nu): {env.nu}")
    print(f"Actuator names: {env.actuator_names}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 60)
    
    # Reset environment
    obs, info = env.reset()
    
    print("\nInitial observation keys:", obs.keys())
    print("Joint positions shape:", obs["qpos"].shape)
    
    if "image_top_cam" in obs:
        print("Top camera image shape:", obs["image_top_cam"].shape)
    if "image_wrist_cam" in obs:
        print("Wrist camera image shape:", obs["image_wrist_cam"].shape)
    if "image_right_cam" in obs:
        print("Right camera image shape:", obs["image_right_cam"].shape)
    
    # Get initial block positions
    block_positions = env.get_block_positions()
    print("\nInitial block positions:")
    for name, pos in block_positions.items():
        print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    
    print(f"\nRunning {args.num_steps} simulation steps with random actions...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for step in range(args.num_steps):
            # Random action (small movements)
            action = env.action_space.sample() * 0.1  # Scale down for smoother motion
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print status every 100 steps
            if (step + 1) % 100 == 0:
                block_positions = env.get_block_positions()
                is_stacked = env.check_stacking_success()
                print(f"Step {step + 1}/{args.num_steps} - Stacked: {is_stacked}")
                
                # Print block heights
                for name, pos in block_positions.items():
                    print(f"  {name}: z={pos[2]:.3f}")
            
            # Small delay for visualization
            time.sleep(0.01)
            
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Final block positions
        block_positions = env.get_block_positions()
        print("\nFinal block positions:")
        for name, pos in block_positions.items():
            print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        
        is_stacked = env.check_stacking_success()
        print(f"\nBlocks successfully stacked: {is_stacked}")
        
        env.close()
        print("\nEnvironment closed successfully!")


if __name__ == "__main__":
    main()
