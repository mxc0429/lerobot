"""
Visualize the three camera views from the block stacking environment.
Saves images from all three cameras to disk.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("Visualize camera views from block stacking environment")
    parser.add_argument("--output_dir", type=str, default="./camera_views", help="Directory to save images")
    parser.add_argument("--randomize", action="store_true", help="Randomize block positions")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating block stacking environment...")
    
    # Create environment with multiple cameras
    env = So101MujocoEnv(
        So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode=None,  # No window rendering
            use_camera=True,
            camera_width=640,  # Higher resolution for better visualization
            camera_height=480,
            camera_names=["top_cam", "wrist_cam", "right_cam"],
            randomize_blocks=args.randomize,
            num_blocks=3,
        )
    )
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"\nEnvironment created successfully!")
    print(f"Number of joints: {env.nq}")
    print(f"Number of actuators: {env.nu}")
    
    # Get block positions
    block_positions = env.get_block_positions()
    print("\nBlock positions:")
    for name, pos in block_positions.items():
        print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    
    # Save images from all cameras
    print(f"\nSaving camera images to {output_dir}/")
    
    for cam_name in ["top_cam", "wrist_cam", "right_cam"]:
        img_key = f"image_{cam_name}"
        if img_key in obs:
            img_array = obs[img_key]
            img = Image.fromarray(img_array)
            
            output_path = output_dir / f"{cam_name}.png"
            img.save(output_path)
            print(f"  ✓ Saved {cam_name}: {output_path}")
    
    # Create a combined view
    print("\nCreating combined view...")
    top_img = obs["image_top_cam"]
    wrist_img = obs["image_wrist_cam"]
    right_img = obs["image_right_cam"]
    
    # Arrange images in a grid (top row: top_cam, bottom row: wrist_cam and right_cam)
    h, w = top_img.shape[:2]
    combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Top left: top camera
    combined[0:h, 0:w] = top_img
    # Bottom left: wrist camera
    combined[h:h*2, 0:w] = wrist_img
    # Bottom right: right camera
    combined[h:h*2, w:w*2] = right_img
    
    # Add labels (simple text overlay would require PIL, so we'll just save)
    combined_img = Image.fromarray(combined)
    combined_path = output_dir / "combined_view.png"
    combined_img.save(combined_path)
    print(f"  ✓ Saved combined view: {combined_path}")
    
    env.close()
    print("\n✅ Done! Check the images in the output directory.")


if __name__ == "__main__":
    main()
