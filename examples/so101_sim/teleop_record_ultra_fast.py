"""
Ultra-fast teleoperation - cameras only render when recording data.
This provides the lowest latency control by decoupling camera rendering from control loop.
"""
import argparse
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("Ultra-fast teleoperation")
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--record_fps", type=int, default=10, help="Data recording FPS (cameras render at this rate)")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_time_s", type=int, default=60)
    parser.add_argument("--reset_time_s", type=int, default=10)
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader_id", type=str, default="my_awesome_leader_arm")
    parser.add_argument("--leader_calibration_dir", type=str, default="~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader")
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--randomize_blocks", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--no_degrees", dest="use_degrees", action="store_false")
    parser.set_defaults(use_degrees=True)
    return parser.parse_args()


def map_leader_to_action(
    leader_obs: Dict[str, float],
    env: So101MujocoEnv,
    use_degrees: bool,
) -> NDArray[np.float32]:
    """Convert leader joint readings into MuJoCo normalized action."""
    joint_targets: Dict[str, float] = {}
    for name in env.actuator_names:
        key = f"{name}.pos"
        if key not in leader_obs:
            continue
        value = float(leader_obs[key])
        if use_degrees:
            value = np.deg2rad(value)
        joint_targets[name] = value
    return env.actuator_target_from_joint_dict(joint_targets)


def make_dataset(env: So101MujocoEnv, repo_id: str, fps: int, camera_names: list) -> LeRobotDataset:
    """Create dataset with multiple camera views."""
    features = {
        "action": {"dtype": "float32", "shape": (env.nu,), "names": None},
        "observation.qpos": {"dtype": "float32", "shape": (env.nq,), "names": None},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    
    for cam_name in camera_names:
        features[f"observation.image_{cam_name}"] = {
            "dtype": "video",
            "shape": (env.cfg.camera_height, env.cfg.camera_width, 3),
            "names": ["height", "width", "channels"],
        }
    
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=8,
    )
    return ds


def create_combined_view(images: Dict[str, np.ndarray]) -> np.ndarray:
    """Create a combined view of all cameras."""
    top = images.get("image_top_cam")
    wrist = images.get("image_wrist_cam")
    right = images.get("image_right_cam")
    
    if top is None:
        return None
    
    h, w = top.shape[:2]
    combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    combined[0:h, 0:w] = top
    if wrist is not None:
        combined[h:h*2, 0:w] = wrist
    if right is not None:
        combined[0:h, w:w*2] = right
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Top", (10, 30), font, 0.7, (0, 255, 0), 2)
    if wrist is not None:
        cv2.putText(combined, "Wrist", (10, h + 30), font, 0.7, (0, 255, 0), 2)
    if right is not None:
        cv2.putText(combined, "Right", (w + 10, 30), font, 0.7, (0, 255, 0), 2)
    
    return combined


def main():
    args = parse_args()
    init_logging()

    # Two environments: one for fast control (no camera), one for recording (with camera)
    camera_names = ["top_cam", "wrist_cam", "right_cam"]
    
    # Fast control environment (no cameras)
    control_env = So101MujocoEnv(
        So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode="window",  # MuJoCo window
            use_camera=False,  # No cameras = fast!
            randomize_blocks=args.randomize_blocks,
            num_blocks=3,
        )
    )
    
    # Recording environment (with cameras, only used periodically)
    record_env = So101MujocoEnv(
        So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode=None,  # No window for this one
            use_camera=True,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_names=camera_names,
            randomize_blocks=False,  # Will sync from control_env
            num_blocks=3,
        )
    )
    
    control_env.reset()
    record_env.reset()

    # Leader arm
    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.port,
            use_degrees=args.use_degrees,
            id=args.leader_id,
            calibration_dir=Path(args.leader_calibration_dir).expanduser(),
        )
    )
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        raise RuntimeError("Leader arm has no calibration.")

    # Controls
    listener, events = init_keyboard_listener()
    dataset = make_dataset(record_env, args.repo_id, args.record_fps, camera_names)

    # Create OpenCV window
    cv2.namedWindow("Camera Views", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Views", 1000, 1000)

    recorded = 0
    successful_stacks = 0
    
    print("=" * 70)
    print("ULTRA-FAST TELEOPERATION")
    print("=" * 70)
    print("Strategy: Separate control and recording")
    print(f"  - Control loop: ~50Hz (no cameras)")
    print(f"  - Recording: {args.record_fps}Hz (with cameras)")
    print(f"  - Camera resolution: {args.camera_width}x{args.camera_height}")
    print("=" * 70)
    print("This provides minimal latency while still recording high-quality data!")
    print("=" * 70)
    
    record_interval = 1.0 / args.record_fps
    
    try:
        while recorded < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded+1} / {args.num_episodes}")
            
            block_positions = control_env.get_block_positions()
            print("\nInitial block positions:")
            for name, pos in block_positions.items():
                print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            
            # Sync record_env with control_env
            record_env.data.qpos[:] = control_env.data.qpos[:]
            record_env.data.qvel[:] = control_env.data.qvel[:]
            
            episode_start = time.perf_counter()
            last_record_time = episode_start
            steps = 0
            dataset.clear_episode_buffer()
            
            while (time.perf_counter() - episode_start) < args.episode_time_s and not events["exit_early"]:
                step_start = time.perf_counter()
                
                # Read leader
                leader_obs = leader.get_action()
                leader_obs = {k: float(v) for k, v in leader_obs.items()}
                action = map_leader_to_action(leader_obs, control_env, args.use_degrees)
                
                # Fast control step (no cameras)
                control_env.step(action)
                
                # Record data at specified FPS
                current_time = time.perf_counter()
                if current_time - last_record_time >= record_interval:
                    # Sync record_env state
                    record_env.data.qpos[:] = control_env.data.qpos[:]
                    record_env.data.qvel[:] = control_env.data.qvel[:]
                    
                    # Step record_env to get camera images
                    obs, reward, terminated, truncated, info = record_env.step(action)
                    
                    # Write transition
                    frame = {
                        "task": "block_stacking",
                        "action": action.astype(np.float32),
                        "observation.qpos": obs["qpos"].astype(np.float32),
                        "reward": np.array([reward], dtype=np.float32),
                        "done": np.array([terminated or truncated], dtype=bool),
                    }
                    
                    for cam_name in camera_names:
                        frame[f"observation.image_{cam_name}"] = obs[f"image_{cam_name}"]
                    
                    dataset.add_frame(frame)
                    
                    # Update preview
                    combined_view = create_combined_view(obs)
                    if combined_view is not None:
                        combined_bgr = cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Camera Views", combined_bgr)
                        cv2.waitKey(1)
                    
                    last_record_time = current_time

                steps += 1
                
                # Minimal sleep to avoid maxing CPU
                time.sleep(0.001)
                
                # Print stats
                if steps % 100 == 0:
                    elapsed = time.perf_counter() - episode_start
                    hz = steps / elapsed
                    print(f"  Steps: {steps}, Control freq: {hz:.1f}Hz")

            # Check stacking
            is_stacked = control_env.check_stacking_success()
            if is_stacked:
                successful_stacks += 1
                print("✅ Blocks successfully stacked!")
            else:
                print("❌ Stacking incomplete")
            
            block_positions = control_env.get_block_positions()
            print("\nFinal block positions:")
            for name, pos in block_positions.items():
                print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            
            dataset.save_episode()
            recorded += 1
            
            print(f"\nProgress: {recorded}/{args.num_episodes} episodes, {successful_stacks} successful")

            if recorded < args.num_episodes and not events["stop_recording"]:
                log_say(f"Reset window - {args.reset_time_s}s")
                t0 = time.perf_counter()
                while (time.perf_counter() - t0) < args.reset_time_s and not events["stop_recording"]:
                    time.sleep(0.1)
            
            control_env.reset()
            record_env.reset()

    finally:
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        try:
            leader.disconnect()
        except Exception as e:
            print(f"Warning: {e}")
        
        try:
            control_env.close()
            record_env.close()
        except Exception as e:
            print(f"Warning: {e}")
        
        try:
            dataset.finalize()
        except Exception as e:
            print(f"Warning: {e}")
        
        print("\n" + "=" * 70)
        print("RECORDING COMPLETE")
        print("=" * 70)
        print(f"Total episodes: {recorded}")
        print(f"Successful stacks: {successful_stacks} ({successful_stacks/max(recorded,1)*100:.1f}%)")
        print(f"Dataset: {args.repo_id}")
        print("=" * 70)
        
        if args.push_to_hub:
            print("\nPushing dataset to Hugging Face Hub...")
            try:
                dataset.push_to_hub()
                print("✅ Dataset uploaded successfully!")
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
