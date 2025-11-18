"""
Teleoperate SO101 Leader to control MuJoCo SO101 sim for block stacking task.
Records dataset with multiple camera views for training SmolVLA.
"""
import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("Teleoperate SO101 for block stacking task and record dataset")
    parser.add_argument("--repo_id", type=str, required=True, default="test/block_stacking_test",help="HF dataset repo id, e.g. user/so101_block_stacking")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to record")
    parser.add_argument("--episode_time_s", type=int, default=60, help="Max time per episode")
    parser.add_argument("--reset_time_s", type=int, default=10, help="Time between episodes for reset")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader_id", type=str, default=None)
    parser.add_argument("--leader_calibration_dir", type=str, default=None)
    parser.add_argument("--camera_width", type=int, default=256, help="Camera image width")
    parser.add_argument("--camera_height", type=int, default=256, help="Camera image height")
    parser.add_argument("--single_camera", action="store_true", help="Use only top camera (faster)")
    parser.add_argument("--randomize_blocks", action="store_true", help="Randomize block positions each episode")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no_degrees", dest="use_degrees", action="store_false")
    parser.add_argument("--skip_calibration", action="store_true")
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
    
    # Add features for each camera
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
        image_writer_threads=8,  # More threads for multiple cameras
    )
    return ds


def main():
    args = parse_args()
    init_logging()
    
    # Note: Rerun visualization can cause crashes, use MuJoCo window instead
    # if args.display:
    #     init_rerun(session_name="so101_block_stacking")

    # Environment with multiple cameras
    camera_names = ["top_cam"] if args.single_camera else ["top_cam", "wrist_cam", "right_cam"]
    env = So101MujocoEnv(
        So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode="window" if args.display else None,
            use_camera=True,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_names=camera_names,
            randomize_blocks=args.randomize_blocks,
            num_blocks=3,
        )
    )
    obs, _ = env.reset()

    # Leader arm
    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.port,
            use_degrees=args.use_degrees,
            id=args.leader_id,
            calibration_dir=Path(args.leader_calibration_dir).expanduser() if args.leader_calibration_dir else None,
        )
    )
    leader.connect(calibrate=not args.skip_calibration)
    if not leader.is_calibrated:
        raise RuntimeError(
            "Leader arm has no calibration. Run without --skip_calibration at least once."
        )

    # Controls
    listener, events = init_keyboard_listener()
    dataset = make_dataset(env, args.repo_id, args.fps, camera_names)

    recorded = 0
    successful_stacks = 0
    
    print("=" * 70)
    print("BLOCK STACKING TELEOPERATION")
    print("=" * 70)
    print(f"Task: Stack blocks using the robot arm")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Randomize blocks: {args.randomize_blocks}")
    print("=" * 70)
    
    try:
        while recorded < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded+1} / {args.num_episodes}")
            
            # Show block positions at start
            block_positions = env.get_block_positions()
            print("\nInitial block positions:")
            for name, pos in block_positions.items():
                print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            
            episode_start = time.perf_counter()
            steps = 0
            dataset.clear_episode_buffer()
            
            while (time.perf_counter() - episode_start) < args.episode_time_s and not events["exit_early"]:
                step_start = time.perf_counter()
                
                # Read leader
                leader_obs = leader.get_action()
                leader_obs = {k: float(v) for k, v in leader_obs.items()}
                action = map_leader_to_action(leader_obs, env, args.use_degrees)
                obs, reward, terminated, truncated, info = env.step(action)

                # Write transition
                frame = {
                    "task": "block_stacking",
                    "action": action.astype(np.float32),
                    "observation.qpos": obs["qpos"].astype(np.float32),
                    "reward": np.array([reward], dtype=np.float32),
                    "done": np.array([terminated or truncated], dtype=bool),
                }
                
                # Add all camera images
                for cam_name in camera_names:
                    frame[f"observation.image_{cam_name}"] = obs[f"image_{cam_name}"]
                
                dataset.add_frame(frame)

                # Rate control - account for actual execution time
                steps += 1
                elapsed = time.perf_counter() - step_start
                sleep_time = max(0.0, (1.0 / args.fps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Check if stacking was successful
            is_stacked = env.check_stacking_success()
            if is_stacked:
                successful_stacks += 1
                print("✅ Blocks successfully stacked!")
            else:
                print("❌ Stacking incomplete")
            
            # Show final block positions
            block_positions = env.get_block_positions()
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
            obs, _ = env.reset()

    finally:
        print("\nCleaning up...")
        try:
            leader.disconnect()
        except Exception as e:
            print(f"Warning: Error disconnecting leader: {e}")
        
        try:
            env.close()
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")
        
        try:
            dataset.finalize()
        except Exception as e:
            print(f"Warning: Error finalizing dataset: {e}")
        
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
                print(f"❌ Error uploading dataset: {e}")


if __name__ == "__main__":
    main()
