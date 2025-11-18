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


def parse_args():
    parser = argparse.ArgumentParser("Teleoperate SO101 Leader to control MuJoCo SO101 sim and record dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="HF dataset repo id, e.g. user/so101_sim")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--episode_time_s", type=int, default=30)
    parser.add_argument("--reset_time_s", type=int, default=5)
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader_id", type=str, default=None, help="Identifier used to load calibration for leader arm")
    parser.add_argument(
        "--leader_calibration_dir",
        type=str,
        default=None,
        help="Path to directory containing calibration JSON (overrides default cache path)",
    )
    parser.add_argument("--use_camera", action="store_true", help="Record front camera images from MuJoCo")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no_degrees", dest="use_degrees", action="store_false", help="Leader values are normalized [-100, 100]")
    parser.add_argument("--skip_calibration", action="store_true", help="Skip leader auto calibration on connect")
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


def make_dataset(env: So101MujocoEnv, repo_id: str, fps: int, use_videos: bool) -> LeRobotDataset:
    features = {
        "action": {"dtype": "float32", "shape": (env.nu,), "names": None},
        "observation.qpos": {"dtype": "float32", "shape": (env.nq,), "names": None},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    if use_videos:
        features["observation.image_front"] = {
            "dtype": "video",
            "shape": (env.cfg.camera_height, env.cfg.camera_width, 3),
            "names": ["height", "width", "channels"],
        }
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=use_videos,
        image_writer_threads=4,
    )
    return ds


def main():
    args = parse_args()
    init_logging()
    if args.display:
        init_rerun(session_name="so101_sim_teleop")

    # Env
    env = So101MujocoEnv(
        So101MujocoConfig(
            render_mode="window" if args.display else None,
            use_camera=args.use_camera,
            camera_width=256,
            camera_height=256,
        )
    )
    obs, _ = env.reset()

    # Leader
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
            "Leader arm has no calibration. Run the script without --skip_calibration at least once "
            "or provide --leader_id/--leader_calibration_dir pointing to an existing calibration file."
        )

    # Controls
    listener, events = init_keyboard_listener()
    dataset = make_dataset(env, args.repo_id, args.fps, args.use_camera)

    recorded = 0
    try:
        while recorded < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded+1} / {args.num_episodes}")
            episode_start = time.perf_counter()
            steps = 0
            dataset.clear_episode_buffer()
            while (time.perf_counter() - episode_start) < args.episode_time_s and not events["exit_early"]:
                # Read leader
                leader_obs = leader.get_action()
                leader_obs = {k: float(v) for k, v in leader_obs.items()}
                action = map_leader_to_action(leader_obs, env, args.use_degrees)
                obs, reward, terminated, truncated, info = env.step(action)

                # Write transition
                frame = {
                    "task": "teleop",
                    "action": action.astype(np.float32),
                    "observation.qpos": obs["qpos"].astype(np.float32),
                    "reward": np.array([reward], dtype=np.float32),
                    "done": np.array([terminated or truncated], dtype=bool),
                }
                if args.use_camera:
                    frame["observation.image_front"] = obs["image_front"]
                dataset.add_frame(frame)

                # Rate control
                steps += 1
                time.sleep(max(0.0, 1.0 / args.fps))

            dataset.save_episode()
            recorded += 1

            if recorded < args.num_episodes and not events["stop_recording"]:
                log_say("Reset window")
                t0 = time.perf_counter()
                while (time.perf_counter() - t0) < args.reset_time_s and not events["stop_recording"]:
                    time.sleep(0.1)
            obs, _ = env.reset()

    finally:
        leader.disconnect()
        dataset.finalize()
        if args.push_to_hub:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()


