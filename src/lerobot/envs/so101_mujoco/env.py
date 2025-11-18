import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"


@dataclass
class So101MujocoConfig:
    xml_path: str = str(DEFAULT_XML)
    render_mode: Optional[str] = "window"  # "window" | "human" | None
    control_timestep: float = 0.02  # 50 Hz
    use_camera: bool = False
    camera_width: int = 256
    camera_height: int = 256
    camera_name: str = "front_cam"
    camera_names: Optional[list[str]] = None  # Multiple cameras: ["top_cam", "wrist_cam", "right_cam"]
    init_qpos: Optional[np.ndarray] = None  # (n,)
    frame_skip: Optional[int] = None  # if None, auto-compute from control_timestep
    randomize_blocks: bool = False  # Randomize block positions on reset
    num_blocks: int = 3  # Number of blocks in the scene


class So101MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "window", "rgb_array"], "render_fps": 50}

    def __init__(self, config: Optional[So101MujocoConfig] = None):
        super().__init__()
        self.cfg = config or So101MujocoConfig()

        if not os.path.exists(self.cfg.xml_path):
            raise FileNotFoundError(f"MJCF file not found at {self.cfg.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nu = self.model.nu
        if self.nu == 0:
            # Fallback to joint-space control size
            self.nu = self.nq

        # Control range derived from position actuators
        if self.model.nu:
            self.ctrl_range = self.model.actuator_ctrlrange.copy()
            self.ctrl_mid = (self.ctrl_range[:, 0] + self.ctrl_range[:, 1]) / 2.0
            span = self.ctrl_range[:, 1] - self.ctrl_range[:, 0]
            self.ctrl_span = np.maximum(span / 2.0, 1e-6)
            self.actuator_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
                for i in range(self.model.nu)
            ]
            self.actuator_name_to_index = {name: idx for idx, name in enumerate(self.actuator_names)}
        else:
            self.ctrl_range = np.zeros((self.nu, 2), dtype=np.float32)
            self.ctrl_mid = np.zeros(self.nu, dtype=np.float32)
            self.ctrl_span = np.ones(self.nu, dtype=np.float32)
            self.actuator_names = [f"act_{i}" for i in range(self.nu)]
            self.actuator_name_to_index = {name: idx for idx, name in enumerate(self.actuator_names)}

        # Action: normalized [-1, 1] for each actuator
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

        # Observation: joint qpos (+ optional camera images)
        obs_dict = {
            "qpos": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.nq,),
                dtype=np.float32,
            )
        }
        if self.cfg.use_camera:
            if self.cfg.camera_names:
                # Multiple cameras
                for cam_name in self.cfg.camera_names:
                    obs_dict[f"image_{cam_name}"] = spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.cfg.camera_height, self.cfg.camera_width, 3),
                        dtype=np.uint8,
                    )
            else:
                # Single camera (backward compatibility)
                obs_dict["image_front"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.cfg.camera_height, self.cfg.camera_width, 3),
                    dtype=np.uint8,
                )
        self.observation_space = spaces.Dict(obs_dict)

        self.renderer = None
        if self.cfg.render_mode in ("window", "human"):
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(model=self.model, data=self.data)
            except Exception:
                self.viewer = None
        else:
            self.viewer = None

        # joint limits from model (or wide bounds if missing)
        jnt_range = getattr(self.model, "jnt_range", None)
        if jnt_range is not None and jnt_range.shape[0] >= self.nq:
            self.joint_limits = jnt_range[: self.nq].copy()
        else:
            self.joint_limits = np.stack([-np.ones(self.nq) * math.pi, np.ones(self.nq) * math.pi], axis=1)

        # Determine how many mj steps per control
        model_dt = self.model.opt.timestep
        if self.cfg.frame_skip is not None:
            self.frame_skip = max(1, int(self.cfg.frame_skip))
        else:
            self.frame_skip = max(1, int(round(self.cfg.control_timestep / model_dt)))
        self.control_dt = self.frame_skip * model_dt

        self._elapsed_steps = 0

    def _get_obs(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "qpos": self.data.qpos.copy().astype(np.float32),
        }
        if self.cfg.use_camera:
            if self.cfg.camera_names:
                # Multiple cameras
                for cam_name in self.cfg.camera_names:
                    img = self._render_offscreen(self.cfg.camera_width, self.cfg.camera_height, cam_name)
                    obs[f"image_{cam_name}"] = img
            else:
                # Single camera (backward compatibility)
                img = self._render_offscreen(self.cfg.camera_width, self.cfg.camera_height, self.cfg.camera_name)
                obs["image_front"] = img
        return obs

    def _render_offscreen(self, width: int, height: int, camera_name: str) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, width=width, height=height)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            cam_id = 0
        renderer.update_scene(self.data, camera=cam_id)
        frame = renderer.render()
        return frame

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        self._elapsed_steps = 0
        mujoco.mj_resetData(self.model, self.data)

        if self.cfg.init_qpos is not None and len(self.cfg.init_qpos) == self.nq:
            self.data.qpos[:] = self.cfg.init_qpos

        if self.model.nu:
            self.data.ctrl[: self.model.nu] = np.clip(self.ctrl_mid, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

        # Randomize block positions if enabled
        if self.cfg.randomize_blocks:
            self._randomize_blocks()

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    
    def _randomize_blocks(self):
        """Randomize block positions on the table."""
        # Table workspace bounds (relative to table center at 0.25, 0, 0.04)
        x_min, x_max = 0.15, 0.35  # 20cm range in x
        y_min, y_max = -0.15, 0.15  # 30cm range in y
        z_table = 0.065  # Height above table surface
        
        block_names = ["block_red", "block_blue", "block_green"][:self.cfg.num_blocks]
        
        for block_name in block_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block_name)
            if body_id == -1:
                continue
            
            # Random position on table
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            
            # Find qpos indices for this body (freejoint has 7 DOFs: 3 pos + 4 quat)
            jnt_id = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            
            # Set position
            self.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z_table]
            # Set orientation (identity quaternion)
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.nu)
        norm_action = np.clip(action, -1.0, 1.0)
        if self.model.nu:
            target_ctrl = self.ctrl_mid + norm_action * self.ctrl_span
            target_ctrl = np.clip(target_ctrl, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            self.data.ctrl[: self.model.nu] = target_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1

        # No task reward by default; plug-in later
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if self.cfg.render_mode in ("window", "human") and self.viewer is not None:
            self.viewer.sync()

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.cfg.render_mode == "rgb_array":
            return self._render_offscreen(self.cfg.camera_width, self.cfg.camera_height, self.cfg.camera_name)
        return None

    def close(self):
        self.viewer = None

    # Convenience helpers -------------------------------------------------
    def actuator_target_from_joint_dict(self, joint_targets: Dict[str, float]) -> np.ndarray:
        """
        Convert a dict of joint targets (in radians) keyed by joint/actuator name into normalized action space.
        Missing joints fall back to current ctrl target.
        """
        if not self.model.nu:
            raise RuntimeError("Model has no actuators to control.")
        target = self.ctrl_mid.copy()
        for name, value in joint_targets.items():
            if name not in self.actuator_name_to_index:
                continue
            idx = self.actuator_name_to_index[name]
            target[idx] = float(value)
        target = np.clip(target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        norm = (target - self.ctrl_mid) / self.ctrl_span
        return np.clip(norm, -1.0, 1.0)
    
    def get_block_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all blocks in the scene."""
        positions = {}
        block_names = ["block_red", "block_blue", "block_green"][:self.cfg.num_blocks]
        
        for block_name in block_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block_name)
            if body_id == -1:
                continue
            positions[block_name] = self.data.xpos[body_id].copy()
        
        return positions
    
    def check_stacking_success(self, tolerance: float = 0.03) -> bool:
        """
        Check if blocks are successfully stacked.
        Returns True if blocks are vertically aligned within tolerance.
        """
        positions = self.get_block_positions()
        if len(positions) < 2:
            return False
        
        # Get positions as list
        pos_list = list(positions.values())
        
        # Check if blocks are stacked (z-coordinates increasing, xy aligned)
        sorted_by_z = sorted(pos_list, key=lambda p: p[2])
        
        for i in range(len(sorted_by_z) - 1):
            lower = sorted_by_z[i]
            upper = sorted_by_z[i + 1]
            
            # Check xy alignment
            xy_dist = np.linalg.norm(lower[:2] - upper[:2])
            if xy_dist > tolerance:
                return False
            
            # Check z distance (should be approximately block height = 0.05m)
            z_dist = upper[2] - lower[2]
            if not (0.04 < z_dist < 0.06):  # Allow some tolerance
                return False
        
        return True



