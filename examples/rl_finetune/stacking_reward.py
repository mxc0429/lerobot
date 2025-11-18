"""
Reward function for block stacking task with SO101 robot.

This reward function is designed for the task of stacking one block on top of another
using a SO101 robot arm with 3 cameras (top, wrist, side).
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import math


class StackingRewardFunction:
    """
    Reward function for block stacking task.
    
    The reward consists of:
    1. Success reward: Large reward when blocks are successfully stacked
    2. Distance rewards: Reward for moving blocks closer to target position
    3. Grasp reward: Reward for successfully grasping the block
    4. Stacking progress: Reward for vertical alignment
    5. Step penalty: Small penalty per step to encourage efficiency
    """
    
    def __init__(
        self,
        success_reward: float = 50.0,
        step_penalty: float = -0.01,
        distance_reward_scale: float = 0.5,
        grasp_reward: float = 5.0,
        stacking_progress_scale: float = 2.0,
        height_reward_scale: float = 1.0,
        stability_reward: float = 10.0,
    ):
        """
        Args:
            success_reward: Reward when task is successfully completed
            step_penalty: Penalty per step to encourage efficiency
            distance_reward_scale: Scale factor for distance-based rewards
            grasp_reward: Reward for successfully grasping a block
            stacking_progress_scale: Scale factor for stacking progress rewards
            height_reward_scale: Scale factor for height-based rewards
            stability_reward: Reward for maintaining stable grasp
        """
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.distance_reward_scale = distance_reward_scale
        self.grasp_reward = grasp_reward
        self.stacking_progress_scale = stacking_progress_scale
        self.height_reward_scale = height_reward_scale
        self.stability_reward = stability_reward
        
        # Track previous state for computing progress
        self.prev_block_pos = None
        self.prev_gripper_state = None
        self.prev_height_diff = None
    
    def compute_reward(
        self,
        observation: Dict,
        action: torch.Tensor,
        next_observation: Optional[Dict] = None,
        done: bool = False,
        info: Optional[Dict] = None,
    ) -> float:
        """
        Compute reward for the current step.
        
        Args:
            observation: Current observation dictionary
            action: Action taken
            next_observation: Next observation (optional, for computing progress)
            done: Whether episode is done
            info: Additional info dict (may contain success flag, block positions, etc.)
        
        Returns:
            Reward value
        """
        reward = self.step_penalty
        
        # 1. Success reward (highest priority)
        if done and info and info.get("success", False):
            reward += self.success_reward
            return reward
        
        # 2. Extract block positions from observation or info
        # This depends on your observation structure
        block_positions = self._extract_block_positions(observation, info)
        
        if block_positions is not None:
            block1_pos, block2_pos = block_positions
            
            # 3. Distance-based reward (encourage moving blocks closer)
            if self.prev_block_pos is not None:
                prev_block1, prev_block2 = self.prev_block_pos
                
                # Compute distance reduction
                prev_dist = np.linalg.norm(prev_block1[:2] - prev_block2[:2])  # XY distance
                curr_dist = np.linalg.norm(block1_pos[:2] - block2_pos[:2])
                dist_reduction = prev_dist - curr_dist
                reward += self.distance_reward_scale * dist_reduction
            
            # 4. Stacking progress reward (vertical alignment)
            # Reward for aligning blocks vertically
            xy_distance = np.linalg.norm(block1_pos[:2] - block2_pos[:2])
            alignment_reward = max(0, 1.0 - xy_distance / 0.05)  # Reward if within 5cm
            reward += self.stacking_progress_scale * alignment_reward
            
            # 5. Height reward (encourage lifting block1 above block2)
            height_diff = block1_pos[2] - block2_pos[2]
            if height_diff > 0.02:  # Block1 is above block2
                reward += self.height_reward_scale * min(height_diff / 0.1, 1.0)
            
            # 6. Stability reward (if blocks are close and aligned)
            if xy_distance < 0.03 and height_diff > 0.01:
                reward += self.stability_reward
            
            # Update previous positions
            self.prev_block_pos = (block1_pos.copy(), block2_pos.copy())
        
        # 7. Grasp reward (based on gripper state)
        gripper_state = self._extract_gripper_state(observation)
        if gripper_state is not None:
            if self.prev_gripper_state is not None:
                # Reward for closing gripper when near a block
                if gripper_state < 0.5 and self.prev_gripper_state >= 0.5:  # Gripper closed
                    # Check if near a block (simplified)
                    if block_positions:
                        block1_pos, _ = block_positions
                        # Assume end-effector position is in observation
                        ee_pos = self._extract_end_effector_pos(observation)
                        if ee_pos is not None:
                            dist_to_block = np.linalg.norm(ee_pos - block1_pos)
                            if dist_to_block < 0.05:  # Within 5cm
                                reward += self.grasp_reward
            
            self.prev_gripper_state = gripper_state
        
        return reward
    
    def _extract_block_positions(
        self, observation: Dict, info: Optional[Dict]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract block positions from observation or info.
        
        This is a placeholder - you need to implement based on your actual observation structure.
        Options:
        1. Use computer vision to detect blocks in images
        2. Use state information if available
        3. Use info dict if positions are provided
        """
        # Option 1: From info dict (if available)
        if info:
            if "block1_pos" in info and "block2_pos" in info:
                return (
                    np.array(info["block1_pos"]),
                    np.array(info["block2_pos"])
                )
        
        # Option 2: From observation state
        if "state" in observation:
            state = observation["state"]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            elif isinstance(state, np.ndarray):
                pass
            else:
                return None
            
            # Assume state contains block positions
            # Adjust indices based on your actual state structure
            # Example: state[6:9] = block1_pos, state[9:12] = block2_pos
            if state.shape[-1] >= 12:
                block1_pos = state[..., 6:9]
                block2_pos = state[..., 9:12]
                return (block1_pos, block2_pos)
        
        # Option 3: Use computer vision (placeholder)
        # You would implement block detection using the camera images here
        # For now, return None
        return None
    
    def _extract_gripper_state(self, observation: Dict) -> Optional[float]:
        """Extract gripper state from observation (0=open, 1=closed)"""
        if "state" in observation:
            state = observation["state"]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            # Assume gripper state is in state vector
            # Adjust index based on your actual state structure
            # Example: state[5] = gripper position
            if isinstance(state, np.ndarray) and state.shape[-1] > 5:
                return float(state[..., 5])  # Adjust index as needed
        
        # Alternative: from action
        if isinstance(observation, dict) and "action" in observation:
            action = observation["action"]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if isinstance(action, np.ndarray) and action.shape[-1] > 5:
                return float(action[..., 5])  # Adjust index as needed
        
        return None
    
    def _extract_end_effector_pos(self, observation: Dict) -> Optional[np.ndarray]:
        """Extract end-effector position from observation"""
        if "state" in observation:
            state = observation["state"]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            # Assume end-effector position is in state vector
            # Adjust indices based on your actual state structure
            # Example: state[0:3] = end-effector position
            if isinstance(state, np.ndarray) and state.shape[-1] >= 3:
                return state[..., 0:3]
        
        return None
    
    def reset(self):
        """Reset internal state"""
        self.prev_block_pos = None
        self.prev_gripper_state = None
        self.prev_height_diff = None


class VisionBasedStackingReward(StackingRewardFunction):
    """
    Extended reward function that uses computer vision to detect blocks.
    
    This version uses the camera images to detect block positions,
    which is more robust but requires implementing block detection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add a block detector here if needed
        self.block_detector = None
    
    def _extract_block_positions(
        self, observation: Dict, info: Optional[Dict]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract block positions using computer vision.
        
        This is a placeholder - you need to implement block detection
        using the camera images (camera1, camera2, camera3).
        """
        # Try info first
        result = super()._extract_block_positions(observation, info)
        if result is not None:
            return result
        
        # Use computer vision to detect blocks
        # This requires implementing block detection
        # For now, return None (you'll need to implement this)
        
        # Example approach:
        # 1. Get images from observation
        # 2. Use object detection (e.g., YOLO, or simple color-based detection)
        # 3. Convert pixel positions to 3D positions using camera calibration
        # 4. Return block positions
        
        return None

