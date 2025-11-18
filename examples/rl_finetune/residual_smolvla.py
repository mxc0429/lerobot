"""
Residual SmolVLA Policy Wrapper

This module provides a wrapper for SmolVLA policy that adds residual RL fine-tuning capability.
It is inspired by the residual policy approach from the RL folder, adapted for lerobot's SmolVLA model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from collections import deque
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


class ResidualMLP(nn.Module):
    """
    MLP for residual policy with actor and critic networks.
    Takes normalized observation + base action and outputs:
    - Actor: residual action correction
    - Critic: value estimate
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        action_scale: float = 0.1,
        critic_hidden_dim: int = 256,
        critic_num_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_scale = action_scale
        
        input_dim = obs_dim + action_dim  # obs + base action
        
        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dim))
        actor_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            actor_layers.append(nn.Linear(hidden_dim, hidden_dim))
            actor_layers.append(nn.ReLU())
        
        # Output layer (no activation, will be scaled)
        actor_layers.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Initialize actor output layer with small weights
        nn.init.normal_(self.actor[-1].weight, std=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        
        # Critic network (value function)
        critic_layers = []
        critic_layers.append(nn.Linear(input_dim, critic_hidden_dim))
        critic_layers.append(nn.ReLU())
        
        for _ in range(critic_num_layers - 1):
            critic_layers.append(nn.Linear(critic_hidden_dim, critic_hidden_dim))
            critic_layers.append(nn.ReLU())
        
        critic_layers.append(nn.Linear(critic_hidden_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize critic output layer
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)
    
    def forward(self, obs_and_action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_and_action: Concatenated [normalized_obs, normalized_base_action]
        Returns:
            Residual action (already scaled)
        """
        return self.actor(obs_and_action) * self.action_scale
    
    def get_action(self, obs_and_action: torch.Tensor) -> torch.Tensor:
        """Get deterministic residual action"""
        return self.forward(obs_and_action)
    
    def get_value(self, obs_and_action: torch.Tensor) -> torch.Tensor:
        """Get value estimate from critic network"""
        return self.critic(obs_and_action)


class ResidualSmolVLAPolicy:
    """
    Wrapper for SmolVLA policy that adds residual RL fine-tuning.
    
    This wrapper:
    1. Loads a base SmolVLA policy
    2. Adds a small residual MLP that learns to correct the base policy's actions
    3. During inference: action = base_action + residual_action
    4. The residual policy can be fine-tuned using RL methods (e.g., PPO)
    """
    
    def __init__(
        self,
        base_policy: SmolVLAPolicy,
        residual_hidden_dim: int = 256,
        residual_num_layers: int = 2,
        residual_action_scale: float = 0.1,
        device: str = "cuda",
    ):
        """
        Args:
            base_policy: The pre-trained SmolVLA policy
            residual_hidden_dim: Hidden dimension for residual MLP
            residual_num_layers: Number of layers in residual MLP
            residual_action_scale: Scale factor for residual actions
            device: Device to run on
        """
        self.base_policy = base_policy
        self.device = torch.device(device)
        self.base_policy.to(self.device)
        self.base_policy.eval()
        
        # Get action dimension from base policy config
        action_dim = base_policy.config.action_dim
        
        # Estimate observation dimension (state features)
        # For SmolVLA, we'll use a simplified state representation
        # In practice, you might want to extract features from the model
        obs_dim = 64  # This should match your state feature dimension
        
        # Create residual policy with critic
        self.residual_policy = ResidualMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=residual_hidden_dim,
            num_layers=residual_num_layers,
            action_scale=residual_action_scale,
            critic_hidden_dim=residual_hidden_dim,
            critic_num_layers=residual_num_layers,
        ).to(self.device)
        
        # History for temporal features
        self.observation_history = deque(maxlen=base_policy.config.obs_horizon)
        self.base_action_history = deque(maxlen=base_policy.config.action_horizon)
        
    def load_residual_weights(self, checkpoint_path: str):
        """Load pre-trained residual policy weights"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "residual_policy_state_dict" in checkpoint:
            self.residual_policy.load_state_dict(checkpoint["residual_policy_state_dict"])
        elif "model_state_dict" in checkpoint:
            # Try to extract residual policy weights
            residual_state_dict = {
                k.replace("residual_policy.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
                if k.startswith("residual_policy.")
            }
            if residual_state_dict:
                self.residual_policy.load_state_dict(residual_state_dict)
            else:
                raise ValueError("Could not find residual policy weights in checkpoint")
        else:
            self.residual_policy.load_state_dict(checkpoint)
        
        print(f"Loaded residual policy weights from {checkpoint_path}")
    
    def extract_state_features(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract state features from observation for SO101 with 3 cameras.
        
        Uses SmolVLA's vision encoders to extract features from images,
        and combines with robot state.
        """
        features = []
        
        # 1. Extract robot state features
        if "state" in observation:
            state = observation["state"]
            if isinstance(state, torch.Tensor):
                # Ensure batch dimension
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                features.append(state.flatten(start_dim=1))
        
        # 2. Extract image features using SmolVLA's encoders
        # SmolVLA has vision encoders that we can use
        if hasattr(self.base_policy, 'model') and hasattr(self.base_policy.model, 'vlm'):
            # Get image keys (camera1, camera2, camera3)
            image_keys = [f"observation.images.camera{i}" for i in [1, 2, 3]]
            image_keys_alt = [f"images.camera{i}" for i in [1, 2, 3]]
            
            # Try to find images in observation
            images_found = []
            for key in image_keys + image_keys_alt:
                if key in observation:
                    img = observation[key]
                    # Ensure correct format: (B, C, H, W)
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    if img.ndim == 4 and img.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
                        img = img.permute(0, 3, 1, 2)
                    images_found.append(img)
            
            if images_found:
                # Use SmolVLA's vision encoder to extract features
                # Note: This is a simplified version - you may need to adjust based on actual model structure
                try:
                    # Prepare images for SmolVLA (resize to 224x224, normalize)
                    import torch.nn.functional as F
                    prepared_images = []
                    for img in images_found:
                        # Resize to 224x224 (SmolVLA standard)
                        if img.shape[-2:] != (224, 224):
                            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                        # Normalize from [0, 1] to [-1, 1] if needed
                        if img.max() <= 1.0:
                            img = img * 2.0 - 1.0
                        prepared_images.append(img)
                    
                    # Extract features using vision encoder
                    # This is a placeholder - you'll need to access the actual encoder
                    # For now, we'll use a simple CNN feature extractor
                    if not hasattr(self, '_image_feature_extractor'):
                        # Create a simple feature extractor if not exists
                        self._image_feature_extractor = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=8, stride=4),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=4, stride=2),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool2d((4, 4)),
                            nn.Flatten(),
                        ).to(self.device)
                    
                    # Extract features from each camera
                    camera_features = []
                    for img in prepared_images:
                        feat = self._image_feature_extractor(img)
                        camera_features.append(feat)
                    
                    # Concatenate all camera features
                    if camera_features:
                        image_features = torch.cat(camera_features, dim=-1)
                        features.append(image_features)
                except Exception as e:
                    print(f"Warning: Could not extract image features: {e}")
        
        # 3. Combine all features
        if features:
            combined_features = torch.cat(features, dim=-1)
        else:
            # Fallback: create a simple feature vector
            batch_size = next(iter(observation.values())).shape[0] if observation else 1
            combined_features = torch.zeros(batch_size, 64, device=self.device)
        
        return combined_features
    
    def reset(self):
        """Reset the policy (clear history)"""
        self.observation_history.clear()
        self.base_action_history.clear()
        if hasattr(self.base_policy, 'reset'):
            self.base_policy.reset()
    
    @torch.no_grad()
    def predict_action(
        self,
        observation: Dict[str, torch.Tensor],
        task: Optional[str] = None,
        use_residual: bool = True,
    ) -> torch.Tensor:
        """
        Predict action using base policy + residual correction.
        
        Args:
            observation: Observation dictionary
            task: Task description (for SmolVLA)
            use_residual: Whether to apply residual correction
        
        Returns:
            Action tensor
        """
        # Get base action from SmolVLA
        base_action = self.base_policy.predict_action(observation, task=task)
        
        if not use_residual:
            return base_action
        
        # Normalize base action (assuming base policy has normalizer)
        # This is a simplified version - adjust based on your actual normalizer
        if hasattr(self.base_policy, 'normalizer'):
            normalized_base_action = self.base_policy.normalizer(
                base_action, "action", forward=True
            )
        else:
            # Fallback: simple normalization (adjust based on your action space)
            normalized_base_action = base_action / 10.0  # Adjust scale as needed
        
        # Extract state features
        state_features = self.extract_state_features(observation)
        
        # Concatenate state features and normalized base action
        residual_input = torch.cat([state_features, normalized_base_action], dim=-1)
        
        # Get residual correction
        residual_action = self.residual_policy.get_action(residual_input)
        
        # Apply residual to base action
        corrected_action = base_action + residual_action
        
        return corrected_action
    
    def get_value(self, observation: Dict[str, torch.Tensor], base_action: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate from critic network.
        
        Args:
            observation: Observation dictionary
            base_action: Base action from SmolVLA (normalized)
        
        Returns:
            Value estimate tensor
        """
        # Normalize base action
        if hasattr(self.base_policy, 'normalizer'):
            normalized_base_action = self.base_policy.normalizer(
                base_action, "action", forward=True
            )
        else:
            normalized_base_action = base_action / 10.0
        
        # Extract state features
        state_features = self.extract_state_features(observation)
        
        # Concatenate state features and normalized base action
        residual_input = torch.cat([state_features, normalized_base_action], dim=-1)
        
        # Get value estimate
        return self.residual_policy.get_value(residual_input)
    
    def get_residual_parameters(self):
        """Get parameters of residual policy (for RL training)"""
        return list(self.residual_policy.parameters())
    
    def get_actor_parameters(self):
        """Get parameters of actor network only"""
        return list(self.residual_policy.actor.parameters())
    
    def get_critic_parameters(self):
        """Get parameters of critic network only"""
        return list(self.residual_policy.critic.parameters())
    
    def get_base_parameters(self):
        """Get parameters of base policy (usually frozen during RL)"""
        return list(self.base_policy.parameters())
    
    def train_mode(self):
        """Set to training mode"""
        self.residual_policy.train()
        self.base_policy.eval()  # Keep base policy in eval mode
    
    def eval_mode(self):
        """Set to evaluation mode"""
        self.residual_policy.eval()
        self.base_policy.eval()

