from __future__ import annotations

import gymnasium as gym

import humanoid_bench
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import torch
from loguru import logger as log

# Disable all logging below CRITICAL level
log.remove()
log.add(lambda msg: False, level="CRITICAL")


def make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    if env_name in [
        "h1hand-push-v0",
        "h1-push-v0",
        "h1hand-cube-v0",
        "h1cube-v0",
        "h1hand-basketball-v0",
        "h1-basketball-v0",
        "h1hand-kitchen-v0",
        "h1-kitchen-v0",
    ]:
        max_episode_steps = 500
    else:
        max_episode_steps = 1000

    def _init():
        import humanoid_bench

        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)

        return env

    return _init


class HumanoidBenchEnv:
    """Wraps HumanoidBench environment to support parallel environments."""

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        # NOTE: HumanoidBench action space is already normalized to [-1, 1]
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs

        # Create the base environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
        )

        if env_name in [
            "h1hand-push-v0",
            "h1-push-v0",
            "h1hand-cube-v0",
            "h1cube-v0",
            "h1hand-basketball-v0",
            "h1-basketball-v0",
            "h1hand-kitchen-v0",
            "h1-kitchen-v0",
        ]:
            self.max_episode_steps = 500
        else:
            self.max_episode_steps = 1000

        # For compatibility with MuJoCo Playground
        self.asymmetric_obs = False  # For comptatibility with MuJoCo Playground
        self.num_obs = self.envs.observation_space.shape[-1]
        self.num_actions = self.envs.action_space.shape[-1]

    def reset(self):
        """Reset the environment."""
        observations = self.envs.reset()
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        return observations

    def render(self):
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos = self.envs.step(actions)

        # This will be used for getting 'true' next observations
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                infos["observations"]["raw"]["obs"][i] = raw_infos[i][
                    "terminal_observation"
                ]

        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds

        return observations, rewards, dones, infos


class HumanoidBenchEnvWithActionBounds:
    """
    Wrapper for HumanoidBench that computes joint-limit-aware action bounds.
    
    This wrapper:
    - Extracts default joint positions from the environment's qpos0_robot
    - Extracts physical joint limits from action_low/action_high
    - Computes action_scale and action_bias for symmetric scaling around defaults
    - Delegates all other operations to the wrapped environment
    """
    
    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        # Create base environment
        self._env = HumanoidBenchEnv(env_name, num_envs, render_mode, device)
        
        # Compute action bounds
        self._compute_action_bounds(env_name)
        
    def _compute_action_bounds(self, env_name):
        """Compute action_scale and action_bias from environment configuration."""
        # Get a temporary single environment to extract limits
        temp_env = gym.make(env_name)
        
        try:
            # Extract physical control limits (stored before normalization)
            action_low = temp_env.unwrapped.action_low  # Physical lower limits
            action_high = temp_env.unwrapped.action_high  # Physical upper limits
            
            # Extract default joint positions from task configuration
            task = temp_env.unwrapped.task
            robot = temp_env.unwrapped.robot
            
            # Map robot class to name in qpos0_robot dict
            robot_class_to_name = {
                'H1': 'h1',
                'H1Hand': 'h1hand',
                'H1SimpleHand': 'h1simplehand',
                'H1Touch': 'h1touch',
                'H1Strong': 'h1strong',
                'G1': 'g1'
            }
            robot_class_name = robot.__class__.__name__
            robot_name = robot_class_to_name.get(robot_class_name, 'h1')
            
            # Parse qpos0_robot to get default joint angles
            qpos0_str = task.qpos0_robot.get(robot_name, "")
            qpos0_values = np.fromstring(qpos0_str, sep=' ')
            
            # The qpos includes: [base_pos(3), base_quat(4), joint_angles(...)]
            # We need only the joint angles part
            n_joints = len(action_low)
            default_joint_angles = qpos0_values[7:7+n_joints]  # Skip base pose (3+4=7)
            
            # Compute symmetric action bounds
            # max_range = max(|default - low|, |default - high|) for each joint
            range_to_lower = np.abs(default_joint_angles - action_low)
            range_to_upper = np.abs(default_joint_angles - action_high)
            max_range = np.maximum(range_to_lower, range_to_upper)
            
            # Convert to torch tensors
            self.action_scale = torch.from_numpy(max_range).to(
                device=self._env.sim_device, dtype=torch.float
            )
            # CRITICAL: action_bias must be 0 because HumanoidBench environment ALREADY adds default_pos!
            # Environment does: motor_target = default_pos + actor_output * pd_control
            # So actor should output: tanh * (max_range / pd_scale) + 0
            self.action_bias = torch.zeros_like(torch.from_numpy(default_joint_angles)).to(
                device=self._env.sim_device, dtype=torch.float
            )
            
            print(f"[ActionBounds] Computed for {robot_name}: "
                  f"scale_mean={self.action_scale.mean():.4f}, "
                  f"bias_mean={self.action_bias.mean():.4f}")
            
        except Exception as e:
            # Fallback: use identity scaling if computation fails
            print(f"[ActionBounds] Warning: Could not compute bounds ({e}). Using defaults.")
            n_act = self._env.num_actions
            self.action_scale = torch.ones(n_act, device=self._env.sim_device)
            self.action_bias = torch.zeros(n_act, device=self._env.sim_device)
        finally:
            temp_env.close()
    
    # Delegate all environment operations to wrapped environment
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment."""
        return getattr(self._env, name)
    
    def reset(self):
        return self._env.reset()
    
    def step(self, actions):
        return self._env.step(actions)
    
    def render(self):
        return self._env.render()

