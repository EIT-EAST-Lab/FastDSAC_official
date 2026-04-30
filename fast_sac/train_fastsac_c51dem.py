import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time

import tqdm
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict, from_module

from fast_sac_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
)
from hyperparams import FastSACProArgs
import tyro
from fast_sac import FastSACProActor, FastSACProCritic

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def main():
    args = tyro.cli(FastSACProArgs)
    print(args)
    run_name = f"FastSACPro_{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        # OLD: from environments.humanoid_bench_env import HumanoidBenchEnv
        from environments.humanoid_bench_env import HumanoidBenchEnvWithActionBounds, HumanoidBenchEnv

        env_type = "humanoid_bench"
        if args.action_bound:
            envs = HumanoidBenchEnvWithActionBounds(args.env_name, args.num_envs, device=device)
        else:
            envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        if args.action_bound:
            render_env = HumanoidBenchEnvWithActionBounds(
                args.env_name, 1, render_mode="rgb_array", device=device
            )
        else:
            render_env = HumanoidBenchEnv(args.env_name, 1, render_mode="rgb_array", device=device)
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv
        from environments.isaaclab_action_bounds import IsaacLabEnvWithActionBounds
        
        # Check if it's a locomotion task
        is_locomotion = any(keyword in args.env_name for keyword in ["Velocity", "Rough", "Flat", "Quadruped"])
        
        env_type = "isaaclab"
        # OLD: Create base environment directly
        base_env = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds if not is_locomotion else None,
        )
        if args.action_bound:
            envs = IsaacLabEnvWithActionBounds(
                base_env=base_env,
                scale_config=0.5,  # Default from IsaacLab locomotion configs
                enable_action_bounds=True,
            )
        else:
            envs = base_env  # Use base environment directly for manipulation
        
        eval_envs = envs
        render_env = envs
    else:
        # OLD: from environments.mujoco_playground_env import make_env
        from environments.mujoco_playground_action_bounds import MuJocoPlaygroundEnvWithActionBounds
        from environments.mujoco_playground_env import PlaygroundEvalEnvWrapper, make_env
        from mujoco_playground import registry

        env_type = "mujoco_playground"
        if args.action_bound:
        # OLD: envs, eval_envs, render_env = make_env(...)
        # Create training env with action bounds
            envs = MuJocoPlaygroundEnvWithActionBounds(
                env_name=args.env_name,
                num_envs=args.num_envs,
                seed=args.seed,
                device_rank=args.device_rank,
                use_tuned_reward=args.use_tuned_reward,
                use_domain_randomization=args.use_domain_randomization,
                use_push_randomization=args.use_push_randomization,
            )
            
            # Create eval and render envs (without action bounds for compatibility)
            eval_env_cfg = registry.get_default_config(args.env_name)
            is_humanoid_task = args.env_name in ["G1JoystickRoughTerrain", "G1JoystickFlatTerrain", "T1JoystickRoughTerrain", "T1JoystickFlatTerrain"]
            if is_humanoid_task and not args.use_push_randomization:
                eval_env_cfg.push_config.enable = False
                eval_env_cfg.push_config.magnitude_range = [0.0, 0.0]
            eval_env_raw = registry.load(args.env_name, config=eval_env_cfg)
            eval_envs = PlaygroundEvalEnvWrapper(eval_env_raw, eval_env_cfg.episode_length, args.env_name, args.num_eval_envs, args.seed)
            
            render_env_cfg = registry.get_default_config(args.env_name)
            if is_humanoid_task and not args.use_push_randomization:
                render_env_cfg.push_config.enable = False
                render_env_cfg.push_config.magnitude_range = [0.0, 0.0]
            render_env_raw = registry.load(args.env_name, config=render_env_cfg)
            render_env = PlaygroundEvalEnvWrapper(render_env_raw, render_env_cfg.episode_length, args.env_name, 1, args.seed)
        else:
            envs, eval_envs, render_env = make_env(
                args.env_name,
                args.seed,
                args.num_envs,
                args.num_eval_envs,
                args.device_rank,
                use_tuned_reward=args.use_tuned_reward,
                use_domain_randomization=args.use_domain_randomization,
                use_push_randomization=args.use_push_randomization,
            )

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    
    # FastSAC Pro uses Distributional Critic and Enhanced Actor
    
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args.gamma, device=device, g_max=10.0
        )
    else:
        reward_normalizer = nn.Identity()

    # FastSAC Pro uses Distributional Critic and Enhanced Actor (Reference style)
    # Construct dummy obs_indices for compatibility with reference-aligned Actor
    # We must provide 'start', 'end', and 'size' because the Actor uses them for slicing
    obs_indices = {"state": {"start": 0, "end": n_obs, "size": n_obs}}
    obs_keys = ["state"]
    
    activation_type = args.activation_type
    if activation_type == "relu":
        activation = nn.ReLU()
    elif activation_type == "gelu":
        activation = nn.GELU()
    elif activation_type == "silu":
        activation = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

    # NEW: Extract action bounds from environment wrapper
    action_scale = envs.action_scale if hasattr(envs, 'action_scale') else None
    action_bias = envs.action_bias if hasattr(envs, 'action_bias') else None
    if action_scale is not None:
        print(f"Using action bounds: scale_mean={action_scale.mean():.4f}, bias_mean={action_bias.mean():.4f}")
    else:
        print("No action bounds available from environment")

    # OLD: actor = FastSACProActor(obs_indices, obs_keys, n_act, ...)
    actor = FastSACProActor(
        obs_indices=obs_indices,
        obs_keys=obs_keys,
        n_act=n_act,
        num_envs=args.num_envs,
        hidden_dim=args.actor_hidden_dim,
        log_std_max=args.log_std_max,
        log_std_min=args.log_std_min,
        use_layer_norm=args.use_layer_norm,
        device=device,
        action_scale=action_scale,  # NEW
        action_bias=action_bias,     # NEW
    )
    actor_detach = FastSACProActor(
        obs_indices=obs_indices,
        obs_keys=obs_keys,
        n_act=n_act,
        num_envs=args.num_envs,
        hidden_dim=args.actor_hidden_dim,
        log_std_max=args.log_std_max,
        log_std_min=args.log_std_min,
        use_layer_norm=args.use_layer_norm,
        device=device,
        action_scale=action_scale,
        action_bias=action_bias,
    )
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    
    # Use explore for evaluation policy
    policy = actor_detach.explore

    qnet = FastSACProCritic(
        obs_indices=obs_indices,
        obs_keys=obs_keys,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
        use_layer_norm=args.use_layer_norm,
        num_q_networks=args.num_q_networks,
        device=device,
    )
    qnet_target = FastSACProCritic(
        obs_indices=obs_indices,
        obs_keys=obs_keys,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
        use_layer_norm=args.use_layer_norm,
        num_q_networks=args.num_q_networks,
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    # Fused AdamW
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_adam,
        betas=(0.9, 0.95),
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_adam,
        betas=(0.9, 0.95),
    )

    # Entropy
    target_entropy = -float(n_act) * args.target_entropy_ratio
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(args.alpha_init)], device=device))
    alpha_optimizer = optim.AdamW(
        [log_alpha], 
        lr=args.alpha_learning_rate,
        fused=args.fused_adam,
        betas=(0.9, 0.95),
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    # Define evaluation and rendering functions
    def evaluate():
        obs_normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs)
                # policy is explore, returns just actions
                actions = policy(obs, deterministic=True)

            next_obs, rewards, dones, _ = eval_envs.step(actions.float())
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        obs_normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        obs_normalizer.eval()
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type == "isaaclab":
            raise NotImplementedError(
                "We don't support rendering for IsaacLab environments"
            )
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs)
                actions = policy(obs, deterministic=True)
            next_obs, _, done, _ = render_env.step(actions.float())
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)

        obs_normalizer.train()
        return renders

    def update_main(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            # next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = data["next"]["observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                # Use get_actions_and_log_probs for Actor update logic
                next_state_actions, next_state_log_pi = actor.get_actions_and_log_probs(data["next"]["observations"])
                
                # C51 Projection with entropy correction in rewards/target
                adjusted_rewards = rewards - discount * bootstrap * log_alpha.exp() * next_state_log_pi
                
                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    adjusted_rewards,
                    bootstrap,
                    discount,
                )
                
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions) # logits
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            
            # Cross Entropy Loss
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(q_optimizer)
        scaler.update()
        
        # Autotune alpha
        alpha_loss = torch.tensor(0.0, device=device)
        if args.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                 alpha_loss = (-log_alpha.exp() * (next_state_log_pi.detach() + target_entropy)).mean()
            
            scaler.scale(alpha_loss).backward()
            scaler.unscale_(alpha_optimizer)
            scaler.step(alpha_optimizer)
            scaler.update()

        logs_dict["buffer_rewards"] = rewards.mean()
        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = target_value_max.detach()
        logs_dict["qf_min"] = target_value_min.detach()
        logs_dict["alpha"] = log_alpha.exp().detach()
        logs_dict["alpha_loss"] = alpha_loss.detach()
        
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            # Use get_actions_and_log_probs
            actions, log_pi = actor.get_actions_and_log_probs(data["observations"])
            
            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs) # [num_q, batch]
            
            # Expectation over Q-networks for actor loss
            qf_value = q_values.mean(dim=0)
            
            actor_loss = (log_alpha.exp().detach() * log_pi - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(actor_optimizer)
        scaler.update()

        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        logs_dict["log_pi"] = log_pi.detach().mean()
        return logs_dict

    if args.compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=mode)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=mode)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=mode)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=mode)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""

    while global_step < args.total_timesteps:
        logs_dict = TensorDict()
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            # policy is explore, returns just actions
            actions = policy(obs=norm_obs)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        if args.reward_normalization:
            update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]

        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )
        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        rb.extend(transition)

        batch_size = args.batch_size // args.num_envs
        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(batch_size)
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                raw_rewards = data["next"]["rewards"]
                data["next"]["rewards"] = normalize_reward(raw_rewards)
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                for param, target_param in zip(
                    qnet.parameters(), qnet_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0 and start_time is not None:
                # speed = (global_step - measure_burnin) / (time.time() - start_time)
                # pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "alpha": logs_dict["alpha"].mean(),
                        "alpha_loss": logs_dict["alpha_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                    }

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab"]:
                            obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length

                    if (
                        args.render_interval > 0
                        and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        if args.use_wandb:
                            wandb.log(
                                {
                                    "render_video": wandb.Video(
                                        np.array(renders).transpose(
                                            0, 3, 1, 2
                                        ),
                                        fps=30,
                                        format="gif",
                                    )
                                },
                                step=global_step,
                            )
                if args.use_wandb:
                    wandb.log(
                        {
                            #"speed": speed,
                            "frame": global_step * args.num_envs,
                            **logs,
                        },
                        step=global_step,
                    )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        pbar.update(1)

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )
    eval_avg_return, eval_avg_length = evaluate()
    logs["eval_avg_return"] = eval_avg_return
    logs["eval_avg_length"] = eval_avg_length
    if args.use_wandb:
        wandb.log(
            {
                "eval_avg_return": eval_avg_return,
                "eval_avg_length": eval_avg_length,
            },
            step=global_step,
        )


if __name__ == "__main__":
    main()
