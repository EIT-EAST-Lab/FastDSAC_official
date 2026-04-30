"""Usage:
python fast_sac/train_fastdsac.py --env_name h1hand-balance_hard-v0 --exp_name FastDSAC_eval_hard --render_interval 5000 --seed 777 --reward_normalization --project "FastTD3 vs FastDSAC"   """

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
from torch.distributions import Normal
from torch.nn.functional import huber_loss

from tensordict import TensorDict, from_module

from fast_sac_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    get_joint_names,
)
from hyperparams import get_args
from fast_sac import Actor, Critic, GaussianDistCritic, Hetero_DSACT_Actor, DSACT_EnhancedActor, DSACT_EnhancedActor_explr

# For Heatmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

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
        from environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
    else:
        from environments.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        env_type = "mujoco_playground"
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
    action_low, action_high = -1.0, 1.0

    mean_std1= -1.0
    mean_std2= -1.0
    tau_b = args.tau_b
    reward_scale = args.reward_scale
    bound_beta = args.bound_beta

    activation_type = args.activation_type
    if activation_type == "relu":
        activation = nn.ReLU()
    elif activation_type == "gelu":
        activation = nn.GELU()
    elif activation_type == "silu":
        activation = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

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

    actor = DSACT_EnhancedActor_explr(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        device=device,
        init_scale=args.init_scale,
        hidden_dim=args.actor_hidden_dim,
        log_std_max=args.log_std_max,
        log_std_min=args.log_std_min,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        activation=activation,
        use_layer_norm=args.use_layer_norm,
        temperature=args.temperature,
    )
    # actor = EnhancedActor(
    #     n_obs=n_obs,
    #     n_act=n_act,
    #     num_envs=args.num_envs,
    #     device=device,
    #     init_scale=args.init_scale,
    #     hidden_dim=args.actor_hidden_dim,
    #     scale_min=args.scale_min,
    #     scale_max=args.scale_max,
    # )
    # actor_detach = Actor(
    #     n_obs=n_obs,
    #     n_act=n_act,
    #     num_envs=args.num_envs,
    #     device=device,
    #     init_scale=args.init_scale,
    #     hidden_dim=args.actor_hidden_dim,
    # )
    # Copy params to actor_detach without grad
    # from_module(actor).data.to_module(actor_detach)
    # policy = actor.forward
    policy = actor.explore

    qnet = GaussianDistCritic(
        n_obs=n_critic_obs,
        n_act=n_act,
        hidden_dim=args.critic_hidden_dim,
        activation=activation,
        use_layer_norm=args.use_layer_norm,
        device=device,
    )
    qnet_target = GaussianDistCritic(
        n_obs=n_critic_obs,
        n_act=n_act,
        hidden_dim=args.critic_hidden_dim,
        activation=activation,
        use_layer_norm=args.use_layer_norm,
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    # q_optimizer = optim.AdamW(
    #     list(qnet.parameters()),
    #     lr=args.critic_learning_rate,
    #     weight_decay=args.weight_decay,
    # )
    # actor_optimizer = optim.AdamW(
    #     list(actor.parameters()),
    #     lr=args.actor_learning_rate,
    #     weight_decay=args.weight_decay,
    # )
    # #TODO (Jolyne): entropy target?
    # # Auto-tune target entropy based on action dimension
    # target_entropy = - float(n_act) * args.target_entropy_ratio
    # #TODO (Jolyne): 并行化alpha，每个环境alpha独立会不会影响buffer和actor、critic的更新？
    # log_alpha = torch.ones(1, requires_grad=True, device=device)
    # log_alpha.data.copy_(torch.tensor([np.log(args.alpha_init)], device=device))  # Start with higher alpha for exploration
    # alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_learning_rate)

    # Fused AdamW like FastSAC Pro
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_adam,
        betas=(0.9, 0.95) if args.tuned_betas else (0.9, 0.999),
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_adam,
        betas=(0.9, 0.95) if args.tuned_betas else (0.9, 0.999),
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

        # Run for a fixed number of steps
        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs)
                # _, _, action_mean = policy(obs)
                _, _, action_mean = actor(obs)
                actions = action_mean

            next_obs, rewards, dones, _ = eval_envs.step(actions.float())
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards.squeeze(), episode_returns
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

        # Quick rollout for rendering
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
                # _, _, action_mean = policy(obs)
                _, _, action_mean = actor(obs)
                actions = action_mean
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
        nonlocal mean_std1, mean_std2

        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"].unsqueeze(-1)
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()    
            bootstrap = bootstrap.unsqueeze(-1)
            discount = args.gamma ** data["next"]["effective_n_steps"].unsqueeze(-1)


            rewards = rewards * reward_scale
            
            def __compute_target_q(r, done, q,q_std, q_next, q_next_sample, log_prob_a_next):
                target_q = r + bootstrap * discount * (
                    q_next.unsqueeze(-1) - log_alpha.exp() * log_prob_a_next
                )
                target_q_sample = r + bootstrap * discount * (
                    q_next_sample.unsqueeze(-1) - log_alpha.exp() * log_prob_a_next
                )
                # td_bound = bound_beta * q_std  #NOTE: Try this
                # difference = torch.clamp(target_q_sample - q.unsqueeze(-1), -td_bound, td_bound)
                # target_q_bound = q.unsqueeze(-1) + difference
                return target_q.detach(), target_q_sample.detach()


            stochaQ1, stochaQ2 = qnet(critic_observations, actions)
            q1_m, q1_std = stochaQ1[:, 0], stochaQ1[:, 1]
            q2_m, q2_std = stochaQ2[:, 0], stochaQ2[:, 1]

            if mean_std1 == -1.0:
                mean_std1 = torch.mean(q1_std.detach())
            else:
                mean_std1 = (1 - tau_b) * mean_std1 + tau_b * torch.mean(q1_std.detach())

            if mean_std2 == -1.0:
                mean_std2 = torch.mean(q2_std.detach())
            else:
                mean_std2 = (1 - tau_b) * mean_std2 + tau_b * torch.mean(q2_std.detach())
            

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor(next_observations)
                #TODO (Jolyne): use Action smoothing?
                next_stochaQ1_target, next_stochaQ2_target = qnet_target(next_critic_observations, next_state_actions)
                next_q1_m, next_q1_std = next_stochaQ1_target[:, 0], next_stochaQ1_target[:, 1]
                next_q2_m, next_q2_std = next_stochaQ2_target[:, 0], next_stochaQ2_target[:, 1]
                # next_q1 = Normal(next_q1_m, next_q1_std).rsample()
                # next_q2 = Normal(next_q2_m, next_q2_std).rsample()
                
                # # Add noise clamping as per reference TODO (Jolyne): change clamp range or discard clamp (rsample)?
                z1 = torch.randn_like(next_q1_m).clamp(-3, 3)
                z2 = torch.randn_like(next_q2_m).clamp(-3, 3)
                next_q1 = next_q1_m + z1 * next_q1_std
                next_q2 = next_q2_m + z2 * next_q2_std
                
                if args.use_cdq:
                    q_next = torch.min(next_q1_m, next_q2_m)
                    q_next_sample = torch.where(next_q1_m < next_q2_m, next_q1, next_q2)
                else:
                    q_next = (next_q1_m + next_q2_m) / 2.0
                    q_next_sample = (next_q1 + next_q2) / 2.0

                target_q1, target_q1_bound = __compute_target_q(
                    rewards, dones, q1_m.detach(), mean_std1, q_next, q_next_sample, next_state_log_pi
                )

                target_q2, target_q2_bound = __compute_target_q(
                    rewards, dones, q2_m.detach(), mean_std2, q_next, q_next_sample, next_state_log_pi
                )

            q1_std_detach = torch.clamp(q1_std, min=0.).detach()
            q2_std_detach = torch.clamp(q2_std, min=0.).detach()
            bias = 0.000001
            ratio1 = (torch.pow(mean_std1, 2) / (torch.pow(q1_std_detach.unsqueeze(-1), 2) + bias)).clamp(min=0.1, max=10)
            ratio2 = (torch.pow(mean_std2, 2) / (torch.pow(q2_std_detach.unsqueeze(-1), 2) + bias)).clamp(min=0.1, max=10)

            q1_loss = torch.mean(ratio1 *(huber_loss(q1_m.unsqueeze(-1), target_q1, delta = 50, reduction='none') 
                                      + q1_std.unsqueeze(-1) *(q1_std_detach.unsqueeze(-1).pow(2) - huber_loss(q1_m.detach().unsqueeze(-1), target_q1_bound, delta = 50, reduction='none'))/(q1_std_detach.unsqueeze(-1) +bias)
                            ))
            q2_loss = torch.mean(ratio2 *(huber_loss(q2_m.unsqueeze(-1), target_q2, delta = 50, reduction='none')
                                      + q2_std.unsqueeze(-1) *(q2_std_detach.unsqueeze(-1).pow(2) - huber_loss(q2_m.detach().unsqueeze(-1), target_q2_bound, delta = 50, reduction='none'))/(q2_std_detach.unsqueeze(-1) +bias)
                            ))
            
            qf_loss = q1_loss + q2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(q_optimizer)
        scaler.update()


        logs_dict["buffer_rewards"] = rewards.mean()
        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["next_q1_m"] = next_q1_m.detach().mean()
        logs_dict["next_q1_max"] = next_q1_m.detach().max()
        logs_dict["next_q1_min"] = next_q1_m.detach().min()
        logs_dict["next_q2_m"] = next_q2_m.detach().mean()
        logs_dict["target_q1"] = target_q1.mean().detach()
        logs_dict["target_q2"] = target_q2.mean().detach()
        logs_dict["mean_std1"] = mean_std1.mean().detach()
        logs_dict["next_q1_std"] = next_q1_std.detach().mean()
        logs_dict["next_q2_std"] = next_q2_std.detach().mean()
        logs_dict["next_q1_std_min"] = next_q1_std.min().detach()
        logs_dict["next_q1_std_max"] = next_q1_std.max().detach()
        logs_dict["next_q2_std_min"] = next_q2_std.min().detach()
        logs_dict["next_q2_std_max"] = next_q2_std.max().detach()
        logs_dict["qf_next_max"] = q_next_sample.max().detach()
        logs_dict["qf_next_min"] = q_next_sample.min().detach()
        logs_dict["qf_next_mean"] = q_next_sample.mean().detach()
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

            pi, log_pi, _, explr_weight = actor(data["observations"], return_explr_weight=True)
            stochaQ1, stochaQ2 = qnet(critic_observations, pi)
            q_m1, q_m2 = stochaQ1[:, 0], stochaQ2[:, 0]
            
            
            if args.use_cdq:
                qf_value = torch.minimum(q_m1, q_m2)
            else:
                qf_value = (q_m1 + q_m2) / 2.0
            actor_loss = ((log_alpha.exp().detach() * log_pi) - qf_value.unsqueeze(-1)).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(actor_optimizer)
        scaler.update()

        # Update alpha
        alpha_optimizer.zero_grad(set_to_none=True)
        # with torch.no_grad():
        #     _, log_pi, _ = actor(data["observations"])
        # alpha_loss = -log_alpha.exp() * (log_pi + target_entropy).detach().mean()
        # Reuse log_pi from actor update (standard SAC practice)
        # Note: log_pi is already cast to float32 above if needed
        alpha_loss = -log_alpha.exp() * (log_pi.detach() + target_entropy).mean()
        scaler.scale(alpha_loss).backward()
        scaler.unscale_(alpha_optimizer)
        alpha_optimizer.step()
        # with torch.no_grad(): by antigravity
        #     log_alpha.clamp_(max=1.0)


        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        logs_dict["alpha"] = log_alpha.exp().detach()
        logs_dict["alpha_loss"] = alpha_loss.detach()
        logs_dict["explr_weight"] = explr_weight.detach()
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
            actions, _, _ = policy(obs=norm_obs, dones=dones)

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
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "alpha": logs_dict["alpha"].mean(),
                        "alpha_loss": logs_dict["alpha_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "next_q1_m": logs_dict["next_q1_m"].mean(),
                        "next_q1_max": logs_dict["next_q1_max"].mean(),
                        "next_q1_min": logs_dict["next_q1_min"].mean(),
                        "next_q2_m": logs_dict["next_q2_m"].mean(),
                        "target_q1": logs_dict["target_q1"].mean(),
                        "target_q2": logs_dict["target_q2"].mean(),
                        "mean_std1": logs_dict["mean_std1"].mean(),
                        "next_q1_std": logs_dict["next_q1_std"].mean(),
                        "next_q2_std": logs_dict["next_q2_std"].mean(),
                        "next_q1_std_min": logs_dict["next_q1_std_min"].mean(),
                        "next_q2_std_min": logs_dict["next_q2_std_min"].mean(),
                        "next_q1_std_max": logs_dict["next_q1_std_max"].mean(),
                        "next_q2_std_max": logs_dict["next_q2_std_max"].mean(),
                        "qf_next_max": logs_dict["qf_next_max"].mean(),
                        "qf_next_min": logs_dict["qf_next_min"].mean(),
                        "qf_next_mean": logs_dict["qf_next_mean"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "log_pi": logs_dict["log_pi"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                    }

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab"]:
                            # NOTE: Hacky way of evaluating performance, but just works
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
                                        ),  # Convert to (T, C, H, W) format
                                        fps=30,
                                        format="gif",
                                    )
                                },
                                step=global_step,
                            )
                if args.use_wandb:
                    # Log explr_weight details
                    if "explr_weight" in logs_dict:
                        ew = logs_dict["explr_weight"]
                        # Log histogram to see distribution (are some dims getting 0 weight?)
                        logs["explr_weight_hist"] = wandb.Histogram(ew.cpu().numpy())
                        
                        # Log mean weight to see if it collapses or stays balanced
                        logs["explr_weight_mean"] = ew.mean()
                        
                        # Optional: Log the weights as a heatmap (if needed to track specific dims)
                        # We can log a generated image or just the raw values.
                        # For 61 dims, a Heatmap/Image is good.
                        # Reshape to (1, 61) to make it an image strip
                        logs["explr_weight_heatmap"] = wandb.Image(
                            ew.mean(0, keepdim=True).cpu().numpy(), 
                            caption="Exploration Weights (Avg over batch)"
                        )
                        
                        # --- Grow Time-Series Heatmap ---
                        if "explr_weights_history" not in locals():
                            explr_weights_history = []
                            explr_steps_history = []
                            
                        # Append current mean weights: Shape (61,)
                        current_mean_weights = ew.mean(0).detach().cpu().numpy()
                        explr_weights_history.append(current_mean_weights)
                        explr_steps_history.append(global_step)
                        
                        # Generate Heatmap every N logs (e.g., every 5th log to save compute)
                        # or just always if it's fast. 61x(T) is small.
                        if len(explr_weights_history) > 0:
                            history_array = np.array(explr_weights_history) # [T, 61]
                            
                            # Create plot
                            plt.figure(figsize=(12, 8))
                            # Transpose: Y=Joints, X=Step
                            # Use subsampling if T is too large?
                            sns.heatmap(history_array.T, cmap="viridis", cbar=True) 
                            
                            # Decorate
                            try:
                                joint_names = get_joint_names()
                                if len(joint_names) == history_array.shape[1]:
                                    plt.yticks(
                                        ticks=np.arange(len(joint_names)) + 0.5, 
                                        labels=joint_names, 
                                        fontsize=6, 
                                        rotation=0
                                    )
                            except:
                                pass # formatting error usually

                            # Set custom xticks to show actual global_step values
                            num_ticks = min(10, len(explr_steps_history))
                            if num_ticks > 0:
                                tick_indices = np.linspace(0, len(explr_steps_history) - 1, num_ticks, dtype=int)
                                tick_labels = [explr_steps_history[i] for i in tick_indices]
                                plt.xticks(ticks=tick_indices + 0.5, labels=tick_labels, rotation=45, fontsize=8)
                                
                            plt.title("Exploration Weights Evolution")
                            plt.xlabel("Train Steps")
                            
                            plt.tight_layout()
                            
                            # Log to WandB
                            logs["explr_history_heatmap"] = wandb.Image(plt)
                            plt.close() # Clean up memory
                        # --------------------------------
                        
                    wandb.log(
                        {
                            "speed": speed,
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
