import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        return self.net(x)

class DistQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        # NOTE: use GELU instead of ReLU, same as DSAC
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim // 4, 2, device=device),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        value_mean, value_std = torch.chunk(x, chunks=2, dim=-1)
        value_std = torch.nn.functional.softplus(value_std) # avoid 0
        
        return torch.cat((value_mean, value_std), dim=-1)

class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = QNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = QNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            device=device,
        )

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.qnet1(obs, actions)
        q2 = self.qnet2(obs, actions)
        return q1, q2

class GaussianDistCritic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.qnet2 = DistQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            device=device,
        )

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        StochaQ1 = self.qnet1(obs, actions)
        StochaQ2 = self.qnet2(obs, actions)
        return StochaQ1, StochaQ2

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, n_act, device=device)
        nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        # nn.init.constant_(self.fc_logstd.bias, 0.5)

        self.n_envs = num_envs

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean


# DSAC_LOG_STD_MAX = 2
# DSAC_LOG_STD_MIN = -20  # Aligned with official JAX implementation
DSAC_LOG_STD_MAX = 0
DSAC_LOG_STD_MIN = -5


class DSACT_Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            activation,
        )

        # Official JAX implementation uses a single layer split into mean and log_std
        self.fc_out = nn.Linear(hidden_dim // 4, n_act * 2, device=device)
        
        # Initialize weights (optional, to match JAX/Haiku defaults if needed, but keeping PyTorch defaults is usually fine)
        # However, we should handle the init_scale for the mean part if we want to preserve previous logic
        # But since it's a single layer now, we might just init the whole thing or split init.
        # Let's try to preserve the specific init for mean if possible, or just use standard init.
        # The previous code used:
        # nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        # nn.init.constant_(self.fc_mu.bias, 0.0)
        # To replicate this on a shared layer is tricky without splitting weights manually.
        # For simplicity and alignment, let's use standard init for now, or apply to chunks.
        
        # Apply specific init to the mean part of the weights?
        # JAX implementation usually uses orthogonal or uniform. 
        # Let's stick to default Linear init for now unless performance degrades.
        
        self.n_envs = num_envs

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs
        x = self.net(x)
        output = self.fc_out(x)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        
        # JAX implementation uses clip (clamp) instead of tanh squashing
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        # log_std = torch.clamp(log_std, DSAC_LOG_STD_MIN, DSAC_LOG_STD_MAX)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean


class DSACT_Actor_explr(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        temperature: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            activation,
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            activation,
        )

        # Official JAX implementation uses a single layer split into mean and log_std
        self.fc_out = nn.Linear(hidden_dim // 4, n_act * 3, device=device)
        
        # Initialize weights (optional, to match JAX/Haiku defaults if needed, but keeping PyTorch defaults is usually fine)
        # However, we should handle the init_scale for the mean part if we want to preserve previous logic
        # But since it's a single layer now, we might just init the whole thing or split init.
        # Let's try to preserve the specific init for mean if possible, or just use standard init.
        # The previous code used:
        # nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        # nn.init.constant_(self.fc_mu.bias, 0.0)
        # To replicate this on a shared layer is tricky without splitting weights manually.
        # For simplicity and alignment, let's use standard init for now, or apply to chunks.
        
        # Apply specific init to the mean part of the weights?
        # JAX implementation usually uses orthogonal or uniform. 
        # Let's stick to default Linear init for now unless performance degrades.
        
        self.n_envs = num_envs
        self.temperature = temperature

    def forward(
        self, obs: torch.Tensor, return_explr_weight: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs
        x = self.net(x)
        output = self.fc_out(x)
        mean, log_std, explr_weight = torch.chunk(output, 3, dim=-1)

        explr_weight_logits = explr_weight / self.temperature
        explr_weight_logits = torch.clamp(explr_weight_logits, -20, 20)
        explr_weight = torch.softmax(explr_weight_logits, dim=-1) * self.n_act

        # JAX implementation uses clip (clamp) instead of tanh squashing
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        # log_std = torch.clamp(log_std, DSAC_LOG_STD_MIN, DSAC_LOG_STD_MAX)

        std = explr_weight * log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True) # NOTE: 标准求和，按照权重相加，本身就是互相约束的啊
        mean = torch.tanh(mean)

        if return_explr_weight:
            return action, log_prob, mean, explr_weight

        return action, log_prob, mean



class DSACT_EnhancedActor(DSACT_Actor):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        log_std_max: float = 0.5,
        log_std_min: float = -20.0,
        scale_min: float = 0.5,
        scale_max: float = 2,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        device: torch.device = None,
    ):
        super().__init__(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            init_scale=init_scale,
            hidden_dim=hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            activation=activation,
            use_layer_norm=use_layer_norm,
            device=device,
        )

        self.register_buffer("expl_scales", 
            torch.rand(num_envs, 1, device=device) * (scale_max - scale_min) + scale_min
        )
        self.register_buffer("scale_min", torch.as_tensor(scale_min, device=device))
        self.register_buffer("scale_max", torch.as_tensor(scale_max, device=device))


    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.scale_max - self.scale_min)
                + self.scale_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.expl_scales.copy_(
                torch.where(dones_view, new_scales, self.expl_scales)
            )

        x = obs
        x = self.net(x)
        mean, log_std = torch.chunk(self.fc_out(x), 2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp() * self.expl_scales
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean



class DSACT_EnhancedActor_explr(DSACT_Actor_explr):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        log_std_max: float = 0.5,
        log_std_min: float = -20.0,
        scale_min: float = 0.5,
        scale_max: float = 2,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        temperature: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            init_scale=init_scale,
            hidden_dim=hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            activation=activation,
            use_layer_norm=use_layer_norm,
            temperature=temperature,
            device=device,
        )

        self.register_buffer("expl_scales", 
            torch.rand(num_envs, 1, device=device) * (scale_max - scale_min) + scale_min
        )
        self.register_buffer("scale_min", torch.as_tensor(scale_min, device=device))
        self.register_buffer("scale_max", torch.as_tensor(scale_max, device=device))


    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.scale_max - self.scale_min)
                + self.scale_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.expl_scales.copy_(
                torch.where(dones_view, new_scales, self.expl_scales)
            )

        x = obs
        x = self.net(x)
        mean, log_std, explr_weight = torch.chunk(self.fc_out(x), 3, dim=-1)
        explr_weight_logits = explr_weight / self.temperature
        explr_weight_logits = torch.clamp(explr_weight_logits, -20, 20)
        explr_weight = torch.softmax(explr_weight_logits * self.expl_scales, dim=-1) * self.n_act
        # explr_weight = torch.softmax(explr_weight_logits, dim=-1) * self.n_act * self.expl_scales
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = explr_weight * log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean


# FastSAC Pro Implementation

class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x  # logits

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class FastSACProCritic(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = num_q_networks
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        self.setup_qnetworks()

        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_qnetworks_with_obs_dim(n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    num_atoms=self.num_atoms,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    hidden_dim=self.hidden_dim,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.process_obs(obs)
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        x = self.process_obs(obs)
        projections = [
            qnet.projection(
                x,
                actions,
                rewards,
                bootstrap,
                discount,
                self.q_support,
                self.q_support.device,
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=-1)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )

    # log_std_max: float = 0.0
    # """the maximum value of the log std FastSAC Pro"""

    # log_std_min: float = -5.0
    # """the minimum value of the log std FastSAC Pro"""


class FastSACProActor(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape

        # Setup the network - this will be overridden in subclasses if needed
        self.setup_network()

        # Register action scaling parameters as buffers
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(n_act, device=device))

        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(n_act, device=device))

    def setup_network(self) -> None:
        """Setup the network architecture. Can be overridden by subclasses."""
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_network_with_input_dim(n_obs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        """Setup network with specific input dimension."""
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=self.device),
            nn.LayerNorm(self.hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.LayerNorm(self.hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
            nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
        )
        self.fc_logstd = nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)
    
    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.process_obs(obs)
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            # Apply tanh to get bounded actions in [-1, 1]
            tanh_action = torch.tanh(raw_action)
            # Scale and bias to get final actions
            action = tanh_action * self.action_scale + self.action_bias

            # Compute log probability with proper Jacobian correction
            log_prob = dist.log_prob(raw_action)
            # Jacobian correction for tanh transformation
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            # Jacobian correction for scaling transformation
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            # Non-tanh case
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        log_prob = log_prob.sum(1)
        return action, log_prob

    @torch.no_grad()
    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False
    ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                tanh_mean = torch.tanh(mean)
                return tanh_mean * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
        else:
            action = raw_action

        return action