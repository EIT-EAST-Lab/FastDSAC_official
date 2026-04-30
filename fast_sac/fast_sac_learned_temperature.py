import torch
import torch.nn as nn

try:
    from .fast_sac import DSACT_EnhancedActor_explr as BaseEnhancedExplorationActor
except ImportError:
    from fast_sac import DSACT_EnhancedActor_explr as BaseEnhancedExplorationActor


def _inverse_sigmoid(value: torch.Tensor, eps: float) -> torch.Tensor:
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


class LearnedTemperatureDSACTEnhancedActorExplr(BaseEnhancedExplorationActor):
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
        scale_max: float = 2.0,
        temperature: float = 1.0,
        temperature_min: float = 0.25,
        temperature_max: float = 10.0,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True,
        device: torch.device | None = None,
        eps: float = 1e-6,
    ):
        super().__init__(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            init_scale=init_scale,
            hidden_dim=hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            scale_min=scale_min,
            scale_max=scale_max,
            activation=activation,
            use_layer_norm=use_layer_norm,
            temperature=temperature,
            device=device,
        )
        if temperature_min <= 0.0:
            raise ValueError("temperature_min must be positive")
        if temperature_max <= temperature_min:
            raise ValueError("temperature_max must be greater than temperature_min")

        init_temperature = float(min(max(temperature, temperature_min), temperature_max))
        span = temperature_max - temperature_min
        normalized_temperature = torch.tensor(
            [(init_temperature - temperature_min) / span],
            device=device,
            dtype=torch.float32,
        )

        self.raw_temperature = nn.Parameter(
            _inverse_sigmoid(normalized_temperature, eps=eps)
        )
        self.register_buffer(
            "temperature_min",
            torch.tensor([temperature_min], device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "temperature_max",
            torch.tensor([temperature_max], device=device, dtype=torch.float32),
        )
        self._temperature_eps = eps

    def get_temperature(self) -> torch.Tensor:
        span = self.temperature_max - self.temperature_min
        return self.temperature_min + span * torch.sigmoid(self.raw_temperature)

    def model_parameters(self) -> list[nn.Parameter]:
        temperature_param_ids = {id(self.raw_temperature)}
        return [
            param for param in self.parameters() if id(param) not in temperature_param_ids
        ]

    def temperature_parameters(self) -> list[nn.Parameter]:
        return [self.raw_temperature]

    def _compute_explr_weights(
        self,
        explr_weight_logits: torch.Tensor,
        scale_multiplier: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = self.get_temperature().clamp_min(self._temperature_eps)
        scaled_logits = torch.clamp(explr_weight_logits / temperature, -20.0, 20.0)
        if scale_multiplier is not None:
            scaled_logits = torch.clamp(scaled_logits * scale_multiplier, -20.0, 20.0)
        explr_probs = torch.softmax(scaled_logits, dim=-1)
        return explr_probs * self.n_act, explr_probs

    def forward(
        self, obs: torch.Tensor, return_explr_weight: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean, log_std, explr_weight_logits = torch.chunk(self.fc_out(x), 3, dim=-1)
        explr_weight, _ = self._compute_explr_weights(explr_weight_logits)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        std = explr_weight * log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        if return_explr_weight:
            return action, log_prob, mean, explr_weight
        return action, log_prob, mean

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dones is not None and dones.sum() > 0:
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.scale_max - self.scale_min)
                + self.scale_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.expl_scales.copy_(
                torch.where(dones_view, new_scales, self.expl_scales)
            )

        x = self.net(obs)
        mean, log_std, explr_weight_logits = torch.chunk(self.fc_out(x), 3, dim=-1)
        explr_weight, _ = self._compute_explr_weights(
            explr_weight_logits,
            scale_multiplier=self.expl_scales,
        )
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        std = explr_weight * log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
