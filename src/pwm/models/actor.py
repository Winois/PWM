from typing import Optional, Dict, Any, Union, Sequence, Type, List
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from pwm.models import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        #latent_action_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 2.0**0.5,
    ):
        super(ActorDeterministicMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]
        #self.layer_dims = [obs_dim] + units + [latent_action_dim]

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class

        init_ = lambda m: model_utils.init(
            m,
            lambda x: nn.init.orthogonal_(x, init_gain),
            lambda x: nn.init.constant_(x, 0),
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.actor = nn.Sequential(*modules)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def forward(self, observations, deterministic=False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module): # 随机
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        #latent_action_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 1.0,
        init_logstd: float = -1.0,
        min_logstd: float = -10.0,
    ):
        super(ActorStochasticMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]
        #self.layer_dims = [obs_dim] + units + [latent_action_dim]

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(nn.Identity())

        self.mu_net = nn.Sequential(*modules)

        self.logstd = torch.nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32) * init_logstd
            #torch.ones(latent_action_dim, dtype=torch.float32) * init_logstd
        )

        self.action_dim = action_dim
        #self.latent_action_dim = latent_action_dim
        self.obs_dim = obs_dim
        self.min_logstd = min_logstd

        for param in self.parameters():
            param.data *= init_gain

    def get_logstd(self):
        return self.logstd

    def clamp_std(self):
        self.logstd.data = torch.clamp(self.logstd.data, self.min_logstd)

    def forward(self, obs, deterministic=False):
        self.clamp_std()
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp()
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample

    def action_log_probs(self, obs):
        self.clamp_std()
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)
        sample = dist.rsample()

        return sample, dist.log_prob(sample)

    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)


class JointStateActionAutoEncoder(nn.Module):
    """State-conditioned action autoencoder used to learn latent actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_action_dim: int,
        units: List[int],
        activation_class: Type = nn.Mish,
    ):
        super().__init__()
        if latent_action_dim <= 0:
            raise ValueError("latent_action_dim must be positive")

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_action_dim = latent_action_dim
        self.encoder = self._build_mlp(
            state_dim + action_dim,
            units,
            latent_action_dim,
            activation_class,
        )
        self.decoder = self._build_mlp(
            state_dim + latent_action_dim,
            list(reversed(units)),
            action_dim,
            activation_class,
        )

    @staticmethod
    def _build_mlp(input_dim, units, output_dim, activation_class):
        dims = [input_dim] + list(units) + [output_dim]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(activation_class())
                layers.append(nn.LayerNorm(dims[index + 1]))
        return nn.Sequential(*layers)

    def encode(self, state, action):
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "state and action must have identical batch/time dimensions"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state feature dim {self.state_dim}")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"expected action feature dim {self.action_dim}")
        return torch.tanh(self.encoder(torch.cat((state, action), dim=-1)))

    def decode(self, state, latent_action):
        if state.shape[:-1] != latent_action.shape[:-1]:
            raise ValueError(
                "state and latent_action must have identical batch/time dimensions"
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state feature dim {self.state_dim}")
        if latent_action.shape[-1] != self.latent_action_dim:
            raise ValueError(
                f"expected latent action feature dim {self.latent_action_dim}"
            )
        return torch.tanh(
            self.decoder(torch.cat((state, latent_action), dim=-1))
        )

    def forward(self, state, action):
        latent_action = self.encode(state, action)
        reconstructed_action = self.decode(state, latent_action)
        return reconstructed_action, latent_action
