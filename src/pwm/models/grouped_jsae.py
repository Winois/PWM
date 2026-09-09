"""State-conditioned autoencoder with topology-local latent action groups."""

from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn

from pwm.models.action_groups import get_action_groups


class GroupedJSAE(nn.Module):
    """Encode and decode actuator groups independently.

    Every group sees the full state but only its own action/latent coordinates.
    This preserves state conditioning while preventing direct coupling between
    unrelated actuator groups in the action autoencoder.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_action_dim: int,
        units: List[int],
        activation_class: Type = nn.Mish,
        env_name: Optional[str] = None,
        groups: Optional[Sequence[Sequence[int]]] = None,
    ):
        super().__init__()
        if latent_action_dim <= 0:
            raise ValueError("latent_action_dim must be positive")

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_action_dim = latent_action_dim
        self.action_groups = get_action_groups(action_dim, env_name, groups)
        if latent_action_dim < len(self.action_groups):
            raise ValueError(
                "latent_action_dim must be at least the number of action groups "
                f"({len(self.action_groups)})"
            )

        self.group_latent_dims = self._allocate_latent_dims(
            latent_action_dim, len(self.action_groups)
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_units = list(reversed(units))
        for action_group, group_latent_dim in zip(
            self.action_groups, self.group_latent_dims
        ):
            self.encoders.append(
                self._build_mlp(
                    state_dim + len(action_group),
                    units,
                    group_latent_dim,
                    activation_class,
                )
            )
            self.decoders.append(
                self._build_mlp(
                    state_dim + group_latent_dim,
                    decoder_units,
                    len(action_group),
                    activation_class,
                )
            )

    @staticmethod
    def _allocate_latent_dims(total_dim: int, group_count: int) -> List[int]:
        base, remainder = divmod(total_dim, group_count)
        return [base + (index < remainder) for index in range(group_count)]

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

    def _check_state(self, state):
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state feature dim {self.state_dim}")

    def encode(self, state, action):
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "state and action must have identical batch/time dimensions"
            )
        self._check_state(state)
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"expected action feature dim {self.action_dim}")

        latent_groups = []
        for encoder, indices in zip(self.encoders, self.action_groups):
            group_action = action[..., indices]
            latent_groups.append(
                torch.tanh(encoder(torch.cat((state, group_action), dim=-1)))
            )
        return torch.cat(latent_groups, dim=-1)

    def decode(self, state, latent_action):
        if state.shape[:-1] != latent_action.shape[:-1]:
            raise ValueError(
                "state and latent_action must have identical batch/time dimensions"
            )
        self._check_state(state)
        if latent_action.shape[-1] != self.latent_action_dim:
            raise ValueError(
                f"expected latent action feature dim {self.latent_action_dim}"
            )

        reconstructed = latent_action.new_empty(
            *latent_action.shape[:-1], self.action_dim
        )
        latent_groups = torch.split(latent_action, self.group_latent_dims, dim=-1)
        for decoder, indices, group_latent in zip(
            self.decoders, self.action_groups, latent_groups
        ):
            group_action = torch.tanh(
                decoder(torch.cat((state, group_latent), dim=-1))
            )
            reconstructed[..., indices] = group_action
        return reconstructed

    def forward(self, state, action):
        latent_action = self.encode(state, action)
        reconstructed_action = self.decode(state, latent_action)
        return reconstructed_action, latent_action
