import pytest
import torch

from pwm.models.action_groups import get_action_groups
from pwm.models.grouped_jsae import GroupedJSAE


@pytest.mark.parametrize(
    ("env_name", "action_dim", "expected"),
    [
        ("cheetah-run-backwards", 6, [[0, 1, 2], [3, 4, 5]]),
        ("hopper-hop", 4, [[0], [1, 2, 3]]),
        ("ant", 8, [[0, 1], [2, 3], [4, 5], [6, 7]]),
    ],
)
def test_action_groups(env_name, action_dim, expected):
    assert get_action_groups(action_dim, env_name) == expected


def test_grouped_jsae_shapes_and_bounds():
    model = GroupedJSAE(
        state_dim=12,
        action_dim=6,
        latent_action_dim=7,
        units=[16, 8],
        activation_class="nn.Mish",
    )
    state = torch.randn(3, 5, 12)
    action = torch.randn(3, 5, 6)

    reconstructed, latent = model(state, action)

    assert reconstructed.shape == action.shape
    assert latent.shape == (3, 5, 7)
    assert model.group_latent_dims == [4, 3]
    assert torch.all(reconstructed.abs() <= 1)
    assert torch.all(latent.abs() <= 1)


def test_each_decoder_only_depends_on_its_latent_group():
    model = GroupedJSAE(5, 6, 6, [8])
    state = torch.randn(2, 5)
    latent = torch.randn(2, 6)
    changed = latent.clone()
    changed[..., :3] += 1.0

    original_action = model.decode(state, latent)
    changed_action = model.decode(state, changed)

    assert not torch.allclose(original_action[..., :3], changed_action[..., :3])
    assert torch.allclose(original_action[..., 3:], changed_action[..., 3:])


def test_custom_groups_must_partition_actions():
    with pytest.raises(ValueError, match="every action index exactly once"):
        GroupedJSAE(5, 4, 4, [8], groups=[[0, 1], [1, 2]])
