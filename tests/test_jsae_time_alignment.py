import os, sys
import torch

# ensure repo root is on sys.path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add src directory where `pwm` package lives
sys.path.insert(0, os.path.join(repo_root, 'src'))

from pwm.models.actor import JointStateActionAutoEncoder


def test_jsae_alignment():
    state_dim = 5
    action_dim = 3
    latent_action_dim = 2
    units = [8, 4]

    jsae = JointStateActionAutoEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_action_dim=latent_action_dim,
        units=units,
    )

    # create time x batch x feat tensors
    L = 6
    B = 2
    states = torch.randn(L, B, state_dim)
    actions = torch.randn(L, B, action_dim)

    # should work: encode/decode forward
    rec, lat = jsae(states, actions)
    assert rec.shape == actions.shape, f"reconstructed shape {rec.shape} != {actions.shape}"
    assert lat.shape[:-1] == states.shape[:-1]

    # now create mismatch in time dim and ensure error is raised
    bad_actions = torch.randn(L + 1, B, action_dim)
    try:
        jsae.encode(states, bad_actions)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched time dims but none raised")


if __name__ == '__main__':
    test_jsae_alignment()
    print('test_jsae_time_alignment passed')
