"""Actuator topology groups used by grouped latent-action models."""

from typing import Dict, List, Optional, Sequence


ACTION_GROUPS: Dict[str, List[List[int]]] = {
    "cheetah": [[0, 1, 2], [3, 4, 5]],
    # DM Control Hopper actuator order: waist, hip, knee, ankle.
    "hopper": [[0], [1, 2, 3]],
    "ant": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "humanoid": [
        [0, 1, 2],
        list(range(3, 9)),
        list(range(9, 15)),
        list(range(15, 18)),
        list(range(18, 21)),
    ],
}

_ACTION_DIM_TO_ENV = {6: "cheetah", 4: "hopper", 8: "ant", 21: "humanoid"}


def _canonical_env_name(env_name: str) -> str:
    """Return the supported domain name from a task or environment name."""
    normalized = env_name.lower().replace("_", "-")
    for name in ACTION_GROUPS:
        if name in normalized:
            return name
    raise ValueError(
        f"unsupported environment {env_name!r}; expected one of "
        f"{sorted(ACTION_GROUPS)}"
    )


def validate_action_groups(
    groups: Sequence[Sequence[int]], action_dim: int
) -> List[List[int]]:
    """Validate that groups form an exact, non-overlapping action partition."""
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    normalized = [list(group) for group in groups]
    if not normalized or any(not group for group in normalized):
        raise ValueError("action groups must be non-empty")
    flattened = [index for group in normalized for index in group]
    if sorted(flattened) != list(range(action_dim)):
        raise ValueError(
            "action groups must contain every action index exactly once; "
            f"got {flattened} for action_dim={action_dim}"
        )
    return normalized


def get_action_groups(
    action_dim: int,
    env_name: Optional[str] = None,
    groups: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[int]]:
    """Resolve actuator groups from an override, environment name, or dimension.

    Dimension-based resolution keeps the class compatible with the current PWM
    construction path, which injects only ``state_dim`` and ``action_dim``.
    """
    if groups is not None:
        return validate_action_groups(groups, action_dim)

    if env_name is None:
        try:
            name = _ACTION_DIM_TO_ENV[action_dim]
        except KeyError as error:
            raise ValueError(
                "env_name (or explicit groups) is required for action_dim="
                f"{action_dim}; known dimensions are {sorted(_ACTION_DIM_TO_ENV)}"
            ) from error
    else:
        name = _canonical_env_name(env_name)

    return validate_action_groups(ACTION_GROUPS[name], action_dim)
