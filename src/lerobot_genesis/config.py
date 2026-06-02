"""Register Genesis with LeRobot as ``--env.type=genesis``.

This is the eval/RL seam: a ``@EnvConfig.register_subclass`` so LeRobot's CLI can discover a Genesis
environment and map its observation keys onto the policy convention. It mirrors LeRobot's own env
configs (e.g. ``AlohaEnv``). Unlike the rest of the package this module imports ``lerobot`` at load
time — the registration decorator needs it — so it is imported on demand, not from the package root.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import ACTION, OBS_IMAGE, OBS_STATE, EnvConfig

_CAMERA = "front"


@EnvConfig.register_subclass("genesis")
@dataclass
class GenesisEnvConfig(EnvConfig):
    """LeRobot env config for a Genesis scene.

    ``GenesisEnv`` emits ``{"pixels", "agent_pos"}``; ``features_map`` routes those onto LeRobot's
    convention keys and ``features`` declares their typed shapes. ``action_dim`` / ``state_dim`` and
    the camera resolution default to a 9-DOF Franka but are meant to be overridden per robot.
    """

    task: str | None = ""
    fps: int = 30
    episode_length: int = 300
    action_dim: int = 9
    state_dim: int = 9
    observation_height: int = 240
    observation_width: int = 320
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features:
            self.features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
                "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
                "pixels": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(self.observation_height, self.observation_width, 3),
                ),
            }
        if not self.features_map:
            self.features_map = {
                "action": ACTION,
                "agent_pos": OBS_STATE,
                "pixels": f"{OBS_IMAGE}.{_CAMERA}",
            }

    @property
    def gym_kwargs(self) -> dict[str, object]:
        return {
            "task": self.task,
            "max_episode_steps": self.episode_length,
            "render_mode": self.render_mode,
        }
