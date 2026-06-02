"""lerobot-genesis — the bridge between the Genesis simulator and Hugging Face LeRobot.

Expose a Genesis scene as a ``gymnasium.Env``, record rollouts as a ``LeRobotDataset``, and (from
v0.2) evaluate LeRobot policies in Genesis. The simulator-bound code sits behind the ``SceneDriver``
seam, so the gym contract and recorder import and test without a GPU or either upstream installed.

``GenesisEnvConfig`` is intentionally not re-exported here — it imports ``lerobot`` at load time;
import it explicitly via ``from lerobot_genesis.config import GenesisEnvConfig``.
"""

from __future__ import annotations

from .driver import GenesisRobotDriver, bundled_franka_mjcf
from .env import GenesisEnv, SceneDriver
from .recording import EpisodeSink, LeRobotDatasetSink, Policy, make_frame, record_episodes

__version__ = "0.1.0"

__all__ = [
    "EpisodeSink",
    "GenesisEnv",
    "GenesisRobotDriver",
    "LeRobotDatasetSink",
    "Policy",
    "SceneDriver",
    "__version__",
    "bundled_franka_mjcf",
    "make_frame",
    "record_episodes",
]
