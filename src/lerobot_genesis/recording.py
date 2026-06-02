"""Record :class:`GenesisEnv` rollouts as LeRobot datasets.

A ``Policy`` is any ``observation -> action`` callable; an :class:`EpisodeSink` is any
add/save/finalize target. :class:`LeRobotDatasetSink` is the real sink over ``LeRobotDataset``;
tests drive a fake one, so the recording loop needs neither LeRobot nor a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .env import GenesisEnv

Policy = Callable[[dict[str, Any]], np.ndarray]


@runtime_checkable
class EpisodeSink(Protocol):
    """Where recorded frames go. The real impl wraps ``LeRobotDataset``; tests use a fake."""

    def add_frame(self, frame: dict[str, Any]) -> None: ...
    def save_episode(self) -> None: ...
    def finalize(self) -> None: ...


def make_frame(obs: dict[str, Any], action: np.ndarray, *, camera: str = "front") -> dict[str, Any]:
    """One LeRobot frame from an observation and the action taken, in convention keys."""
    return {
        f"observation.images.{camera}": np.asarray(obs["pixels"], dtype=np.uint8),
        "observation.state": np.asarray(obs["agent_pos"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
    }


def record_episodes(
    env: GenesisEnv,
    policy: Policy,
    sink: EpisodeSink,
    *,
    n_episodes: int,
    camera: str = "front",
) -> int:
    """Roll ``policy`` in ``env`` for ``n_episodes``, recording each step; return the frame count.

    ``finalize`` is called once at the end — it is mandatory for ``LeRobotDataset`` or the written
    parquet is left incomplete.
    """
    frames = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = np.asarray(policy(obs), dtype=np.float32)
            sink.add_frame(make_frame(obs, action, camera=camera))
            frames += 1
            obs, _, terminated, truncated, _ = env.step(action)
        sink.save_episode()
    sink.finalize()
    return frames


class LeRobotDatasetSink:
    """An :class:`EpisodeSink` backed by ``LeRobotDataset`` — one sink writes one dataset.

    ``lerobot`` is imported lazily, so importing this module never requires it. Frames follow the
    canonical ``{dtype, shape, names}`` feature schema and the convention keys
    ``observation.images.<camera>`` / ``observation.state`` / ``action``.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        fps: int,
        state_dim: int,
        action_dim: int,
        image_shape: tuple[int, int, int],
        camera: str = "front",
        task: str = "",
        root: str | Path | None = None,
        use_videos: bool = True,
        robot_type: str = "genesis",
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        h, w, _ = image_shape
        self._task = task
        features = {
            f"observation.images.{camera}": {
                "dtype": "video" if use_videos else "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        }
        self._dataset = LeRobotDataset.create(
            repo_id,
            fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
        )

    def add_frame(self, frame: dict[str, Any]) -> None:
        # LeRobot's v3 add_frame requires a per-frame task string; supply it from the sink so the
        # generic make_frame stays format-agnostic.
        self._dataset.add_frame({**frame, "task": frame.get("task", self._task)})

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def finalize(self) -> None:
        self._dataset.finalize()

    @property
    def dataset(self) -> Any:
        """The underlying ``LeRobotDataset`` (e.g. for ``push_to_hub`` or inspection)."""
        return self._dataset
