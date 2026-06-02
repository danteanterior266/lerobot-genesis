"""Shared fakes for the CPU test suite — a scripted scene + sink behind the public Protocols.

Kept out of the package: these stand in for Genesis and LeRobot so the gym contract and the
recording loop are verified without a GPU or either upstream installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FakeDriver:
    """A scripted scene that succeeds on the Nth applied action."""

    action_dim = 6
    state_dim = 6
    image_shape = (8, 12, 3)

    def __init__(self, succeed_after: int | None = None) -> None:
        self._succeed_after = succeed_after
        self._applied = 0
        self.last_action: np.ndarray | None = None

    def reset(self) -> None:
        self._applied = 0

    def apply_action(self, action: np.ndarray) -> None:
        self.last_action = action
        self._applied += 1

    def step(self) -> None:
        pass

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((8, 12, 3), np.uint8), np.arange(6, dtype=np.float32)

    def is_success(self) -> bool:
        return self._succeed_after is not None and self._applied >= self._succeed_after


class FakeSink:
    """Records what the recording loop hands it, for assertions."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.episodes = 0
        self.finalized = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.episodes += 1

    def finalize(self) -> None:
        self.finalized += 1
