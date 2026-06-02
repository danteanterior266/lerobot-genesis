"""The recording loop, verified with a fake env + fake sink — no LeRobot, no GPU."""

from __future__ import annotations

from typing import Any

import numpy as np
from tests.test_env import FakeDriver

from lerobot_genesis import EpisodeSink, GenesisEnv, make_frame, record_episodes


class FakeSink:
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


def test_fake_sink_satisfies_the_protocol() -> None:
    assert isinstance(FakeSink(), EpisodeSink)


def test_make_frame_uses_convention_keys() -> None:
    obs = {"pixels": np.zeros((4, 4, 3), np.uint8), "agent_pos": np.zeros(6, np.float32)}
    frame = make_frame(obs, np.ones(6, np.float32), camera="front")
    assert set(frame) == {"observation.images.front", "observation.state", "action"}
    assert frame["observation.images.front"].dtype == np.uint8
    assert frame["action"].dtype == np.float32


def test_records_frames_episodes_and_one_finalize() -> None:
    env = GenesisEnv(FakeDriver(succeed_after=3), max_episode_steps=10)
    sink = FakeSink()

    frames = record_episodes(env, lambda _obs: np.zeros(6, np.float32), sink, n_episodes=2)

    assert frames == len(sink.frames) == 6  # 3 steps/episode (terminates on success), 2 episodes
    assert sink.episodes == 2
    assert sink.finalized == 1  # exactly once, at the very end
    assert all("observation.state" in frame for frame in sink.frames)
