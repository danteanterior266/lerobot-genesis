"""The Gymnasium contract, verified with a fake driver — no Genesis, no GPU."""

from __future__ import annotations

import numpy as np

from lerobot_genesis import GenesisEnv, SceneDriver


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


def test_fake_driver_satisfies_the_protocol() -> None:
    assert isinstance(FakeDriver(), SceneDriver)


def test_reset_returns_observation_and_unsuccessful_info() -> None:
    obs, info = GenesisEnv(FakeDriver()).reset(seed=0)
    assert info == {"is_success": False}
    assert set(obs) == {"pixels", "agent_pos"}
    assert obs["pixels"].shape == (8, 12, 3) and obs["pixels"].dtype == np.uint8
    assert obs["agent_pos"].shape == (6,) and obs["agent_pos"].dtype == np.float32


def test_spaces_match_the_driver() -> None:
    env = GenesisEnv(FakeDriver())
    assert env.action_space.shape == (6,)
    assert env.observation_space["pixels"].shape == (8, 12, 3)
    assert env.observation_space.contains(env.reset()[0])


def test_step_returns_the_gymnasium_five_tuple() -> None:
    env = GenesisEnv(FakeDriver())
    env.reset()
    _obs, reward, terminated, truncated, info = env.step(np.zeros(6, np.float32))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "is_success" in info


def test_terminates_on_success_with_reward() -> None:
    env = GenesisEnv(FakeDriver(succeed_after=2))
    env.reset()
    _, _, terminated, _, _ = env.step(np.zeros(6, np.float32))
    assert not terminated
    _, reward, terminated, _, info = env.step(np.zeros(6, np.float32))
    assert terminated and info["is_success"] and reward == 1.0


def test_truncates_at_max_episode_steps() -> None:
    env = GenesisEnv(FakeDriver(), max_episode_steps=3)
    env.reset()
    truncations = [env.step(np.zeros(6, np.float32))[3] for _ in range(3)]
    assert truncations == [False, False, True]


def test_action_is_forwarded_to_the_driver() -> None:
    driver = FakeDriver()
    env = GenesisEnv(driver)
    env.reset()
    env.step(np.ones(6, np.float32))
    assert np.allclose(driver.last_action, 1.0)


def test_render_returns_the_frame() -> None:
    env = GenesisEnv(FakeDriver())
    env.reset()
    assert env.render().shape == (8, 12, 3)


def test_unknown_render_mode_rejected() -> None:
    try:
        GenesisEnv(FakeDriver(), render_mode="human")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unsupported render_mode")
