"""Env registration — runs only where ``lerobot`` is installed (skipped on the CPU-only matrix)."""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")


def test_genesis_registers_as_an_env_type() -> None:
    from lerobot.envs.configs import EnvConfig

    from lerobot_genesis.config import GenesisEnvConfig  # registers on import

    assert "genesis" in EnvConfig.get_known_choices()

    cfg = GenesisEnvConfig(action_dim=9, state_dim=9)
    assert set(cfg.features) == {"action", "agent_pos", "pixels"}
    assert cfg.features_map["agent_pos"] == "observation.state"
    assert cfg.gym_kwargs["max_episode_steps"] == cfg.episode_length
