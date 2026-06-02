"""A real Genesis smoke test for the reference driver — needs a GPU + genesis-world.

Marked ``gpu`` and skipped unless ``genesis`` imports, so it never runs on the CPU-only CI matrix.
This is the one place the driver touches a real simulator; the rest of the suite uses the seam.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("genesis")

pytestmark = pytest.mark.gpu


def test_franka_driver_steps_and_observes() -> None:
    from lerobot_genesis import GenesisRobotDriver, bundled_franka_mjcf

    driver = GenesisRobotDriver(robot=bundled_franka_mjcf(), steps_per_action=2)
    assert driver.action_dim == driver.state_dim == 9
    assert driver.image_shape == (240, 320, 3)

    driver.reset()
    driver.apply_action(np.zeros(driver.action_dim, np.float32))
    driver.step()
    rgb, state = driver.observe()
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    assert state.shape == (9,)
