"""Record a Franka arm reaching a target pose, as a LeRobotDataset — a complete end-to-end example.

Run on a machine with a CUDA GPU and ``pip install "lerobot-genesis[all]"``:

    python examples/record_franka.py

It builds a Franka in a Genesis scene, drives it toward a goal pose with a scripted policy, records
two episodes, and reloads the dataset to confirm it is valid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot_genesis import (
    GenesisEnv,
    GenesisRobotDriver,
    LeRobotDatasetSink,
    bundled_franka_mjcf,
    record_episodes,
)

ROOT = Path("./outputs/franka_reach")
REPO_ID = "example/franka_reach"
CAMERA = "front"
TASK = "move the franka arm to the target pose"

# A reachable goal pose (radians, 9 DOF: 7 arm + 2 fingers) the scripted policy drives toward.
GOAL_QPOS = np.array([0.6, -0.785, 0.0, -1.5, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=np.float32)


def main() -> None:
    driver = GenesisRobotDriver(
        robot=bundled_franka_mjcf(),
        # PD gains from the Genesis Franka reference; without them the arm won't track a target.
        kp=[4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        kv=[450, 450, 350, 350, 200, 200, 200, 10, 10],
        success_fn=lambda d: bool(np.max(np.abs(d.joint_positions() - GOAL_QPOS)) < 0.15),
    )
    env = GenesisEnv(driver, task=TASK, max_episode_steps=200)

    # The action that commands the goal pose; the arm converges to it and the episode terminates.
    goal_action = driver.normalize(GOAL_QPOS)

    sink = LeRobotDatasetSink(
        REPO_ID,
        fps=30,
        state_dim=driver.state_dim,
        action_dim=driver.action_dim,
        image_shape=driver.image_shape,
        camera=CAMERA,
        task=TASK,
        root=ROOT,
    )
    n_frames = record_episodes(
        env, policy=lambda _obs: goal_action, sink=sink, n_episodes=2, camera=CAMERA
    )
    print(f"recorded {n_frames} frames across 2 episodes -> {ROOT}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(REPO_ID, root=ROOT)
    print(f"reloaded dataset: {dataset.num_frames} frames, {dataset.num_episodes} episodes")


if __name__ == "__main__":
    main()
