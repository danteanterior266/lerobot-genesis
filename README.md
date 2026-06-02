# lerobot-genesis

[![CI](https://github.com/arun-prasath2005/lerobot-genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/arun-prasath2005/lerobot-genesis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lerobot-genesis.svg)](https://pypi.org/project/lerobot-genesis/)
[![Python](https://img.shields.io/pypi/pyversions/lerobot-genesis.svg)](https://pypi.org/project/lerobot-genesis/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The bridge between the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics
simulator and Hugging Face [LeRobot](https://github.com/huggingface/lerobot).

Genesis is a fast, open robotics simulator; LeRobot is the open standard for robot datasets and
policies. There is no official link between them — this package is that link, the same role
NVIDIA's *IsaacLab-Arena* plays for Isaac Lab. It lets you expose any Genesis scene as a
Gymnasium environment, record rollouts as a `LeRobotDataset`, and (from `v0.2`) evaluate a trained
LeRobot policy in Genesis through the standard `lerobot-eval` workflow.

```text
   Genesis scene ──▶ GenesisEnv (gym.Env) ──▶ LeRobotDataset ──▶ train a LeRobot policy
                          ▲                                              │
                          └────────────────── lerobot-eval ◀────────────┘   (v0.2)
```

## Install

```bash
pip install "lerobot-genesis[all]"   # bridge + genesis-world + lerobot
```

`genesis-world` needs a CUDA GPU and a few system libraries — see the
[Genesis install guide](https://genesis-world.readthedocs.io). The bridge itself is pure Python;
`pip install lerobot-genesis` (no extras) imports without either heavy dependency present.

## Quickstart — record a dataset

Drive a robot in a Genesis scene and save the rollout in LeRobot's format:

```python
from lerobot_genesis import GenesisEnv, GenesisRobotDriver, LeRobotDatasetSink, record_episodes

driver = GenesisRobotDriver(robot="path/to/robot.urdf")          # any URDF / MJCF / USD
env = GenesisEnv(driver, task="reach the target")
sink = LeRobotDatasetSink(
    "me/my-genesis-dataset",
    fps=30,
    state_dim=driver.state_dim,
    action_dim=driver.action_dim,
    image_shape=driver.image_shape,
    task="reach the target",
)

record_episodes(env, policy=lambda obs: env.action_space.sample(), sink=sink, n_episodes=10)
```

A runnable end-to-end example (a Franka arm, no asset download) lives in
[`examples/record_franka.py`](examples/record_franka.py).

## How it's designed

The simulator-specific work sits behind one small `Protocol`, so the Gymnasium contract and the
recorder are testable on a CPU with no GPU and no upstream installed.

- **`GenesisEnv`** — a `gymnasium.Env` over a Genesis scene. It owns the gym contract (the 5-tuple
  step, `info["is_success"]`, the `pixels`/`agent_pos` observation, a continuous action `Box`) and
  delegates the physics to an injected **`SceneDriver`**.
- **`GenesisRobotDriver`** — the reference `SceneDriver`: loads a robot, maps a normalised action
  onto a chosen subset of joints, steps physics, and reads a camera and joint state. Bring your own
  driver for richer scenes (objects, tasks, rewards).
- **`record_episodes` / `LeRobotDatasetSink`** — roll a policy and write a `LeRobotDataset`
  (the verified v3 `create → add_frame → save_episode → finalize` loop).
- **`GenesisEnvConfig`** — registers Genesis with LeRobot as `--env.type=genesis`.

## Relationship to `gym-genesis`

Hugging Face's [`gym-genesis`](https://github.com/huggingface/gym-genesis) provides raw Gymnasium
task scenes for Genesis. This package is complementary: it adds the LeRobot integration layer —
dataset recording, env registration, and policy evaluation — and can wrap a `gym-genesis` env just
as well as your own.

## Roadmap

- **v0.1** — `GenesisEnv`, reference driver, `LeRobotDataset` recorder, `--env.type=genesis`.
- **v0.2** — EnvHub `make_env` + `lerobot-eval` policy evaluation; GR00T `modality.json` adapter;
  native vectorised (multi-env) rollouts.

## License

[MIT](LICENSE). An independent, community project — not affiliated with the Genesis or LeRobot teams.
