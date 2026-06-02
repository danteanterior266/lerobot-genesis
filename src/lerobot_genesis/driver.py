"""A reference :class:`~lerobot_genesis.env.SceneDriver` for a single robot in a Genesis scene.

It loads a robot (URDF / MJCF / USD), maps a normalised action onto a chosen subset of its joints,
steps physics, and reads a camera and joint state. It is deliberately general — no task logic, no
objects beyond a ground plane — so it suits simple reach/teleop setups and serves as the template
for richer drivers. ``genesis`` is imported lazily, so importing this module is CPU-safe.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]
SuccessFn = Callable[["GenesisRobotDriver"], bool]


def bundled_franka_mjcf() -> str:
    """Path to the Franka Panda MJCF bundled with ``genesis`` (handy for examples; no download)."""
    import genesis as gs

    return os.path.join(os.path.dirname(gs.__file__), "assets", "xml/franka_emika_panda/panda.xml")


class GenesisRobotDriver:
    """Drive one robot in a Genesis scene for the :class:`~lerobot_genesis.env.GenesisEnv` contract.

    The scene is built eagerly so ``action_dim`` / ``state_dim`` / ``image_shape`` are known before
    the env reads them. Actions are normalised to ``[-1, 1]`` and mapped onto the controlled joints'
    real limits, so a policy never needs the robot's kinematics. ``gs.init`` configures the process
    globally (backend, precision); it is called once and reused if already initialised.

    Parameters
    ----------
    robot:
        Path to a URDF / MJCF / USD robot. The morph is chosen from the file extension.
    action_joints:
        Names of the joints the action controls, in order. Defaults to every DOF in model order.
        Each named joint must be single-DOF (typical for arms and grippers).
    home_qpos:
        Joint angles (radians, one per controlled joint) the robot resets to. Defaults to the
        midpoint of each joint's range.
    kp, kv:
        Optional PD gains for the controlled joints. Position control needs gains stiff enough to
        track a target; pass them when the robot model doesn't define adequate actuators.
    success_fn:
        Optional ``driver -> bool`` predicate used for ``is_success`` (e.g. end-effector reached a
        goal). Without it the episode only ends on truncation.
    steps_per_action:
        Physics steps advanced per control step (action decimation).
    """

    def __init__(
        self,
        *,
        robot: str,
        action_joints: Sequence[str] | None = None,
        home_qpos: Sequence[float] | None = None,
        kp: Sequence[float] | None = None,
        kv: Sequence[float] | None = None,
        success_fn: SuccessFn | None = None,
        steps_per_action: int = 4,
        camera_resolution: tuple[int, int] = (320, 240),
        camera_pos: Sequence[float] = (1.4, 0.0, 1.1),
        camera_lookat: Sequence[float] = (0.3, 0.0, 0.4),
        camera_fov: float = 45.0,
        backend: str = "gpu",
        show_viewer: bool = False,
    ) -> None:
        import genesis as gs

        self._success_fn = success_fn
        self._steps_per_action = max(1, int(steps_per_action))
        self._resolution = (int(camera_resolution[0]), int(camera_resolution[1]))  # (W, H)

        self._init_genesis(gs, backend)
        self._scene = gs.Scene(show_viewer=show_viewer)
        self._scene.add_entity(gs.morphs.Plane())
        self._robot = self._scene.add_entity(self._morph(gs, robot))
        self._camera = self._scene.add_camera(
            res=self._resolution,
            pos=tuple(camera_pos),
            lookat=tuple(camera_lookat),
            fov=float(camera_fov),
            GUI=False,
        )
        self._scene.build()

        # DOF indices and limits are only valid after build(); resolve the controlled subset by name
        # so the action never depends on the model's internal joint ordering.
        self._dofs = self._resolve_dofs(action_joints)
        lo, hi = self._robot.get_dofs_limit()
        self._low: FloatArray = np.asarray(lo.cpu(), dtype=np.float32)[self._dofs]
        self._high: FloatArray = np.asarray(hi.cpu(), dtype=np.float32)[self._dofs]
        self._span: FloatArray = np.maximum(self._high - self._low, 1e-6)
        self._home: FloatArray = self._resolve_home(home_qpos)

        if kp is not None:
            self._robot.set_dofs_kp(np.asarray(kp, dtype=np.float32), self._dofs)
        if kv is not None:
            self._robot.set_dofs_kv(np.asarray(kv, dtype=np.float32), self._dofs)

    # -- SceneDriver contract ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return len(self._dofs)

    @property
    def state_dim(self) -> int:
        return len(self._dofs)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self._resolution[1], self._resolution[0], 3)

    def reset(self) -> None:
        self._robot.set_dofs_position(self._home, self._dofs, zero_velocity=True)
        for _ in range(self._steps_per_action):
            self._scene.step()

    def apply_action(self, action: FloatArray) -> None:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self._low + (a + 1.0) * 0.5 * self._span
        self._robot.control_dofs_position(target, self._dofs)

    def step(self) -> None:
        for _ in range(self._steps_per_action):
            self._scene.step()

    def observe(self) -> tuple[ImageArray, FloatArray]:
        rendered = self._camera.render(rgb=True)
        rgb = rendered[0] if isinstance(rendered, tuple) else rendered
        return np.asarray(rgb, dtype=np.uint8), self.joint_positions()

    def is_success(self) -> bool:
        return bool(self._success_fn(self)) if self._success_fn is not None else False

    # -- helpers (also useful for scripted policies) -------------------------------------------
    def joint_positions(self) -> FloatArray:
        """Current angles of the controlled joints (radians)."""
        return np.asarray(self._robot.get_dofs_position(self._dofs).cpu(), dtype=np.float32)

    def normalize(self, qpos: Sequence[float]) -> FloatArray:
        """Map controlled-joint angles to the ``[-1, 1]`` action that commands them."""
        q = np.asarray(qpos, dtype=np.float32)
        scaled: FloatArray = 2.0 * (q - self._low) / self._span - 1.0
        return scaled.clip(-1.0, 1.0).astype(np.float32)

    @property
    def robot(self) -> object:
        """The underlying Genesis ``RigidEntity`` (for end-effector pose, custom rewards, etc.)."""
        return self._robot

    @staticmethod
    def _init_genesis(gs: Any, backend: str) -> None:
        try:
            gs.init(backend=getattr(gs, backend), logging_level="error")
        except Exception as exc:  # gs.init is process-global; reuse an existing initialisation
            if "already initialized" not in str(exc).lower():
                raise
        # Genesis sets torch's global default device to the GPU. Restore the usual CPU default, or
        # downstream torch code (LeRobot's eval rollout and policies) gets forced onto cuda. The sim
        # runs on Genesis's own backend regardless; we read its tensors via explicit .cpu().
        import torch

        torch.set_default_device("cpu")

    @staticmethod
    def _morph(gs: Any, path: str) -> Any:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".urdf":
            return gs.morphs.URDF(file=path, fixed=True)
        if ext in (".usd", ".usda", ".usdc"):
            return gs.morphs.USD(file=path, fixed=True)
        return gs.morphs.MJCF(file=path)

    def _resolve_dofs(self, action_joints: Sequence[str] | None) -> list[int]:
        if action_joints is None:
            return list(range(int(self._robot.n_dofs)))
        dofs: list[int] = []
        for name in action_joints:
            idx = self._robot.get_joint(name).dof_idx_local
            dofs.extend(idx if hasattr(idx, "__iter__") else [idx])
        return dofs

    def _resolve_home(self, home_qpos: Sequence[float] | None) -> FloatArray:
        midpoint: FloatArray = (self._low + self._high) * 0.5
        if home_qpos is None:
            return midpoint.astype(np.float32)
        q = np.asarray(home_qpos, dtype=np.float32)
        if q.shape != (len(self._dofs),):
            raise ValueError(f"home_qpos must have {len(self._dofs)} values, got {q.shape}")
        return q.clip(self._low, self._high).astype(np.float32)
