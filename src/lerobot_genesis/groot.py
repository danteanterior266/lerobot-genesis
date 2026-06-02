"""GR00T dataset adapter — emit the ``meta/modality.json`` NVIDIA Isaac-GR00T expects.

GR00T trains on a LeRobot dataset plus a ``meta/modality.json`` that splits the concatenated state
and action vectors into named fields, maps camera keys, and lists language annotations. Two caveats
from the GR00T docs (verified 2026-06): it currently reads the LeRobot **v2** layout (convert a v3
dataset with GR00T's ``scripts/lerobot_conversion/convert_v3_to_v2.py`` first), and a custom robot
trains under ``--embodiment-tag new_embodiment``. This module only writes ``modality.json`` and
needs neither GR00T nor LeRobot, so it stays pure and CPU-testable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: GR00T's embodiment tag for a robot that isn't one of its presets (selects a fresh action head).
EMBODIMENT_TAG_NEW = "new_embodiment"

Field = tuple[str, int]  # (name, width) — a named slice of the concatenated state/action vector


def _slots(fields: Sequence[Field]) -> dict[str, dict[str, int]]:
    slots: dict[str, dict[str, int]] = {}
    cursor = 0
    for name, width in fields:
        if width <= 0:
            raise ValueError(f"field {name!r} must have positive width, got {width}")
        slots[name] = {"start": cursor, "end": cursor + width}
        cursor += width
    return slots


def build_modality(
    *,
    state_fields: Sequence[Field],
    action_fields: Sequence[Field],
    video_keys: Mapping[str, str] | Sequence[str],
    annotations: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a GR00T ``modality.json`` mapping.

    ``state_fields`` / ``action_fields`` are ``(name, width)`` in vector order; widths must sum to
    the dataset's state / action dimension. ``video_keys`` is either a ``{new_key: original_key}``
    map or a list of camera names (each mapped to ``observation.images.<camera>``).
    """
    if isinstance(video_keys, Mapping):
        video = {new: {"original_key": original} for new, original in video_keys.items()}
    else:
        video = {camera: {"original_key": f"observation.images.{camera}"} for camera in video_keys}
    return {
        "state": _slots(state_fields),
        "action": _slots(action_fields),
        "video": video,
        "annotation": {name: {} for name in annotations},
    }


def write_modality_json(dataset_root: str | Path, modality: dict[str, Any]) -> Path:
    """Write ``modality`` to ``<dataset_root>/meta/modality.json`` and return its path."""
    path = Path(dataset_root) / "meta" / "modality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")
    return path
