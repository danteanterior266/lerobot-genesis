"""The GR00T modality.json adapter — pure, no GR00T/LeRobot/GPU."""

from __future__ import annotations

import json

import pytest

from lerobot_genesis.groot import EMBODIMENT_TAG_NEW, build_modality, write_modality_json


def test_state_and_action_fields_become_contiguous_slices() -> None:
    modality = build_modality(
        state_fields=[("arm", 7), ("gripper", 2)],
        action_fields=[("arm", 7), ("gripper", 2)],
        video_keys=["front"],
    )
    assert modality["state"] == {"arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 9}}
    assert modality["action"]["gripper"] == {"start": 7, "end": 9}


def test_video_keys_from_camera_list_and_mapping() -> None:
    from_list = build_modality(
        state_fields=[("s", 1)], action_fields=[("a", 1)], video_keys=["wrist"]
    )
    assert from_list["video"] == {"wrist": {"original_key": "observation.images.wrist"}}

    from_map = build_modality(
        state_fields=[("s", 1)],
        action_fields=[("a", 1)],
        video_keys={"ego": "observation.images.front"},
    )
    assert from_map["video"] == {"ego": {"original_key": "observation.images.front"}}


def test_annotations_listed_and_default_empty() -> None:
    assert (
        build_modality(state_fields=[("s", 1)], action_fields=[("a", 1)], video_keys=[])[
            "annotation"
        ]
        == {}
    )
    with_ann = build_modality(
        state_fields=[("s", 1)], action_fields=[("a", 1)], video_keys=[], annotations=["task"]
    )
    assert with_ann["annotation"] == {"task": {}}


def test_non_positive_width_rejected() -> None:
    with pytest.raises(ValueError, match="positive width"):
        build_modality(state_fields=[("bad", 0)], action_fields=[("a", 1)], video_keys=[])


def test_write_modality_json_roundtrips(tmp_path) -> None:
    modality = build_modality(
        state_fields=[("arm", 9)], action_fields=[("arm", 9)], video_keys=["front"]
    )
    path = write_modality_json(tmp_path, modality)
    assert path == tmp_path / "meta" / "modality.json"
    assert json.loads(path.read_text(encoding="utf-8")) == modality


def test_embodiment_tag_constant() -> None:
    assert EMBODIMENT_TAG_NEW == "new_embodiment"
