from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import pytest
from core.data import Detection, Pose
from core.geometry import overlap, iou, point_box_distance
from models.pose import associate_poses
from utils.config import load_config, validate
from utils.video import VideoReader, VideoWriter


def test_geometry_partial_overlap():
    assert overlap((0, 0, 10, 10), (5, 0, 15, 10)) == .5
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert point_box_distance((15, 5), (0, 0, 10, 10)) == 5


def test_pose_matching_one_to_one_preserves_original_coordinates():
    people = [Detection((0, 0, 100, 100), .9, "person", 10), Detection((100, 0, 200, 100), .9, "person", 20)]
    poses = [Pose((102, 0, 202, 100), (155, 45), None), Pose((1, 0, 101, 100), (40, 50), (60, 50))]
    result = associate_poses(people, poses, .2)
    assert result[10].left == (40, 50)
    assert result[20].left == (155, 45)
    assert len(result) == 2


@pytest.mark.parametrize("key,value", [("wrist_inside_required_frames", 1), ("bag_outside_required_frames", 0), ("return_required_frames", 1.5)])
def test_one_frame_and_fractional_temporal_config_rejected(key, value):
    cfg = load_config()
    cfg["temporal"][key] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_input_output_collision_rejected():
    cfg = load_config()
    cfg["output_video_path"] = cfg["input_video_path"]
    with pytest.raises(ValueError):
        validate(cfg)


def test_mp4_write_and_decode(tmp_path):
    path = tmp_path / "test.mp4"
    with VideoWriter(path, 15, (320, 240)) as writer:
        for _ in range(12):
            writer.write(np.full((240, 320, 3), 50, np.uint8))
    with VideoReader(path) as reader:
        assert (reader.width, reader.height) == (320, 240)
        assert reader.fps == pytest.approx(15)
        assert sum(1 for _ in reader) == 12


def test_missing_video_errors_cleanly(tmp_path):
    with pytest.raises(OSError):
        VideoReader(tmp_path / "missing.mp4")
