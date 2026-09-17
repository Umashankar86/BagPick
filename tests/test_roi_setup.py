import yaml
import pytest
from core.data import Detection, Runtime
from core.backpack import update_backpack
from utils.config import load_config
from utils.roi_setup import padded_box, save_roi, selection_to_original, suggest_roi


def test_roi_comes_from_bag_and_is_clipped_to_video():
    assert padded_box((100, 200, 200, 400), 250, 420, .2) == (80, 160, 220, 420)


def test_selection_maps_preview_back_to_original_pixels():
    assert selection_to_original((100, 150, 50, 100), (1920, 1080), (960, 540)) == (200, 300, 300, 500)
    assert selection_to_original((0, 0, 0, 0), (1920, 1080), (960, 540)) is None


def test_auto_roi_rejects_ambiguity_but_deduplicates_same_bag():
    bag = Detection((100, 200, 200, 400), .9, "backpack")
    duplicate = Detection((102, 200, 202, 400), .8, "backpack")
    assert suggest_roi([bag, duplicate], 1000, 1000, 0) == (100, 200, 200, 400)
    with pytest.raises(ValueError, match="Multiple backpacks"):
        suggest_roi([bag, Detection((500, 200, 600, 400), .8, "backpack")], 1000, 1000, 0)
    with pytest.raises(ValueError, match="No backpack"):
        suggest_roi([], 1000, 1000, .1)


def test_saving_preserves_user_settings_and_comments(tmp_path):
    path = tmp_path / "config.yaml"
    original = "# user settings\nroi:\n  x1: 1 # left\n  y1: 2\n  x2: 3\n  y2: 4\n  overlap_threshold: 0.6\ncontact:\n  enable_contact_detector: true\n"
    path.write_text(original)
    save_roi(path, (100, 200, 300, 400))
    result = path.read_text()
    assert "x1: 100 # left" in result
    assert yaml.safe_load(result)["contact"]["enable_contact_detector"] is True
    assert path.with_name("config.yaml.roi-backup").read_text() == original


def test_bag_outside_roi_remains_visible_but_cannot_initialize():
    from core.state_machine import StateMachine
    cfg = load_config()
    runtime = Runtime((0, 0, 50, 50))
    for frame in range(1, 20):
        runtime.frame = frame
        update_backpack(runtime, [Detection((100, 100, 200, 250), .9, "backpack", 1)], cfg)
        StateMachine(cfg).update(runtime)
    assert runtime.backpack.visible
    assert runtime.backpack.box == (100, 100, 200, 250)
    assert not runtime.initialized
