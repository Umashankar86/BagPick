import json
from unittest.mock import patch
from core.data import State
from core.engine import Engine
from models.contact import ContactService, create_contact_detector
from pipeline import run_pipeline
from tests.settings import load_config
from utils.demo import observation_at
from utils.video import VideoReader


def test_full_cycle_survives_enabled_but_failed_contact_load():
    cfg = load_config()
    cfg["contact"]["enable_contact_detector"] = True
    with patch("models.contact.CVPR2020ContactDetector", side_effect=ImportError("missing optional library")):
        contact = create_contact_detector(cfg["contact"])
    e = Engine(cfg, contact)
    for index in range(240):
        e.step(None, observation_at(index, cfg), index + 1, index / 15)
    assert [event["state"] for event in e.runtime.transitions] == ["PICKING", "PICKED", "PERSON_LEAVES", "PLACING", "PLACED"]
    assert e.runtime.contact_status == "unavailable"


def test_full_cycle_survives_contact_inference_failure():
    class Broken:
        def predict(self, frame):
            raise OSError("contact device failed")
    cfg = load_config()
    e = Engine(cfg, ContactService(cfg["contact"], Broken(), "enabled"))
    for index in range(240):
        e.step(None, observation_at(index, cfg), index + 1, index / 15)
    assert len(e.runtime.transitions) == 5
    assert e.runtime.state == State.PLACED
    assert e.runtime.contact_status == "unavailable"


def test_demo_writes_full_annotated_video_and_evidence_report(tmp_path):
    cfg = load_config()
    cfg["output_video_path"] = str(tmp_path / "custom_output.mp4")
    report = run_pipeline(cfg, demo=True)
    assert report["state_sequence"] == ["PLACED", "PICKING", "PICKED", "PERSON_LEAVES", "PLACING", "PLACED"]
    assert report["synthetic_demo"] is True
    path = tmp_path / "custom_output.mp4"
    assert json.loads(path.with_suffix(".events.json").read_text())["frames_processed"] == 240
    with VideoReader(path) as video:
        assert video.fps == 15
        assert sum(1 for _ in video) == 240
