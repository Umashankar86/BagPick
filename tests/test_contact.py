from dataclasses import replace
from unittest.mock import patch
import pytest
from models.contact import (ContactService, HandQuery, RawContact, associate_contacts,
                            create_contact_detector)
from tests.settings import load_config


@pytest.fixture
def cfg():
    return load_config()["contact"]


def test_disabled_does_not_import_or_load_provider(cfg):
    with patch("models.contact.CVPR2020ContactDetector", side_effect=AssertionError("must not load")):
        service = create_contact_detector(cfg)
        assert service.status == "disabled"
        assert service.detect(None, [], None) == []


def test_failed_model_load_warns_and_continues(cfg, caplog):
    cfg["enable_contact_detector"] = True
    with patch("models.contact.CVPR2020ContactDetector", side_effect=ImportError("optional dependency missing")):
        service = create_contact_detector(cfg)
    assert service.status == "unavailable"
    assert service.detect(None, [HandQuery(1, "left", (20, 20))], (0, 0, 40, 40)) == []
    assert "Continuing without contact" in caplog.text


def test_inference_failure_warns_once_and_disables(cfg, caplog):
    class Broken:
        def predict(self, frame):
            raise RuntimeError("bad tensor")
    service = ContactService(cfg, Broken(), "enabled")
    for _ in range(5):
        assert service.detect(None, [HandQuery(1, "left", (20, 20))], (0, 0, 40, 40)) == []
    assert service.status == "unavailable"
    assert caplog.text.count("Contact inference failed") == 1


def associate(cfg, raw, query=(20, 20)):
    return associate_contacts([raw], [HandQuery(1, "left", query)], (10, 10, 50, 60), cfg)[0]


def test_positive_contact_with_matching_object(cfg):
    result = associate(cfg, RawContact((10, 10, 30, 30), True, .9, (11, 11, 51, 61)))
    assert result.backpack_associated
    assert result.association == "object overlap"


def test_unrelated_object_does_not_get_rescued_by_wrist_proximity(cfg):
    result = associate(cfg, RawContact((10, 10, 30, 30), True, .9, (200, 200, 250, 260)))
    assert not result.backpack_associated


def test_no_object_region_uses_proximity_only_for_learned_positive(cfg):
    assert associate(cfg, RawContact((10, 10, 30, 30), True, .9)).backpack_associated
    assert not associate(cfg, RawContact((10, 10, 30, 30), False, .9)).backpack_associated


def test_low_confidence_contact_never_supports_backpack(cfg):
    assert not associate(cfg, RawContact((10, 10, 30, 30), True, .2, (10, 10, 50, 60))).backpack_associated


def test_hand_assignment_is_one_to_one(cfg):
    results = associate_contacts([RawContact((10, 10, 30, 30), True, .9)],
                                 [HandQuery(1, "left", (20, 20)), HandQuery(2, "right", (21, 21))],
                                 (10, 10, 50, 60), cfg)
    assert len(results) == 1
    assert results[0].person_id == 1


def test_malformed_provider_results_disable_safely(cfg, caplog):
    class Bad:
        def predict(self, frame):
            return [RawContact((10, 10, 30, 30), True, float("nan"))]
    service = ContactService(cfg, Bad(), "enabled")
    assert service.detect(None, [HandQuery(1, "left", (20, 20))], (10, 10, 50, 60)) == []
    assert service.status == "unavailable"
