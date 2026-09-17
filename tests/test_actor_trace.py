from core.actor_trace import bind_actor, person_label
from core.data import Runtime, State, Person
from tests.test_events import advance, engine
from tests.settings import load_config
from tracking.recognition import PersonRecognition
import numpy as np


def test_no_recognition_never_assumes_same_returning_person():
    e = engine(load_config())
    r = advance(e, 0, 240)
    assert [x["actor_id"] for x in r.transitions] == [None] * 5
    assert [x["track_id"] for x in r.transitions] == [1, 1, 1, 8, 8]
    assert r.actor_bindings == {}
    assert r.logical_actor_id is None
    assert not r.actor_trace
    assert person_label(r, 8, False) == "Person unknown"
    assert person_label(r, 8, True) == "Person unknown [track 8]"


def test_return_candidate_not_assigned_before_return_is_confirmed():
    e = engine(load_config())
    r = advance(e, 0, 155)
    assert r.state == State.PERSON_LEAVES
    assert r.logical_actor_id is None
    assert 8 not in r.actor_bindings
    assert "Person 1" not in person_label(r, 8, True)


def test_bystanders_do_not_inherit_actor_label():
    r = Runtime((0, 0, 100, 100))
    bind_actor(r, 5, "confirmed pickup", new_event=True)
    assert person_label(r, 99, False) == "Person unknown"
    assert r.actor_bindings == {}


def test_event_binding_only_consumes_independent_recognition():
    r = Runtime((0, 0, 100, 100))
    r.actor_bindings = {5: 1, 11: 2}
    bind_actor(r, 5, "pickup", new_event=True)
    bind_actor(r, 11, "return")
    assert r.logical_actor_id == 2
    bind_actor(r, 99, "next pickup", new_event=True)
    assert r.logical_actor_id is None
    assert 99 not in r.actor_bindings


def recognition():
    return PersonRecognition(load_config()["recognition"], encoder=object())


def confirm(model, runtime, tid, feature, visible=None):
    for _ in range(model.cfg["confirmation_samples"]):
        model.observe(runtime, tid, feature, visible or {tid})


def test_same_appearance_new_track_requires_several_samples():
    model, r = recognition(), Runtime((0, 0, 100, 100))
    model.cfg["confirmation_samples"] = 2
    confirm(model, r, 1, [1, 0, 0])
    model.observe(r, 111, [1, 0, 0], {111})
    assert 111 not in r.actor_bindings
    confirm(model, r, 111, [1, 0, 0])
    assert r.actor_bindings == {1: 1, 111: 1}
    assert r.actor_trace[-1]["reason"] == "appearance_match"


def test_different_person_not_linked_to_bag_owner():
    model, r = recognition(), Runtime((0, 0, 100, 100))
    confirm(model, r, 1, [1, 0, 0])
    confirm(model, r, 111, [0, 1, 0])
    assert r.actor_bindings == {1: 1, 111: 2}


def test_uncertain_similarity_stays_unknown():
    model, r = recognition(), Runtime((0, 0, 100, 100))
    confirm(model, r, 1, [1, 0, 0])
    uncertain = (model.cfg["new_identity_max_similarity"] + model.cfg["match_similarity"]) / 2
    confirm(model, r, 111, [uncertain, np.sqrt(1 - uncertain**2), 0])
    assert 111 not in r.actor_bindings
    assert r.identity_details[111]["status"] == "unknown"


def test_simultaneously_visible_people_cannot_share_identity():
    model, r = recognition(), Runtime((0, 0, 100, 100))
    confirm(model, r, 1, [1, 0, 0])
    confirm(model, r, 111, [1, 0, 0], {1, 111})
    assert 111 not in r.actor_bindings


def test_contradictory_appearance_revokes_track_identity():
    model, r = recognition(), Runtime((0, 0, 100, 100))
    confirm(model, r, 1, [1, 0, 0])
    for _ in range(model.cfg["confirmation_samples"]):
        model.observe(r, 1, [0, 1, 0], {1})
    assert r.actor_bindings.get(1) != 1


def test_runtime_failure_clears_identity_and_continues():
    class Broken:
        def encode(self, crop):
            raise RuntimeError("test dependency failure")
    model = PersonRecognition(load_config()["recognition"], encoder=Broken())
    r = Runtime((0, 0, 100, 100), frame=1)
    r.people[1] = Person(1, (0, 0, 100, 200), visible=True)
    r.actor_bindings[1] = 1
    model.update(np.zeros((240, 120, 3), np.uint8), r)
    assert r.recognition_status == "unavailable"
    assert not r.actor_bindings


def test_disabled_does_not_load_model():
    cfg = load_config()["recognition"]
    cfg.update(enabled=False, model_path="missing.xml")
    assert PersonRecognition(cfg).status == "disabled"


def test_missing_model_graceful():
    cfg = load_config()["recognition"]
    cfg["model_path"] = "weights/no_such_model.xml"
    assert PersonRecognition(cfg).status == "unavailable"


def test_different_returner_completes_event_with_separate_identity():
    e = engine(load_config())
    e.runtime.actor_bindings = {1: 1, 8: 2}
    r = advance(e, 0, 240)
    assert [x["actor_id"] for x in r.transitions] == [1, 1, 1, 2, 2]
    assert r.transitions[-1]["same_as_pickup_person"] is False
    assert r.state == State.PLACED


def test_two_close_gallery_candidates_remain_ambiguous():
    from collections import deque
    from tracking.recognition import normalized
    model, r = recognition(), Runtime((0, 0, 100, 100))
    model.gallery = {1: deque([normalized([1, 0, 0])]),
                     2: deque([normalized([.98, .2, 0])])}
    confirm(model, r, 111, [1, .1, 0])
    assert 111 not in r.actor_bindings


def test_same_raw_id_after_long_absence_must_revalidate():
    class Encoder:
        def encode(self, crop):
            return [1, 0, 0]
    model = PersonRecognition(load_config()["recognition"], Encoder())
    model.cfg["confirmation_samples"] = 2
    r = Runtime((0, 0, 100, 100), frame=1)
    confirm(model, r, 1, [1, 0, 0])
    model.last_seen[1] = 1
    r.frame = 100
    r.people[1] = Person(1, (0, 0, 100, 200), visible=True)
    model.update(np.zeros((240, 120, 3), np.uint8), r)
    assert 1 not in r.actor_bindings
    assert r.identity_details[1]["status"] == "pending"


def test_small_crops_do_not_advance_confirmation():
    model = recognition()
    r = Runtime((0, 0, 100, 100), frame=1)
    r.people[1] = Person(1, (0, 0, 10, 20), visible=True)
    model.update(np.zeros((240, 120, 3), np.uint8), r)
    assert not model.pending
    assert model.inferences == 0


def test_single_check_configuration_matches_immediately():
    cfg = load_config()
    cfg["recognition"]["confirmation_samples"] = 1
    from utils.config import validate
    validate(cfg)
    model = PersonRecognition(cfg["recognition"], encoder=object())
    r = Runtime((0, 0, 100, 100))
    model.observe(r, 1, [1, 0, 0], {1})
    model.observe(r, 111, [1, 0, 0], {111})
    assert r.actor_bindings == {1: 1, 111: 1}
    assert r.actor_trace[-1]["samples"] == 1
