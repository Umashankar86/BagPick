from copy import deepcopy
from dataclasses import replace
import pytest
from core.data import Detection, Observation, Pose, State
from core.engine import Engine
from models.contact import ContactService, RawContact
from tests.settings import load_config
from utils.demo import observation_at


@pytest.fixture
def cfg():
    return load_config()


def engine(cfg):
    return Engine(cfg, ContactService(cfg["contact"]))


def advance(e, start, end, transform=None):
    for index in range(start, end):
        obs = observation_at(index, e.cfg)
        if transform:
            obs = transform(obs, index)
        e.step(None, obs, index + 1, index / 15)
    return e.runtime


def test_complete_sequence_disabled_contact_new_returning_id_and_gaps(cfg):
    e = engine(cfg)
    r = advance(e, 0, 240)
    assert [x["state"] for x in r.transitions] == ["PICKING", "PICKED", "PERSON_LEAVES", "PLACING", "PLACED"]
    assert r.transitions[0]["actor_id"] is None
    assert r.transitions[3]["actor_id"] is None
    assert r.transitions[3]["track_id"] == 8
    assert {event["actor_id"] for event in r.transitions} == {None}
    assert r.state == State.PLACED
    assert all(not x["contact_detected"] for x in r.transitions)
    assert len(r.backpack.history) <= cfg["interaction"]["history_size"]
    assert len(r.backpack.boxes) <= cfg["interaction"]["history_size"]


def test_no_initialization_without_backpack_in_roi(cfg):
    e = engine(cfg)
    for frame in range(30):
        obs = Observation(backpacks=[Detection((450, 100, 500, 160), .99, "backpack")])
        e.step(None, obs, frame + 1, frame / 15)
    assert not e.runtime.initialized
    assert not e.runtime.transitions


def test_one_frame_wrist_entry_never_picks(cfg):
    e = engine(cfg)
    advance(e, 0, 35)
    e.step(None, observation_at(60, cfg), 36, 2.4)
    advance(e, 36, 40)
    assert not e.runtime.transitions


def test_disabled_missing_wrist_is_not_outside_roi(cfg):
    e = engine(cfg)
    advance(e, 0, 82)
    r = advance(e, 82, 125, lambda obs, _: replace(obs, poses={}))
    assert r.state == State.PICKING
    assert not r.interaction.wrist_fresh


def test_missing_backpack_is_not_removal(cfg):
    e = engine(cfg)
    advance(e, 0, 82)
    r = advance(e, 82, 125, lambda obs, _: replace(obs, backpacks=[]))
    assert r.state == State.PICKING
    assert not r.backpack.visible


def test_missing_wrist_does_not_mean_person_left(cfg):
    e = engine(cfg)
    advance(e, 0, 110)
    for idx in range(110, 150):
        obs = replace(observation_at(110, cfg), poses={})
        e.step(None, obs, idx + 1, idx / 15)
    assert e.runtime.state == State.PICKED


def test_unrelated_visible_person_does_not_replace_departed_actor(cfg):
    e = engine(cfg)
    advance(e, 0, 125)
    for idx in range(125, 145):
        bystander = Detection((0, 100, 90, 400), .9, "person", 50)
        e.step(None, Observation([bystander]), idx + 1, idx / 15)
    assert e.runtime.state == State.PERSON_LEAVES


def test_short_person_loss_does_not_leave(cfg):
    e = engine(cfg)
    advance(e, 0, 110)
    for idx in range(110, 113):
        e.step(None, Observation(), idx + 1, idx / 15)
    advance(e, 113, 125)
    assert e.runtime.state == State.PICKED


def test_short_id_change_rebinds_same_spatial_actor(cfg):
    e = engine(cfg)
    advance(e, 0, 110)
    obs = observation_at(110, cfg)
    obs = replace(obs, people=[replace(obs.people[0], track_id=44)], poses={44: obs.poses[1]})
    e.step(None, obs, 111, 111 / 15)
    assert e.runtime.actor_id == 44
    assert e.runtime.state == State.PICKED


def test_wrist_loss_cannot_finalize_placement(cfg):
    e = engine(cfg)
    advance(e, 0, 195)
    r = advance(e, 195, 240, lambda obs, _: replace(obs, poses={}))
    assert r.state == State.PLACING


def test_return_requires_backpack_enter_roi(cfg):
    e = engine(cfg)
    advance(e, 0, 145)
    for idx in range(145, 175):
        obs = observation_at(60, cfg)
        obs = replace(obs, backpacks=[Detection((420, 240, 460, 300), .9, "backpack", 30)])
        e.step(None, obs, idx + 1, idx / 15)
    assert e.runtime.state == State.PERSON_LEAVES


def test_grace_pauses_but_does_not_count_stale_frames(cfg):
    e = engine(cfg)
    advance(e, 0, 35)
    for idx in range(35, 39):
        e.step(None, observation_at(60, cfg), idx + 1, idx / 15)
    assert e.runtime.counters["picking"] == 4
    stale = replace(observation_at(60, cfg), poses={})
    for idx in range(39, 42):
        e.step(None, stale, idx + 1, idx / 15)
        assert e.runtime.state == State.PLACED
        assert e.runtime.counters["picking"] == 4
    e.step(None, stale, 43, 43 / 15)
    assert e.runtime.counters["picking"] == 0


def test_actor_changes_cannot_accumulate_confirmation(cfg):
    e = engine(cfg)
    advance(e, 0, 35)
    for idx in range(35, 55):
        tid = 10 + idx % 2
        obs = observation_at(60, cfg)
        obs = replace(obs, people=[replace(obs.people[0], track_id=tid)], poses={tid: obs.poses[1]})
        e.step(None, obs, idx + 1, idx / 15)
    assert e.runtime.state == State.PLACED


def test_untracked_person_cannot_start_event(cfg):
    e = engine(cfg)
    advance(e, 0, 35)
    for idx in range(35, 60):
        obs = observation_at(60, cfg)
        obs = replace(obs, people=[replace(obs.people[0], track_id=None)], poses={})
        e.step(None, obs, idx + 1, idx / 15)
    assert e.runtime.state == State.PLACED


def test_contact_strong_support_never_bypasses_fresh_roi_geometry(cfg):
    class Positive:
        def predict(self, frame):
            return [RawContact((100, 100, 400, 400), True, .99, (290, 260, 350, 350))]
    cfg["contact"]["supporting_confirmation_reduction_frames"] = 100
    e = Engine(cfg, ContactService(cfg["contact"], Positive(), "enabled"))
    r = advance(e, 0, 40)
    assert r.state == State.PLACED
    assert not r.transitions
    e.step(None, observation_at(60, cfg), 41, 41 / 15)
    assert e.runtime.state == State.PLACED


def test_contact_makes_valid_pick_more_confident_without_one_frame_transition(cfg):
    class Positive:
        def predict(self, frame):
            return [RawContact((290, 255, 325, 290), True, .99, (291, 260, 352, 350))]
    e = Engine(cfg, ContactService(cfg["contact"], Positive(), "enabled"))
    advance(e, 0, 35)
    for idx in range(35, 39):
        e.step(None, observation_at(60, cfg), idx + 1, idx / 15)
    assert e.runtime.state == State.PICKING
    assert e.runtime.transitions[0]["contact_confirmed"]
    assert e.contact_inferences == 1
    assert "contact confirmed" in e.runtime.transitions[0]["reason"]


def test_hysteresis_does_not_flip_inside_on_boundary_jitter(cfg):
    e = engine(cfg)
    advance(e, 0, 30)
    prior = observation_at(30, cfg)
    for idx, x in enumerate([365, 367, 364, 366]):
        obs = replace(prior, backpacks=[Detection((x, 260, x + 60, 350), .9, "backpack", 20)])
        e.step(None, obs, 31 + idx, (31 + idx) / 15)
        assert e.runtime.backpack.inside_roi
        assert not e.runtime.backpack.outside_roi


def test_perception_only_has_no_transitions(cfg):
    e = Engine(cfg, ContactService(cfg["contact"]), perception_only=True)
    r = advance(e, 0, 240)
    assert r.state == State.PLACED
    assert not r.transitions
