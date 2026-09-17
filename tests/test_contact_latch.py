from core.contact_evidence import request_check, record_result, apply_confirmation, sync_phase
from core.data import Runtime, Interaction, State
from models.contact import ContactResult

OPTIONS = dict(max_attempts_per_interaction=3, retry_interval_frames=3)


def ready():
    r = Runtime((0, 0, 100, 100))
    r.initialized = r.backpack.visible = True
    r.interaction = Interaction(actor_id=1, hand="left", person_visible=True, wrist_fresh=True,
                                wrist_near_backpack=True, wrist_inside_roi=True)
    return r


def test_success_stops_all_further_checks_and_is_not_fresh_detection():
    r = ready()
    r.frame = 1
    key = request_check(r, OPTIONS, True)
    record_result(r, key, [ContactResult(1, "left", True, .9, backpack_associated=True)], .6)
    for frame in range(2, 40):
        r.frame = frame
        assert request_check(r, OPTIONS, True) is None
        apply_confirmation(r)
        assert r.interaction.contact_confirmed
        assert r.interaction.contact_confirmation_frame == 1
        assert not r.interaction.contact_detected


def test_negative_uncertain_and_missing_results_have_bounded_spaced_retries():
    r = ready()
    calls = []
    for frame in range(1, 40):
        r.frame = frame
        key = request_check(r, OPTIONS, True)
        if key:
            calls.append(frame)
            record_result(r, key, [ContactResult(1, "left", True, .2, backpack_associated=True)], .6)
    assert calls == [1, 4, 7]
    assert not r.contact_attempts[(1, "left")].confirmation_frame


def test_successful_retry_stops_before_attempt_budget_is_used():
    r = ready()
    r.frame = 1
    key = request_check(r, OPTIONS, True)
    record_result(r, key, [], .6)
    r.frame = 4
    key = request_check(r, OPTIONS, True)
    record_result(r, key, [ContactResult(1, "left", True, .95, backpack_associated=True)], .6)
    for frame in range(5, 20):
        r.frame = frame
        assert request_check(r, OPTIONS, True) is None
    assert r.contact_attempts[(1, "left")].attempts == 2


def test_retries_pause_away_from_bag_and_budget_does_not_reset():
    r = ready()
    for frame in range(1, 50):
        r.frame = frame
        r.interaction.wrist_near_backpack = frame % 2 == 1
        request_check(r, OPTIONS, True)
    assert r.contact_attempts[(1, "left")].attempts == 3


def test_confirmation_not_transferred_to_another_actor_or_hand():
    r = ready()
    r.frame = 1
    key = request_check(r, OPTIONS, True)
    record_result(r, key, [ContactResult(1, "left", True, .9, backpack_associated=True)], .6)
    r.interaction.hand = "right"
    apply_confirmation(r)
    assert not r.interaction.contact_confirmed
    r.interaction.actor_id, r.interaction.hand = 2, "left"
    apply_confirmation(r)
    assert not r.interaction.contact_confirmed


def test_pickup_confirmation_clears_and_return_gets_new_budget():
    r = ready()
    r.frame = 1
    key = request_check(r, OPTIONS, True)
    record_result(r, key, [ContactResult(1, "left", True, .9, backpack_associated=True)], .6)
    r.state = State.PICKING
    sync_phase(r)
    assert r.contact_attempts[(1, "left")].confirmation_frame == 1
    r.state = State.PICKED
    sync_phase(r)
    apply_confirmation(r)
    assert not r.contact_attempts
    assert not r.interaction.contact_confirmed
    assert request_check(r, OPTIONS, True) is None
    r.state = State.PERSON_LEAVES
    r.frame = 100
    assert request_check(r, OPTIONS, True) == (1, "left")
    assert r.contact_attempts[(1, "left")].attempts == 1


def test_unrelated_positive_does_not_stop_retry_for_selected_hand():
    r = ready()
    r.frame = 1
    key = request_check(r, OPTIONS, True)
    record_result(r, key, [ContactResult(2, "left", True, .99, backpack_associated=True)], .6)
    r.frame = 4
    assert request_check(r, OPTIONS, True) == key
