"""Bounded contact checks with a historical confirmation per actor/hand/phase."""
from core.data import ContactAttempt, State


def sync_phase(runtime):
    phase = ("pickup" if runtime.state in (State.PLACED, State.PICKING) else
             "placement" if runtime.state in (State.PERSON_LEAVES, State.PLACING) else None)
    if phase != runtime.contact_phase:
        runtime.contact_attempts.clear()
        runtime.contact_phase = phase


def request_check(runtime, options, enabled):
    sync_phase(runtime)
    i = runtime.interaction
    eligible = (enabled and runtime.initialized and runtime.contact_phase is not None
                and runtime.backpack.visible and i.person_visible and i.wrist_fresh
                and i.wrist_near_backpack and i.actor_id is not None and i.hand is not None)
    if runtime.state in (State.PLACED, State.PERSON_LEAVES):
        eligible = eligible and i.wrist_inside_roi
    if not eligible:
        return None
    key = (i.actor_id, i.hand)
    attempt = runtime.contact_attempts.setdefault(key, ContactAttempt())
    if attempt.confirmation_frame is not None or attempt.attempts >= options["max_attempts_per_interaction"]:
        return None
    if attempt.attempts and runtime.frame - attempt.last_frame < options["retry_interval_frames"]:
        return None
    attempt.attempts += 1
    attempt.last_frame = runtime.frame
    return key


def record_result(runtime, key, results, threshold):
    if key is None:
        return
    attempt = runtime.contact_attempts[key]
    for result in results:
        if (result.person_id, result.hand) == key and result.detected and result.backpack_associated and result.confidence >= threshold:
            attempt.confirmation_frame = runtime.frame
            attempt.confidence = result.confidence
            break


def apply_confirmation(runtime):
    """Expose historical evidence without replaying a fresh contact detection."""
    i = runtime.interaction
    i.contact_confirmed = False
    i.contact_confirmation_frame = None
    attempt = runtime.contact_attempts.get((i.actor_id, i.hand))
    if attempt is not None and attempt.confirmation_frame is not None:
        i.contact_confirmed = True
        i.contact_confirmation_frame = attempt.confirmation_frame
        if not i.contact_available:
            i.contact_confidence = attempt.confidence
