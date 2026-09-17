"""Only the five intended transitions, each requiring temporal primary evidence."""
import logging
from core.data import State
from core.geometry import distance
from core.temporal import count
from core.actor_trace import bind_actor

LOG = logging.getLogger(__name__)
NEXT = {State.PLACED: State.PICKING, State.PICKING: State.PICKED,
        State.PICKED: State.PERSON_LEAVES, State.PERSON_LEAVES: State.PLACING,
        State.PLACING: State.PLACED}


class StateMachine:
    def __init__(self, cfg):
        self.cfg = cfg

    def transition(self, r, target, reason):
        if NEXT[r.state] != target:
            raise ValueError(f"Illegal transition {r.state} -> {target}")
        previous = r.state
        if target in (State.PICKING, State.PLACING):
            r.actor_id, r.actor_hand = r.interaction.actor_id, r.interaction.hand
            bind_actor(r, r.actor_id,
                       "Confirmed pickup participant" if target == State.PICKING else
                       "Confirmed backpack return; identity from independent appearance model",
                       new_event=target == State.PICKING)
        if target == State.PICKING:
            r.pickup_person_id = r.logical_actor_id
            r.pickup_bag_anchor, r.pickup_wrist_anchor = r.backpack.center, r.interaction.wrist
        event = dict(frame=r.frame, timestamp=r.timestamp, previous=previous.value,
                     state=target.value, reason=reason, actor_id=r.logical_actor_id,
                     track_id=r.actor_id,
                     pickup_person_id=r.pickup_person_id,
                     same_as_pickup_person=(r.logical_actor_id == r.pickup_person_id
                                           if r.logical_actor_id is not None and r.pickup_person_id is not None else None),
                     identity_evidence=r.identity_details.get(r.actor_id, {"status": "unknown"}).copy(),
                     contact_confidence=r.interaction.contact_confidence,
                     contact_detected=r.interaction.contact_detected,
                     contact_confirmed=r.interaction.contact_confirmed,
                     contact_confirmation_frame=r.interaction.contact_confirmation_frame,
                     bag_inside_roi=r.backpack.inside_roi,
                     wrist_inside_roi=r.interaction.wrist_inside_roi)
        r.transitions.append(event)
        LOG.info("frame=%s %s -> %s | %s | actor=%s track=%s contact=%.3f bag_roi=%s wrist_roi=%s",
                 r.frame, previous.value, target.value, reason, r.logical_actor_id, r.actor_id,
                 event["contact_confidence"], event["bag_inside_roi"], event["wrist_inside_roi"])
        r.previous_state, r.state, r.backpack.state = previous, target, target
        r.transition_reason = reason
        r.counters.clear()
        r.candidate_key = None
        if target in (State.PERSON_LEAVES, State.PLACED):
            r.actor_id = r.actor_hand = None
        if target == State.PLACED:
            r.pickup_bag_anchor = r.pickup_wrist_anchor = None

    def update(self, r):
        t, settings = self.cfg["temporal"], self.cfg["interaction"]
        b, i = r.backpack, r.interaction
        fresh = b.visible and i.person_visible and i.wrist_fresh
        person = r.people.get(i.actor_id)
        person_grace = bool(person and r.frame - person.last_seen <= t["tracking_loss_tolerance_frames"])
        bag_grace = b.box is not None and r.frame - b.last_seen <= t["backpack_loss_tolerance_frames"]
        hold = bag_grace and i.wrist_usable and person_grace
        r.uncertain = not fresh

        if not r.initialized:
            stable = b.visible and b.inside_roi and b.motion_known and not b.moving
            n = count(r, "initialization", stable, fresh=b.visible, hold=bag_grace)
            if n >= t["initialization_required_frames"]:
                r.initialized = True
                r.counters.clear()
                r.transition_reason = "Initialized: backpack stable inside configured ROI"
                LOG.info("frame=%d %s", r.frame, r.transition_reason)
            return r.state

        if r.state in (State.PLACED, State.PERSON_LEAVES):
            key = (i.actor_id, i.hand)
            if key != r.candidate_key and fresh:
                r.counters.clear()
                r.candidate_key = key

        # A single confident learned result is historical supporting evidence.
        # ROI/wrist/backpack confirmation still requires multiple fresh frames.
        support = i.contact_confirmed and fresh
        reduction = self.cfg["contact"]["supporting_confirmation_reduction_frames"] if support else 0

        if r.state == State.PLACED:
            primary = b.inside_roi and i.wrist_inside_roi and i.wrist_near_backpack and i.actor_id is not None
            n = count(r, "picking", primary, fresh=fresh, hold=hold)
            required = max(2, t["wrist_inside_required_frames"] - reduction)
            if fresh and primary and n >= required:
                reason = "Wrist persistently inside ROI and near placed backpack"
                self.transition(r, State.PICKING, reason + ("; contact confirmed" if support else ""))

        elif r.state == State.PICKING:
            moved = r.pickup_bag_anchor is not None and b.center is not None and distance(b.center, r.pickup_bag_anchor) >= settings["pickup_displacement_pixels"]
            wrist_moved = r.pickup_wrist_anchor is not None and i.wrist is not None and distance(i.wrist, r.pickup_wrist_anchor) >= settings["wrist_pickup_displacement_pixels"]
            primary = b.outside_roi and not i.wrist_inside_roi and moved and wrist_moved and i.wrist_near_backpack
            n = count(r, "picked", primary, fresh=fresh, hold=hold)
            if fresh and primary and n >= t["bag_outside_required_frames"]:
                self.transition(r, State.PICKED, "Observed backpack and interacting wrist moved outside ROI together")

        elif r.state == State.PICKED:
            actor = r.people.get(r.actor_id)
            missing = actor is None or not actor.visible
            if count(r, "person_missing", missing) >= t["person_missing_required_frames"]:
                self.transition(r, State.PERSON_LEAVES, "Interacting person absent for configured consecutive frames")

        elif r.state == State.PERSON_LEAVES:
            primary = (i.actor_id is not None and i.wrist_inside_roi and i.wrist_near_backpack
                       and b.roi_overlap >= self.cfg["roi"]["return_overlap_threshold"])
            n = count(r, "returning", primary, fresh=fresh, hold=hold)
            required = max(2, t["return_required_frames"] - reduction)
            if fresh and primary and n >= required:
                self.transition(r, State.PLACING, "Returning person, wrist and backpack interact with ROI" + ("; contact confirmed" if support else ""))

        elif r.state == State.PLACING:
            primary = b.inside_roi and b.motion_known and not b.moving and not i.wrist_inside_roi and not i.other_hand_engaged
            n = count(r, "placed", primary, fresh=fresh, hold=hold)
            # No contact output means UNKNOWN, never proof that a hand released.
            if fresh and primary and n >= t["bag_inside_required_frames"]:
                self.transition(r, State.PLACED, "Backpack stable inside ROI; observed interacting wrist has left ROI")
        return r.state
