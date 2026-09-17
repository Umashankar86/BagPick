"""Actor and hand association based on track continuity and wrist geometry."""
from core.data import Interaction, State
from core.geometry import contains, iou, point_box_distance
from core.actor_trace import bind_actor


def choose_interaction(runtime, cfg):
    bag = runtime.backpack
    tolerance = cfg["temporal"]["wrist_loss_tolerance_frames"]
    near = cfg["interaction"]["wrist_backpack_max_distance_pixels"]
    locked = runtime.state in (State.PICKING, State.PICKED, State.PLACING)
    actor = runtime.people.get(runtime.actor_id)
    # Short in-view ID churn: rebind only ONE overlapping, wrist-near-bag candidate.
    # After a confirmed departure, the return candidate can have any tracker ID.
    if locked and actor and not actor.visible and bag.visible:
        if runtime.frame - actor.last_seen <= cfg["temporal"]["tracking_loss_tolerance_frames"]:
            replacements = [p for p in runtime.people.values() if p.visible and p.track_id != actor.track_id
                            and iou(p.box, actor.box) >= cfg["tracking"]["id_change_min_iou"]
                            and any(w.fresh(runtime.frame) and point_box_distance(w.point, bag.box) <= near
                                    for w in p.wrists.values())]
            if len(replacements) == 1:
                actor = replacements[0]
                runtime.actor_id = actor.track_id
                bind_actor(runtime, actor.track_id, "Short in-view track loss; unique spatial match")
    candidates = []
    people = [actor] if locked and actor else ([] if locked else list(runtime.people.values()))
    for person in people:
        if person is None:
            continue
        for hand, wrist in person.wrists.items():
            if locked and runtime.actor_hand is not None and hand != runtime.actor_hand:
                continue
            if not wrist.usable(runtime.frame, tolerance) or bag.box is None:
                continue
            distance = point_box_distance(wrist.point, bag.box)
            contact = person.contacts.get(hand)
            supportive = bool(contact and contact.backpack_associated)
            # Geometry gates candidate validity; contact breaks comparable ties.
            rank = (not person.visible, distance > near, not contains(runtime.roi, wrist.point),
                    not supportive, distance, person.track_id, hand)
            candidates.append((rank, person, hand, wrist, contact))
    if not candidates:
        return Interaction(actor_id=runtime.actor_id if locked else None,
                           hand=runtime.actor_hand if locked else None,
                           person_visible=bool(actor and actor.visible) if locked else False)
    _, person, hand, wrist, contact = min(candidates, key=lambda item: item[0])
    other = person.wrists["right" if hand == "left" else "left"]
    return Interaction(
        actor_id=person.track_id, hand=hand, wrist=wrist.point,
        wrist_fresh=person.visible and wrist.fresh(runtime.frame), wrist_usable=True,
        person_visible=person.visible, wrist_inside_roi=contains(runtime.roi, wrist.point),
        wrist_near_backpack=point_box_distance(wrist.point, bag.box) <= near,
        other_hand_engaged=other.usable(runtime.frame, tolerance) and contains(runtime.roi, other.point)
                           and point_box_distance(other.point, bag.box) <= near,
        wrist_moving=wrist.moving,
        contact_detected=bool(contact and contact.backpack_associated),
        contact_confidence=contact.confidence if contact else 0.0,
        contact_available=contact is not None,
    )
