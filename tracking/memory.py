"""Update tracked people without inventing permanent identities."""
from core.data import Person
from core.temporal import append_sample, magnitude, motion


def update_people(runtime, observation, cfg):
    settings = cfg["interaction"]
    for person in runtime.people.values():
        person.visible = False
        person.contacts.clear()
        for wrist in person.wrists.values():
            wrist.vector, wrist.moving = (0.0, 0.0), False
    runtime.untracked_people = [p for p in observation.people if p.track_id is None]
    for detection in observation.people:
        tid = detection.track_id
        if tid is None:
            continue
        person = runtime.people.setdefault(tid, Person(tid, detection.box))
        person.box, person.confidence = detection.box, detection.confidence
        person.visible, person.last_seen = True, runtime.frame
        pose = observation.poses.get(tid)
        for name, wrist in person.wrists.items():
            point = getattr(pose, name) if pose else None
            if point is not None:
                wrist.point, wrist.last_seen = point, runtime.frame
                append_sample(wrist.history, runtime.frame, point, settings["history_size"])
                wrist.vector, known = motion(wrist.history, runtime.frame, settings["motion_window_frames"])
                wrist.moving = known and magnitude(wrist.vector) >= settings["wrist_movement_threshold_pixels"]
    # Keep the locked actor until departure is confirmed. Discard other expired tracks.
    retention = max(cfg["temporal"]["person_missing_required_frames"], cfg["tracking"]["track_buffer"])
    for tid in list(runtime.people):
        if tid != runtime.actor_id and runtime.frame - runtime.people[tid].last_seen > retention:
            del runtime.people[tid]
