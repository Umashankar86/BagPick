"""Human-readable observations; stale evidence is explicitly marked."""
import cv2
import numpy as np
from core.actor_trace import person_label


def visible_people_label(runtime, show_debug):
    """Recognition display is independent of bag interaction/state transitions."""
    people = sorted((p for p in runtime.people.values() if p.visible), key=lambda p: p.track_id)
    labels = [person_label(runtime, p.track_id, show_debug) for p in people[:3]]
    if len(people) > 3:
        labels.append(f"+{len(people) - 3} more")
    return "VISIBLE: " + (" | ".join(labels) if labels else "none")


def label_origin(rect, text_size, frame_size, panel_height):
    """Keep the label inside the image and below the information panel."""
    text_width, text_height = text_size
    frame_width, frame_height = frame_size
    x = max(4, min(int(rect[0]), frame_width - text_width - 8))
    y = min(frame_height - 8, max(panel_height + text_height + 10, int(rect[1]) - 6))
    return x, y


def annotate(frame, runtime, cfg, demo=False, perception_only=False):
    image = frame.copy()
    font, scale, width = cv2.FONT_HERSHEY_SIMPLEX, cfg["font_scale"], cfg["line_thickness"]

    def text(label, point, color=(235, 235, 235)):
        cv2.putText(image, label, tuple(map(int, point)), font, scale, color, 1, cv2.LINE_AA)

    def box(rect, color, label):
        x1, y1, x2, y2 = map(int, rect)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, width)
        text(label, (x1, max(14, y1 - 6)), color)

    box(runtime.roi, (255, 180, 50), "PLACEMENT ROI")
    bag = runtime.backpack
    if bag.box:
        box(bag.box, (0, 190, 255) if bag.visible else (110, 110, 110),
            f"backpack {bag.confidence:.2f}" + ("" if bag.visible else " [stale]"))
    person_labels = []
    for person in runtime.people.values():
        if not person.visible:
            continue
        cv2.rectangle(image, tuple(map(int, person.box[:2])), tuple(map(int, person.box[2:])), (90, 230, 90), width)
        person_labels.append((person.box, person_label(runtime, person.track_id, cfg["show_debug"])))
        for hand, wrist in person.wrists.items():
            if wrist.point is not None:
                fresh = wrist.fresh(runtime.frame)
                color = (220, 80, 230) if fresh else (100, 100, 100)
                cv2.circle(image, tuple(map(int, wrist.point)), 5, color, -1 if fresh else 1)
                text(hand[0].upper() + ("" if fresh else "?"), (wrist.point[0] + 7, wrist.point[1]), color)
    for person in runtime.untracked_people:
        box(person.box, (150, 150, 150), "person [awaiting track ID]")
    if cfg["show_trajectories"] and len(bag.history) > 1:
        points = np.array([sample[1] for sample in bag.history], np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, (0, 160, 210), 1)
    inter = runtime.interaction
    contact_label = (f"CONFIRMED (frame {inter.contact_confirmation_frame})" if inter.contact_confirmed
                     else ("YES" if inter.contact_detected else "NO") if runtime.contact_sampled or runtime.contact_status != "enabled"
                     else "NOT CHECKED")
    actor_label = f"Person {runtime.logical_actor_id} (appearance)" if runtime.logical_actor_id is not None else "unknown"
    if runtime.state.value == "PERSON_LEAVES":
        actor_label = "awaiting bag return; see visible people below"
    lines = [
        "SYNTHETIC DEMO / scripted observations" if demo else "YOLO11 + BoT-SORT + YOLO11 Pose",
        f"STATE: {runtime.state.value}" + (" [perception only]" if perception_only else ""),
        f"ACTOR: {actor_label}  HAND: {inter.hand}",
        visible_people_label(runtime, cfg["show_debug"]),
        f"CONTACT: {contact_label} ({runtime.contact_status})",
        f"REID: {runtime.recognition_status}  PICKUP PERSON: {runtime.pickup_person_id or 'unknown'}",
    ]
    if cfg["show_debug"]:
        lines += [f"Frame {runtime.frame}  ROI overlap {bag.roi_overlap:.0%}  {'UNCERTAIN' if runtime.uncertain else 'observed'}",
                  runtime.transition_reason[:85]]
    panel_height = 9 + 20 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (22, 25, 30), -1)
    image = cv2.addWeighted(overlay, 0.83, image, 0.17, 0)
    for i, line in enumerate(lines):
        text(line, (10, 20 + 20 * i))
    # Draw labels last with opaque backing, so the panel cannot dim/hide them.
    for rect, label in person_labels:
        size, baseline = cv2.getTextSize(label, font, scale, 1)
        x, y = label_origin(rect, size, (image.shape[1], image.shape[0]), panel_height)
        cv2.rectangle(image, (x - 3, y - size[1] - 4),
                      (x + size[0] + 3, y + baseline + 3), (22, 25, 30), -1)
        text(label, (x, y), (90, 230, 90))
    return image
