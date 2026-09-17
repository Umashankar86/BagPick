"""Select the event's backpack and maintain bounded motion/ROI memory."""
from core.data import State
from core.geometry import center, contains, distance, iou, overlap
from core.temporal import append_sample, motion, magnitude


def update_backpack(runtime, detections, cfg):
    bag, settings = runtime.backpack, cfg["interaction"]
    bag.visible, bag.moving, bag.motion_known = False, False, False
    bag.vector = (0.0, 0.0)
    candidates = list(detections)
    if not runtime.initialized:
        in_roi = [b for b in candidates if contains(runtime.roi, center(b.box))
                  and overlap(b.box, runtime.roi) >= cfg["roi"]["overlap_threshold"]]
        # Keep detections outside the ROI visible for calibration. The state
        # machine still requires stable INSIDE evidence before initialization.
        candidates = in_roi or candidates
    elif bag.box is not None and runtime.state != State.PERSON_LEAVES:
        exact = [b for b in candidates if bag.track_id is not None and b.track_id == bag.track_id]
        candidates = exact or [b for b in candidates if distance(center(b.box), bag.center)
                               <= settings["bag_association_max_distance_pixels"]]
    if not candidates:
        return
    if bag.box is None or runtime.state == State.PERSON_LEAVES:
        chosen = max(candidates, key=lambda b: b.confidence)
    else:
        chosen = max(candidates, key=lambda b: (iou(bag.box, b.box), -distance(center(b.box), bag.center), b.confidence))
    bag.box, bag.confidence, bag.track_id = chosen.box, chosen.confidence, chosen.track_id
    bag.visible, bag.last_seen = True, runtime.frame
    append_sample(bag.boxes, runtime.frame, chosen.box, settings["history_size"])
    append_sample(bag.history, runtime.frame, center(chosen.box), settings["history_size"])
    bag.vector, bag.motion_known = motion(bag.history, runtime.frame, settings["motion_window_frames"])
    bag.moving = bag.motion_known and magnitude(bag.vector) >= settings["bag_movement_threshold_pixels"]
    bag.center_inside_roi = contains(runtime.roi, bag.center)
    bag.roi_overlap = overlap(bag.box, runtime.roi)
    if bag.roi_overlap >= cfg["roi"]["overlap_threshold"] and bag.center_inside_roi:
        bag.inside_roi = True
    elif bag.roi_overlap <= cfg["roi"]["exit_overlap_threshold"] and not bag.center_inside_roi:
        bag.inside_roi = False
    # The ambiguous boundary band never counts as confirmed outside.
    bag.outside_roi = not bag.center_inside_roi and bag.roi_overlap <= cfg["roi"]["exit_overlap_threshold"]
