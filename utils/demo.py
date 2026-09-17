"""Synthetic engineering fixture. No learned-model accuracy is implied."""
import cv2
import numpy as np
from core.data import Detection, Observation, Pose
from utils.video import VideoWriter

FRAME_COUNT = 240


def observation_at(index, cfg):
    """Geometric scenario, not state labels; the real state machine infers events."""
    x1, y1, x2, y2 = (cfg["roi"][k] for k in ("x1", "y1", "x2", "y2"))
    w, h = x2 - x1, y2 - y1
    bx, by = x1 + w * .32, y1 + h * .4
    bw, bh = w * .38, h * .45
    shift = w * 1.5
    person_id = 1 if index < 145 else 8
    dx, dy = 0.0, 0.0
    hand = (x1 - 65.0, by - 20.0)
    if 40 <= index < 60:
        t = (index - 40) / 19
        hand = (x1 - 65 + t * (bx + 15 - x1 + 65), by + 10)
    elif 60 <= index < 80:
        hand = (bx + 15, by + 10)
    elif 80 <= index < 105:
        t = (index - 80) / 24
        dx, dy = shift * t, -40 * t
        hand = (bx + dx + 15, by + dy + 10)
    elif 105 <= index < 125:
        dx, dy = shift, -40
        hand = (bx + dx + 15, by + dy + 10)
    elif 125 <= index < 145:
        return Observation()
    elif 145 <= index < 175:
        t = (index - 145) / 29
        dx, dy = shift * (1 - t), -40 * (1 - t)
        hand = (bx + dx + 15, by + dy + 10)
    elif 175 <= index < 195:
        hand = (bx + 15, by + 10)
    elif 195 <= index < 215:
        t = (index - 195) / 19
        hand = (bx + 15 - t * (bx + 15 - x1 + 65), by + 10)
    bag = Detection((bx + dx, by + dy, bx + dx + bw, by + dy + bh), .95, "backpack", 20 if person_id == 1 else 30)
    person_box = (max(10.0, x1 - 130 + dx), 140.0, min(cfg["demo"]["width"] - 5.0, x1 + 110 + dx), cfg["demo"]["height"] - 10.0)
    person = Detection(person_box, .98, "person", person_id)
    other_hand = (person_box[0] + 25, by - 45)
    pose = Pose(person_box, hand, other_hand)
    # Actual gaps exercise grace handling; they do not advance confirmation.
    bags = [] if index in (68, 69) else [bag]
    poses = {} if index in (73, 74) else {person_id: pose}
    return Observation([person], bags, poses)


def render(index, cfg):
    width, height = cfg["demo"]["width"], cfg["demo"]["height"]
    image = np.full((height, width, 3), (39, 43, 49), dtype=np.uint8)
    cv2.line(image, (0, height - 25), (width, height - 25), (110, 110, 110), 2)
    obs = observation_at(index, cfg)
    for person in obs.people:
        x1, y1, x2, y2 = map(int, person.box)
        cx = (x1 + x2) // 2
        cv2.circle(image, (cx, y1 + 30), 24, (180, 185, 190), -1)
        cv2.line(image, (cx, y1 + 55), (cx, y2 - 90), (175, 180, 185), 18)
        for foot in (cx - 35, cx + 35):
            cv2.line(image, (cx, y2 - 90), (foot, y2), (175, 180, 185), 12)
        pose = obs.poses.get(person.track_id)
        if pose:
            for wrist in (pose.left, pose.right):
                cv2.line(image, (cx, y1 + 75), tuple(map(int, wrist)), (180, 185, 190), 8)
                cv2.circle(image, tuple(map(int, wrist)), 7, (195, 205, 225), -1)
    for bag in obs.backpacks:
        x1, y1, x2, y2 = map(int, bag.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (55, 120, 205), -1)
        cv2.rectangle(image, (x1 + 8, y1 + 25), (x2 - 8, y2 - 8), (40, 75, 155), 2)
        cv2.ellipse(image, ((x1 + x2) // 2, y1), (12, 10), 0, 180, 360, (55, 120, 205), 4)
    cv2.putText(image, "SYNTHETIC INPUT - NOT A REAL RECORDING", (10, 24), 0, .5, (255, 220, 160), 1, cv2.LINE_AA)
    return image


def generate_video(path, cfg):
    with VideoWriter(path, cfg["demo"]["fps"], (cfg["demo"]["width"], cfg["demo"]["height"]), cfg["video"]["codec"]) as writer:
        for index in range(FRAME_COUNT):
            writer.write(render(index, cfg))
    return path
