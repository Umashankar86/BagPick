"""Read-only perception diagnostic for the final placement in a recording."""
import json
from core.data import Observation
from core.engine import Engine
from models.contact import ContactService
from models.detector import Detector
from models.pose import PoseEstimator
from tracking.tracker import Tracker
from utils.config import load_config, resolve_path
from utils.video import VideoReader


def main():
    cfg = load_config()
    detector, pose = Detector(cfg["models"]), PoseEstimator(cfg["models"])
    tracker = Tracker(detector, cfg["tracking"])
    engine = Engine(cfg, ContactService(cfg["contact"]))
    rows = []
    try:
        with VideoReader(resolve_path(cfg["input_video_path"])) as reader:
            for idx, frame in enumerate(reader):
                detections = tracker.update(frame)
                people = [d for d in detections if d.label == "person"]
                r = engine.step(frame, Observation(people, [d for d in detections if d.label == "backpack"], pose.estimate(frame, people)), idx + 1, idx / reader.fps)
                i, b = r.interaction, r.backpack
                row = dict(frame=r.frame, state=r.state.value, bag_visible=b.visible,
                           overlap=round(b.roi_overlap, 2), moving=b.moving,
                           actor=i.actor_id, hand=i.hand, wrist=i.wrist,
                           wrist_fresh=i.wrist_fresh, wrist_roi=i.wrist_inside_roi,
                           near=i.wrist_near_backpack, other=i.other_hand_engaged,
                           counters=dict(r.counters))
                rows.append(row)
                if idx >= 150 and idx % 3 == 0:
                    print(json.dumps(row), flush=True)
    finally:
        tracker.close()
    resolve_path("outputs/placement_diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
