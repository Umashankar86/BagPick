"""Run real pretrained inference and retain reviewable smoke-test artifacts.

The repeated-image MP4 checks integration, not pickup/placement accuracy.
"""
import json
import urllib.request
import cv2
from models.contact import CVPR2020ContactDetector
from models.detector import Detector
from models.pose import PoseEstimator
from tracking.tracker import Tracker
from pipeline import run_pipeline
from utils.config import load_config, resolve_path
from utils.logging_utils import configure_logging
from utils.video import VideoWriter


def main():
    import torch
    torch.set_num_threads(4)
    cfg = load_config()
    cfg["visualization"]["live_preview"] = False
    configure_logging(log_path=resolve_path("outputs/model_verification.log"))
    image_path = resolve_path("inputs/model_smoke_source.jpg")
    if not image_path.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Smoke-test source image could not be decoded")
    detector, pose = Detector(cfg["models"]), PoseEstimator(cfg["models"])
    tracker = Tracker(detector, cfg["tracking"])
    try:
        first = tracker.update(image)
        second = tracker.update(image)
        people = [d for d in second if d.label == "person"]
        poses = pose.estimate(image, people)
        ids1 = {d.track_id for d in first if d.label == "person" and d.track_id is not None}
        ids2 = {d.track_id for d in people if d.track_id is not None}
        assert people and ids1 & ids2, "BoT-SORT must preserve at least one visible person's ID"
        assert poses and any(p.left or p.right for p in poses.values()), "At least one wrist must be observed"
        assert all(d.label in ("person", "backpack") for d in second)
    finally:
        tracker.close()
    source = resolve_path("inputs/model_smoke_input.mp4")
    with VideoWriter(source, 4, (image.shape[1], image.shape[0])) as writer:
        for _ in range(12):
            writer.write(image)
    cfg["input_video_path"] = str(source)
    cfg["output_video_path"] = "outputs/model_smoke_processed.mp4"
    cfg["contact"]["enable_contact_detector"] = False
    report = run_pipeline(cfg, perception_only=True)

    contact = CVPR2020ContactDetector(cfg["contact"])
    reference = resolve_path(cfg["contact"]["repository_path"]) / "assets/boardgame_848_sU8S98MT1Mo_00013957.png"
    frame = cv2.imread(str(reference))
    if frame is None:
        raise RuntimeError("Contact reference image could not be decoded")
    predictions = contact.predict(frame)
    assert predictions, "Contact model must produce at least one hand result on its reference image"
    for item in predictions:
        color = (50, 220, 80) if item.detected else (0, 170, 255)
        p = tuple(map(int, item.hand_region))
        cv2.rectangle(frame, p[:2], p[2:], color, 2)
        cv2.putText(frame, f"{item.contact_state} score={item.confidence:.2f}", (p[0], max(14, p[1] - 8)), 0, .45, color, 1)
    cv2.imwrite(str(resolve_path("outputs/contact_model_smoke.jpg")), frame)
    summary = dict(yolo_model="yolo11n.pt", pose_model="yolo11n-pose.pt",
                   person_count=len(people), associated_pose_count=len(poses),
                   stable_track_ids=sorted(ids1 & ids2),
                   actual_contact_model_loaded=True, contact_hand_count=len(predictions),
                   contact_predictions=[item.__dict__ for item in predictions],
                   contact_reference="Upstream annotated boardgame example (smoke test only)",
                   video_smoke_report=report,
                   note="Real model inference verified. Repeated still image is not event-sequence acceptance.")
    resolve_path("outputs/model_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
