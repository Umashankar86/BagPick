"""Full-frame YOLO11 Pose; globally greedy one-to-one box/track association."""
from core.data import Pose
from core.geometry import iou
from utils.config import resolve_path


def associate_poses(people, poses, min_iou):
    pairs = sorted([(iou(p.box, pose.box), p.track_id, j)
                    for p in people if p.track_id is not None
                    for j, pose in enumerate(poses)], reverse=True)
    used_people, used_poses, result = set(), set(), {}
    for score, tid, j in pairs:
        if score < min_iou:
            break
        if tid not in used_people and j not in used_poses:
            result[tid] = poses[j]
            used_people.add(tid)
            used_poses.add(j)
    return result


class PoseEstimator:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        path = resolve_path(cfg["pose_model"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(path))

    def estimate(self, frame, people):
        if not people:
            return {}
        result = self.model.predict(frame, conf=self.cfg["pose_confidence"],
                                    imgsz=self.cfg["image_size"], device=self.cfg["device"], verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            return {}
        poses = []
        # Ultralytics xy is already mapped back to original frame coordinates.
        for box, points in zip(result.boxes.xyxy.cpu().tolist(), result.keypoints.data.cpu().tolist()):
            def wrist(index):
                x, y, confidence = points[index]
                return (float(x), float(y)) if confidence >= self.cfg["keypoint_confidence"] else None
            poses.append(Pose(tuple(map(float, box)), wrist(9), wrist(10)))
        return associate_poses(people, poses, self.cfg["pose_person_min_iou"])
