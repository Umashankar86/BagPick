"""YOLO11 Detect: COCO person (0) and backpack (24) only."""
from core.data import Detection
from utils.config import resolve_path


class Detector:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        path = resolve_path(cfg["detection_model"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(path))

    def predict(self, frame):
        return self.model.predict(frame, **self.arguments())[0]

    def arguments(self):
        return dict(classes=[0, 24], conf=self.cfg["detection_confidence"],
                    imgsz=self.cfg["image_size"], device=self.cfg["device"], verbose=False)

    @staticmethod
    def decode(result):
        if result.boxes is None:
            return []
        boxes = result.boxes.cpu()
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        return [Detection(tuple(map(float, box)), float(score),
                          "person" if int(label) == 0 else "backpack",
                          int(tid) if tid is not None else None)
                for box, score, label, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), ids)
                if int(label) in (0, 24)]
