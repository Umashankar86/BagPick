"""Calibrate a fixed placement ROI from the actual video's backpack reference."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import yaml
from core.geometry import iou
from utils.config import ROOT, resolve_path, validate


def padded_box(box, width, height, padding):
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("ROI padding must be nonnegative")
    dx, dy = (box[2] - box[0]) * padding, (box[3] - box[1]) * padding
    return (max(0, math.floor(box[0] - dx)), max(0, math.floor(box[1] - dy)),
            min(width, math.ceil(box[2] + dx)), min(height, math.ceil(box[3] + dy)))


def suggest_roi(detections, width, height, padding):
    bags = sorted((d for d in detections if d.label == "backpack"), key=lambda d: d.confidence, reverse=True)
    distinct = []
    for bag in bags:
        if all(iou(bag.box, other.box) < .5 for other in distinct):
            distinct.append(bag)
    if not distinct:
        raise ValueError("No backpack detected in the reference frame. Use --setup-roi to draw around the bag, or choose --roi-frame.")
    if len(distinct) > 1:
        raise ValueError("Multiple backpacks detected. Use --setup-roi to select the intended bag.")
    return padded_box(distinct[0].box, width, height, padding)


def selection_to_original(rect, original_size, displayed_size):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    sx, sy = original_size[0] / displayed_size[0], original_size[1] / displayed_size[1]
    return (max(0, math.floor(x * sx)), max(0, math.floor(y * sy)),
            min(original_size[0], math.ceil((x + w) * sx)),
            min(original_size[1], math.ceil((y + h) * sy)))


def save_roi(path, box):
    """Replace only coordinate scalars; retain thresholds, comments and other edits."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    document = yaml.compose(text)
    values = dict(zip(("x1", "y1", "x2", "y2"), box))
    roi_node = next((v for k, v in document.value if k.value == "roi"), None)
    if roi_node is None:
        updated = text.rstrip() + "\n\n" + yaml.safe_dump({"roi": values}, sort_keys=False)
    else:
        edits, missing = [], dict(values)
        for key, value in roi_node.value:
            if key.value in values:
                edits.append((value.start_mark.index, value.end_mark.index, str(values[key.value])))
                missing.pop(key.value)
        if missing:
            raise ValueError("ROI block must contain x1, y1, x2, y2 before saving")
        updated = text
        for start, end, replacement in sorted(edits, reverse=True):
            updated = updated[:start] + replacement + updated[end:]
    shutil.copy2(path, path.with_name(path.name + ".roi-backup"))
    path.write_text(updated, encoding="utf-8")


def setup_roi(cfg, config_path=None, *, automatic=False, frame_index=0, padding=.15):
    import cv2
    if frame_index < 0:
        raise ValueError("--roi-frame must be nonnegative")
    source = resolve_path(cfg["input_video_path"])
    if not source.is_file():
        raise FileNotFoundError(f"Missing reference video: {source}")
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot read reference frame {frame_index}")
    height, width = frame.shape[:2]
    if automatic:
        from models.detector import Detector
        detector = Detector(cfg["models"])
        box = suggest_roi(detector.decode(detector.predict(frame)), width, height, padding)
    else:
        scale = min(1.0, 1100 / width, 700 / height)
        display = cv2.resize(frame, (round(width * scale), round(height * scale)))
        title = "Draw around the INITIAL bag + small margin | ENTER save | C cancel"
        try:
            rect = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
        finally:
            cv2.destroyAllWindows()
        box = selection_to_original(rect, (width, height), (display.shape[1], display.shape[0]))
        if box is None:
            print("ROI selection cancelled. Configuration unchanged.")
            return None
    updated = deepcopy(cfg)
    updated["roi"].update(dict(zip(("x1", "y1", "x2", "y2"), box)))
    validate(updated)
    config_path = Path(config_path) if config_path else ROOT / "config/config.yaml"
    save_roi(config_path, box)
    cv2.rectangle(frame, box[:2], box[2:], (255, 180, 50), 2)
    cv2.putText(frame, "FIXED PLACEMENT ROI - initial backpack reference", (10, 30), 0, .7, (255, 180, 50), 2)
    preview = resolve_path("outputs/roi_preview.jpg")
    preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview), frame)
    print(f"Saved fixed ROI {box} in {config_path}. Preview: {preview}")
    print("The ROI stays at this initial location; it does not follow the moving bag.")
    return box
