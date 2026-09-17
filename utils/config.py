"""One YAML configuration with validation before model loading."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base, update):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(base[key], dict) and key != "options":
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            _merge(base[key], value)
        else:
            base[key] = value


def load_config(path=None):
    defaults = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    cfg = deepcopy(defaults)
    if path:
        update = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(update, dict):
            raise ValueError("Config must be a YAML mapping")
        _merge(cfg, update)
    validate(cfg)
    return cfg


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def roi_box(cfg):
    return tuple(float(cfg["roi"][k]) for k in ("x1", "y1", "x2", "y2"))


def validate(cfg):
    recognition = cfg["recognition"]
    if type(recognition["enabled"]) is not bool:
        raise ValueError("recognition.enabled must be true or false")
    for key in ("match_similarity", "continuity_similarity", "new_identity_max_similarity", "match_margin"):
        if not isinstance(recognition[key], (int, float)) or not 0 <= recognition[key] <= 1:
            raise ValueError(f"recognition.{key} must be between 0 and 1")
    if recognition["new_identity_max_similarity"] >= recognition["match_similarity"]:
        raise ValueError("Recognition needs an uncertainty band between new and matching identities")
    if not recognition["new_identity_max_similarity"] < recognition["continuity_similarity"] <= recognition["match_similarity"]:
        raise ValueError("Recognition continuity threshold must lie above new-identity threshold and at/below match threshold")
    for key in ("sample_interval_frames", "confirmation_samples", "gallery_samples", "revalidate_gap_frames", "min_crop_height", "min_crop_width"):
        if type(recognition[key]) is not int or recognition[key] < 1:
            raise ValueError(f"Invalid recognition.{key}")
    if recognition["gallery_samples"] < recognition["confirmation_samples"]:
        raise ValueError("Recognition gallery must hold all confirmation samples")
    x1, y1, x2, y2 = roi_box(cfg)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError("ROI must have finite nonnegative coordinates with x1<x2 and y1<y2")
    for group in ("models", "tracking", "roi", "contact"):
        for key, value in cfg[group].items():
            if any(s in key for s in ("confidence", "overlap", "iou")) or key.endswith("_thresh"):
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"{group}.{key} must be between 0 and 1")
    if cfg["roi"]["exit_overlap_threshold"] >= cfg["roi"]["overlap_threshold"]:
        raise ValueError("ROI exit overlap must be lower than entry overlap (hysteresis)")
    for key, value in cfg["temporal"].items():
        minimum = 2 if key.endswith("required_frames") else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"temporal.{key} must be an integer >= {minimum}")
    for key in ("debug",):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a YAML boolean")
    if type(cfg["contact"]["enable_contact_detector"]) is not bool:
        raise ValueError("contact.enable_contact_detector must be true or false")
    if type(cfg["visualization"].get("live_preview", True)) is not bool:
        raise ValueError("visualization.live_preview must be true or false")
    for key in ("max_attempts_per_interaction", "retry_interval_frames"):
        value = cfg["contact"].get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"contact.{key} must be a positive integer")
    for key, value in cfg["interaction"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"interaction.{key} must be positive")
    if cfg["interaction"]["history_size"] < cfg["interaction"]["motion_window_frames"] + 1:
        raise ValueError("history_size must exceed motion_window_frames")
    for key in ("history_size", "motion_window_frames"):
        if type(cfg["interaction"][key]) is not int:
            raise ValueError(f"interaction.{key} must be an integer")
    for key, value in cfg["contact"].items():
        if key.endswith("_pixels") and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"contact.{key} must be nonnegative and finite")
    reduction = cfg["contact"]["supporting_confirmation_reduction_frames"]
    if type(reduction) is not int or reduction < 0:
        raise ValueError("contact.supporting_confirmation_reduction_frames must be a nonnegative integer")
    if len(cfg["video"]["codec"]) != 4:
        raise ValueError("video.codec must contain four characters")
    if cfg["video"]["fallback_fps"] <= 0 or cfg["demo"]["fps"] <= 0:
        raise ValueError("Frame rates must be positive")
    if cfg["missing_video_behavior"] not in ("demo", "error"):
        raise ValueError("missing_video_behavior must be demo or error")
    if resolve_path(cfg["input_video_path"]) == resolve_path(cfg["output_video_path"]):
        raise ValueError("Input and output video paths must differ")
