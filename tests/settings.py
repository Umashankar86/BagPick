"""Deterministic fixture settings independent of the user's video calibration."""
from utils.config import load_config as load_user_config


def load_config():
    cfg = load_user_config()
    cfg["roi"].update(x1=240, y1=180, x2=400, y2=380)
    cfg["contact"]["enable_contact_detector"] = False
    cfg["visualization"]["live_preview"] = False
    return cfg
