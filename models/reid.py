"""Replaceable person appearance encoder. No face recognition or bag matching."""
import numpy as np
from utils.config import resolve_path


class OpenVINOEncoder:
    """encode(BGR person crop) -> normalized appearance descriptor."""

    def __init__(self, cfg):
        import openvino as ov
        path = resolve_path(cfg["model_path"])
        if not path.is_file() or not path.with_suffix(".bin").is_file():
            raise FileNotFoundError("ReID weights missing; run python -m scripts.setup_reid")
        self.model = ov.Core().compile_model(str(path), "CPU", {"INFERENCE_NUM_THREADS": 2})
        self.output = self.model.output(0)

    def encode(self, crop):
        import cv2
        # Official input: BGR, raw 0..255, NCHW 1x3x256x128 (normalization in IR).
        tensor = cv2.resize(crop, (128, 256)).transpose(2, 0, 1)[None].astype(np.float32)
        vector = np.asarray(self.model([tensor])[self.output]).reshape(-1)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
