"""Ultralytics BoT-SORT. Detection and tracking happen in ONE model call."""
from pathlib import Path
import tempfile
import yaml


class Tracker:
    def __init__(self, detector, cfg):
        self.detector = detector
        self._directory = tempfile.TemporaryDirectory(prefix="bagpick_tracker_")
        self.path = Path(self._directory.name) / "botsort.yaml"
        settings = {k: cfg[k] for k in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh")}
        settings.update(tracker_type="botsort", fuse_score=True, gmc_method="none",
                        proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto")
        self.path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    def update(self, frame):
        result = self.detector.model.track(frame, persist=True, tracker=str(self.path),
                                           **self.detector.arguments())[0]
        return self.detector.decode(result)

    def close(self):
        self._directory.cleanup()
