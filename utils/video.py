"""Checked video IO with guaranteed resource cleanup."""
from pathlib import Path
import math
import cv2


class VideoReader:
    def __init__(self, path, fallback_fps=30.0):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Cannot open video: {path}")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else fallback_fps
        if min(self.width, self.height) <= 0:
            self.capture.release()
            raise OSError("Video dimensions are invalid")

    def __iter__(self):
        while True:
            success, frame = self.capture.read()
            if not success:
                break
            yield frame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.capture.release()


class VideoWriter:
    def __init__(self, path, fps, size, codec="mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.count = 0
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError(f"Cannot create MP4 writer: {path} (codec={codec})")

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("Output frame size changed")
        self.writer.write(frame)
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.writer.release()
