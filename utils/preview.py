"""Live processed-frame preview; closing the window finishes a valid partial MP4."""
import logging
import cv2

LOG = logging.getLogger(__name__)


class LivePreview:
    def __init__(self, enabled):
        self.enabled = enabled
        self.opened = False
        self.name = "BagPick - live processing (Q / Esc to stop)"

    def show(self, image, index, total, fps):
        if not self.enabled:
            return True
        try:
            if self.opened and cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if not self.opened:
                cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.name, 1100, 700)
                self.opened = True
            remaining = max(0, total - index) / fps if total and fps > 0 else 0
            canvas = image.copy()
            label = f"Processing {index}/{total or '?'} | {fps:.1f} FPS | ETA {remaining:.0f}s | Q: stop"
            cv2.rectangle(canvas, (0, canvas.shape[0] - 32), (canvas.shape[1], canvas.shape[0]), (25, 25, 25), -1)
            cv2.putText(canvas, label, (10, canvas.shape[0] - 11), 0, .55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(self.name, canvas)
            return cv2.waitKey(1) & 0xFF not in (ord("q"), ord("Q"), 27)
        except cv2.error as exc:
            LOG.warning("Live preview unavailable (%s); processing continues to MP4", exc)
            self.enabled = False
            return True

    def close(self):
        if self.opened:
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass
