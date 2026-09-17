from unittest.mock import patch
import numpy as np
from utils.preview import LivePreview


def test_preview_quit_and_cleanup():
    with patch("utils.preview.cv2.namedWindow"), patch("utils.preview.cv2.resizeWindow"), \
         patch("utils.preview.cv2.imshow") as shown, patch("utils.preview.cv2.waitKey", return_value=ord("q")), \
         patch("utils.preview.cv2.destroyWindow") as closed:
        preview = LivePreview(True)
        assert not preview.show(np.zeros((100, 200, 3), dtype=np.uint8), 1, 10, 3)
        shown.assert_called_once()
        preview.close()
        closed.assert_called_once()


def test_disabled_preview_never_opens_window():
    with patch("utils.preview.cv2.namedWindow") as window:
        assert LivePreview(False).show(None, 1, 10, 3)
        window.assert_not_called()


def test_closing_preview_window_requests_clean_stop():
    preview = LivePreview(True)
    preview.opened = True
    with patch("utils.preview.cv2.getWindowProperty", return_value=0):
        assert not preview.show(None, 1, 10, 3)
