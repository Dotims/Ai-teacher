"""
capture.py – Screen capture module.

Uses mss to grab the primary monitor and returns raw PNG bytes
entirely in memory (no disk I/O).
"""

from io import BytesIO
import os
import mss
from PIL import Image


MAX_CAPTURE_SIDE = int(os.getenv("MAX_CAPTURE_SIDE", "1280"))


def capture_screen() -> bytes:
    """Capture the primary monitor and return PNG bytes."""
    with mss.mss() as sct:
        # mss monitor list: index 0 = virtual screen, 1 = primary monitor
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        # Convert raw BGRA pixels to a PIL Image, then to PNG bytes
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        if MAX_CAPTURE_SIDE > 0 and max(img.size) > MAX_CAPTURE_SIDE:
            img.thumbnail((MAX_CAPTURE_SIDE, MAX_CAPTURE_SIDE), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
