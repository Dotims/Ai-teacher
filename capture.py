"""
capture.py – Screen capture module.

Uses mss to grab the primary monitor and returns raw PNG bytes
entirely in memory (no disk I/O).
"""

from io import BytesIO
import mss
from PIL import Image


def capture_screen() -> bytes:
    """Capture the primary monitor and return PNG bytes."""
    with mss.mss() as sct:
        # mss monitor list: index 0 = virtual screen, 1 = primary monitor
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        # Convert raw BGRA pixels to a PIL Image, then to PNG bytes
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
