"""
audio_capture.py – System audio capture via WASAPI Loopback.

Captures system audio, buffers it in memory, and allows saving to a WAV file.
"""

import os
import wave
import tempfile
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600
DTYPE = "int16"


def _find_wasapi_loopback_device() -> int | None:
    """Find a WASAPI loopback device for system audio capture."""
    try:
        host_apis = sd.query_hostapis()
        wasapi_index = None
        for i, api in enumerate(host_apis):
            if "WASAPI" in api["name"]:
                wasapi_index = i
                break

        if wasapi_index is None:
            return None

        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if (
                dev["hostapi"] == wasapi_index
                and dev["max_input_channels"] > 0
                and "loopback" in dev["name"].lower()
            ):
                return i

        wasapi_api = host_apis[wasapi_index]
        default_output = wasapi_api.get("default_output_device", -1)
        if default_output >= 0:
            return default_output

    except Exception:
        pass

    return None


class SystemAudioCapture:
    """Captures system audio output and saves it to a WAV file."""

    def __init__(self):
        self._frames = []
        self._stream = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start capturing system audio to memory."""
        if self._running:
            return

        self._frames.clear()
        device = _find_wasapi_loopback_device()

        def _audio_callback(indata, frames, time_info, status):
            if status:
                pass
            pcm = indata.copy()
            if pcm.dtype != np.int16:
                pcm = (pcm * 32767).astype(np.int16)
            self._frames.append(pcm.tobytes())

        try:
            extra_settings = None
            if device is not None:
                try:
                    extra_settings = sd.WasapiSettings(auto_convert=True)
                except AttributeError:
                    pass

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                device=device,
                callback=_audio_callback,
                extra_settings=extra_settings,
            )
            self._stream.start()
            self._running = True
        except Exception:
            # Fallback output device
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=_audio_callback,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        """Stop capturing."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False

    def save_wav(self) -> str | None:
        """Save captured buffer to a temporary WAV file and return the path."""
        if not self._frames:
            return None

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, "stealth_interview_audio.wav")

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self._frames))

        return filepath
