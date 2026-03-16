"""
audio_capture.py – System audio capture via PyAudio WASAPI Loopback.

Uses PyAudioWPatch (as_loopback=True) on the default WASAPI output device,
so we always capture exactly what the user hears in their speakers/headphones –
no more guessing between SteelSeries Sonar virtual endpoints.
"""

import os
import wave
import tempfile
import threading
import time
import queue
import numpy as np


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


BLOCK_SIZE = max(1024, _env_int("AUDIO_BLOCK_SIZE", 2048))
SPEECH_PROB_THRESHOLD = 0.5


class SystemAudioCapture:
    """Captures system audio via WASAPI loopback on the default output device."""

    def __init__(self, on_audio_ready_callback=None):
        self._frames = []
        self._running = False
        self.on_audio_ready = on_audio_ready_callback
        self._enable_vad_debug = os.getenv("ENABLE_VAD_DEBUG", "0") == "1"
        self._capture_mode = (os.getenv("AUDIO_CAPTURE_MODE", "system").strip().lower() or "system")
        if self._capture_mode not in {"system", "mic", "both"}:
            self._capture_mode = "system"

        self.is_speech_active = False
        self._trigger_active = False
        self._data_queue: queue.Queue = queue.Queue()

        self._sample_rate: int = _env_int("AUDIO_CAPTURE_SAMPLE_RATE", 48000)
        self._active_source_name: str = "loopback"
        self._last_rms: float = 0.0

        self._torch = None
        self.model = None
        if self._enable_vad_debug:
            self._try_init_vad()

        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._pa = None

    def _try_init_vad(self) -> None:
        try:
            import torch
            self._torch = torch
            print("[VAD] Ladowanie modelu Silero VAD...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
            print("[VAD] Silero VAD zaladowane.")
        except Exception as exc:
            self._torch = None
            self.model = None
            print(f"[VAD] Niedostepny ({exc}). Kontynuuje bez VAD.")

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        self._frames.clear()
        self.is_speech_active = False
        self._trigger_active = False
        self._last_rms = 0.0
        self._running = True

        import pyaudiowpatch as pyaudio
        self._pa = pyaudio.PyAudio()

        # Locate the default WASAPI output device for loopback.
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception as exc:
            print(f"[Audio] WASAPI niedostepne: {exc}")
            self._running = False
            self._pa.terminate()
            self._pa = None
            return

        default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
        if default_out_idx < 0:
            print("[Audio] Brak domyslnego urzadzenia wyjsciowego WASAPI.")
            self._running = False
            self._pa.terminate()
            self._pa = None
            return

        speakers = self._pa.get_device_info_by_index(default_out_idx)
        try:
            loopback_dev = self._pa.get_wasapi_loopback_analogue_by_dict(speakers)
        except Exception as exc:
            print(f"[Audio] Nie mozna znalezc loopback dla {speakers.get('name')}: {exc}")
            self._running = False
            self._pa.terminate()
            self._pa = None
            return

        device_index = int(loopback_dev["index"])
        num_channels = int(loopback_dev.get("maxInputChannels", 2)) or 2
        sample_rate = int(loopback_dev.get("defaultSampleRate", 48000))
        self._sample_rate = sample_rate
        self._active_source_name = str(speakers.get("name", "loopback"))

        print(f"[Audio] Domyslne urzadzenie WASAPI: {self._active_source_name}")
        print(f"[Audio] Loopback: {loopback_dev.get('name')} | ch={num_channels} | rate={sample_rate}")

        def _capture_loopback():
            try:
                import pyaudiowpatch as pyaudio
                stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=num_channels,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=BLOCK_SIZE,
                )
                print(f"[Audio] Loopback uruchomiony: {self._active_source_name}")
                while self._running:
                    try:
                        raw = stream.read(BLOCK_SIZE, exception_on_overflow=False)
                    except Exception as exc:
                        if self._running:
                            print(f"[Audio] Blad odczytu loopback: {exc}")
                        break
                    # bytes → int16 → float32 → mono
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if num_channels > 1:
                        pcm = pcm.reshape(-1, num_channels).mean(axis=1)
                    pcm = np.nan_to_num(pcm, nan=0.0, posinf=0.0, neginf=0.0)
                    pcm = np.clip(pcm, -1.0, 1.0)
                    self._data_queue.put(pcm)
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[Audio] Nie mozna otworzyc loopback: {exc}")

        def _capture_mic():
            try:
                import pyaudiowpatch as pyaudio
                mic_info = self._pa.get_default_input_device_info()
                mic_rate = int(mic_info.get("defaultSampleRate", 48000))
                mic_stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=mic_rate,
                    input=True,
                    input_device_index=int(mic_info["index"]),
                    frames_per_buffer=BLOCK_SIZE,
                )
                print(f"[Audio] Mikrofon uruchomiony: {mic_info.get('name', '?')}")
                while self._running:
                    try:
                        raw = mic_stream.read(BLOCK_SIZE, exception_on_overflow=False)
                    except Exception:
                        break
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    pcm = np.clip(pcm, -1.0, 1.0)
                    self._data_queue.put(pcm)
                try:
                    mic_stream.stop_stream()
                    mic_stream.close()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[Audio] Blad mikrofonu: {exc}")

        if self._capture_mode == "system":
            self._capture_thread = threading.Thread(target=_capture_loopback, daemon=True)
        elif self._capture_mode == "mic":
            self._capture_thread = threading.Thread(target=_capture_mic, daemon=True)
        else:  # both – run loopback; mic would need mixing, keep loopback as primary
            self._capture_thread = threading.Thread(target=_capture_loopback, daemon=True)

        self._capture_thread.start()
        self._process_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._process_thread.start()

    def _process_queue(self) -> None:
        if self._torch is not None:
            self._torch.set_num_threads(1)
        hangover_chunks = int(0.5 * self._sample_rate / BLOCK_SIZE)
        current_hangover = 0

        while self._running:
            try:
                audio = self._data_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Soft gain for quiet loopback streams so Whisper sees audible levels.
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if 0.0 < peak < 0.12:
                gain = min(4.0, 0.18 / peak)
                audio = np.clip(audio * gain, -1.0, 1.0)

            # Track RMS for WAV save log.
            power = float(np.mean(audio.astype(np.float64) ** 2))
            if np.isfinite(power) and power >= 0.0:
                self._last_rms = float(np.sqrt(power))

            if self._trigger_active:
                pcm16 = (audio * 32767).astype(np.int16).tobytes()
                self._frames.append(pcm16)

                speech_prob = 0.0
                if self._enable_vad_debug and self.model is not None and self._torch is not None:
                    try:
                        tensor = self._torch.from_numpy(audio)
                        speech_prob = self.model(tensor, self._sample_rate).item()
                    except Exception:
                        speech_prob = 0.0
                    if peak > 0.005:
                        print(f"[Audio] Glosnosc: {peak:.4f} | VAD: {speech_prob:.2f}")

                if speech_prob > SPEECH_PROB_THRESHOLD:
                    if not self.is_speech_active:
                        self.is_speech_active = True
                        print("[Audio] Wykryto mowe...")
                    current_hangover = hangover_chunks
                else:
                    if current_hangover > 0:
                        current_hangover -= 1
                    elif self.is_speech_active:
                        self.is_speech_active = False

    def set_trigger(self, active: bool) -> None:
        if active and not self._trigger_active:
            # Drain stale buffered data before starting a new recording.
            while not self._data_queue.empty():
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    break
            self._frames.clear()
            self.is_speech_active = False
        self._trigger_active = active

    def save_and_clear(self) -> str | None:
        self.set_trigger(False)
        wav_path = self._save_internal_buffer()
        self._frames.clear()
        return wav_path

    def stop(self) -> None:
        self._running = False
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _save_internal_buffer(self) -> str | None:
        if not self._frames:
            return None

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"stealth_audio_{int(time.time())}.wav")

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(b"".join(self._frames))

        total_bytes = sum(len(chunk) for chunk in self._frames)
        total_samples = total_bytes // 2
        duration_sec = total_samples / float(self._sample_rate) if total_samples else 0.0
        rms = self._last_rms if np.isfinite(self._last_rms) else 0.0
        print(
            f"[Audio] Zapisano WAV: {filepath} | czas={duration_sec:.2f}s | "
            f"zrodlo={self._active_source_name} | rms={rms:.4f}"
        )
        return filepath
