"""
audio_capture.py – System audio capture with Silero VAD.

Captures system audio via WASAPI Loopback, processes it through Silero VAD to detect speech,
and automatically triggers a callback with the saved WAV file after 2 seconds of silence.
"""

import os
import wave
import tempfile
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import torch

SAMPLE_RATE = 16000
CHANNELS = 1
# Silero VAD requires chunks of 512, 1024, or 1536 samples for 16kHz
BLOCK_SIZE = 512
DTYPE = "int16"

# VAD Configuration
SILENCE_DURATION_SEC = 2.0
SILENCE_CHUNKS_THRESHOLD = int((SILENCE_DURATION_SEC * SAMPLE_RATE) / BLOCK_SIZE)
SPEECH_PROB_THRESHOLD = 0.5


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
    """Captures system audio, uses Silero VAD, and auto-saves after silence."""

    def __init__(self, on_audio_ready_callback):
        self._frames = []
        self._stream = None
        self._running = False
        self.on_audio_ready = on_audio_ready_callback
        
        # VAD State
        self.is_speech_active = False
        self.audio_queue = queue.Queue()
        self._trigger_active = False
        
        # Load Silero VAD
        print("Ładowanie modelu Silero VAD...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
        print("Silero VAD załadowane.")

        self._process_thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start capturing system audio and VAD processing."""
        if self._running:
            return

        self._frames.clear()
        self.is_speech_active = False
        self._trigger_active = False
        device = _find_wasapi_loopback_device()

        def _audio_callback(indata, frames, time_info, status):
            if status:
                pass
            pcm = indata.copy()
            if pcm.dtype != np.int16:
                pcm = (pcm * 32767).astype(np.int16)
            
            # Put raw bytes and float32 array in queue for processing
            float32_audio = pcm.astype(np.float32) / 32767.0
            self.audio_queue.put((pcm.tobytes(), float32_audio))

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
            self._running = True
            
            self._process_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._process_thread.start()
            
            self._stream.start()
        except Exception as e:
            # Fallback output device
            print(f"Błąd uruchamiania loopback: {e}. Uruchamiam fallback.")
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=_audio_callback,
            )
            self._running = True
            
            self._process_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._process_thread.start()
            
            self._stream.start()

    def _process_queue(self):
        """Worker thread to process audio chunks through VAD."""
        torch.set_num_threads(1)
        
        # Hangover frames (e.g. 0.5 sec) to avoid cutting speech abruptly
        hangover_chunks = int(0.5 * SAMPLE_RATE / BLOCK_SIZE)
        current_hangover = 0

        while self._running:
            try:
                raw_bytes, float32_audio = self.audio_queue.get(timeout=0.1)
                
                if getattr(self, '_trigger_active', False):
                    # Convert to tensor for Silero VAD
                    tensor = torch.from_numpy(float32_audio).squeeze()
                    
                    # Get speech probability
                    speech_prob = self.model(tensor, SAMPLE_RATE).item()
                    
                    if speech_prob > SPEECH_PROB_THRESHOLD:
                        if not self.is_speech_active:
                            self.is_speech_active = True
                            print("🎤 Wykryto mowę w buforze...")
                        self._frames.append(raw_bytes)
                        current_hangover = hangover_chunks
                    else:
                        if current_hangover > 0:
                            self._frames.append(raw_bytes)
                            current_hangover -= 1
                        else:
                            if self.is_speech_active:
                                self.is_speech_active = False
                else:
                    # Drop chunks when trigger is inactive
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"VAD Error: {e}")

    def set_trigger(self, active: bool):
        """Enables or disables buffering logic from the VAD sensor."""
        if active and not self._trigger_active:
            # Drain queue to avoid stale audio from background buffer
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._frames.clear()
            self.is_speech_active = False
            
        self._trigger_active = active

    def save_and_clear(self) -> str | None:
        """Saves current buffer to WAV and clears it."""
        self.set_trigger(False)
        wav_path = self._save_internal_buffer()
        self._frames.clear()
        return wav_path

    def stop(self) -> None:
        """Stop capturing."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
            
    def _save_internal_buffer(self) -> str | None:
        """Save captured buffer to a temporary WAV file and return the path."""
        if not self._frames:
            return None

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"stealth_audio_{int(time.time())}.wav")

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self._frames))

        return filepath
