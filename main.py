"""
main.py – Application entry point.

Boots the PyQt6 app, registers global hotkeys, and orchestrates:
  • Ctrl+Alt+S / [Rozwiąż z ekranu] → screenshot → GPT-4o → display
  • Ctrl+Alt+A / [Start/Stop]       → start audio recording / stop recording 
                                      → transcribe (Whisper) -> capture screen 
                                      → GPT-4o → display
"""

import os
import sys
import threading

import keyboard

# FIX: Import ai_service (which imports faster-whisper -> onnxruntime) BEFORE PyQt6
# This prevents a known silent crash (access violation) due to OpenMP/dll conflicts on Windows.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ai_service import (
    analyze_screenshot,
    analyze_screenshot_with_context,
    transcribe_audio
)

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from capture import capture_screen
from audio_capture import SystemAudioCapture
from gui import AssistantWindow


# ------------------------------------------------------------------
# Signal bridge  (thread-safe → GUI thread)
# ------------------------------------------------------------------

class _SignalBridge(QObject):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    show_loading = pyqtSignal()
    voice_active = pyqtSignal(bool)
    voice_text = pyqtSignal(str)
    audio_ready = pyqtSignal(str)  # Emitted when VAD detects silence


# ------------------------------------------------------------------
# REST API Workflow Manager
# ------------------------------------------------------------------

class _WorkflowManager:
    """Manages the audio recording and GPT-4o + Whisper pipeline."""

    def __init__(self, bridge: _SignalBridge, get_prompt_type_cb) -> None:
        self._bridge = bridge
        self._get_prompt_type = get_prompt_type_cb
        self._audio_capture = SystemAudioCapture(on_audio_ready_callback=self._on_vad_silence)
        self._bridge.audio_ready.connect(self._process_audio_and_solve)
        self._processing = threading.Lock()

    def _on_vad_silence(self, wav_path: str):
        """Callback from VAD background thread."""
        self._bridge.audio_ready.emit(wav_path)

    @property
    def is_recording(self) -> bool:
        return self._audio_capture.is_running

    def toggle_recording(self) -> None:
        if self._audio_capture.is_running:
            # Manually stopped
            wav_path = self._audio_capture._save_internal_buffer()
            self._audio_capture.stop()
            self._bridge.voice_active.emit(False)
            if wav_path:
                self._bridge.audio_ready.emit(wav_path)
        else:
            self._start()

    def solve_screen_only(self) -> None:
        """One-shot capture + solve without audio context."""
        if not self._processing.acquire(blocking=False):
            return

        prompt_type = self._get_prompt_type()
        def _worker() -> None:
            try:
                self._bridge.show_loading.emit()
                screenshot_bytes = capture_screen()
                answer = analyze_screenshot(screenshot_bytes, prompt_type)
                self._bridge.result_ready.emit(answer)
            except Exception as exc:
                self._bridge.error_occurred.emit(str(exc))
            finally:
                self._processing.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _start(self) -> None:
        self._audio_capture.start()
        self._bridge.voice_active.emit(True)

    def _process_audio_and_solve(self, wav_path: str) -> None:
        if not self._processing.acquire(blocking=False):
            return

        prompt_type = self._get_prompt_type()
        def _worker() -> None:
            try:
                # Ensure the UI reflects that listening stopped
                self._audio_capture.stop()
                self._bridge.voice_active.emit(False)
                
                self._bridge.show_loading.emit()
                
                # 1. Capture Screen immediately
                screenshot_bytes = capture_screen()

                # 2. Transcribe audio
                transcript = transcribe_audio(wav_path)
                self._bridge.voice_text.emit(f"🗣️ Rekruter: {transcript}\n\nAnalizuję kod...")
                
                # 3. LLM analysis
                answer = analyze_screenshot_with_context(screenshot_bytes, transcript, prompt_type)
                self._bridge.result_ready.emit(f"🗣️ Transkrypcja: {transcript}\n\n---\n{answer}")

            except Exception as exc:
                self._bridge.error_occurred.emit(str(exc))
            finally:
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                self._processing.release()

        threading.Thread(target=_worker, daemon=True).start()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = AssistantWindow()
    bridge = _SignalBridge()

    # --------------- signal connections ---------------
    bridge.show_loading.connect(
        lambda: (window.show(), window.set_loading(True))
    )
    bridge.result_ready.connect(
        lambda text: (window.set_loading(False), window.set_response(text))
    )
    bridge.error_occurred.connect(
        lambda err: (
            window.set_loading(False),
            window.set_response(f"❌ Błąd: {err}"),
        )
    )
    bridge.voice_active.connect(
        lambda active: (
            window.show() if active else None,
            window.set_voice_active(active),
            window._response_area.setPlainText("") if active else None,
        )
    )
    bridge.voice_text.connect(lambda text: window.append_voice_text(text))

    # Helpers
    def get_selected_prompt_type():
        text = window.prompt_combo.currentText()
        if text == "Rozmowa HR (Angielski)":
            return "hr_english"
        elif text == "Pytania Techniczne":
            return "technical"
        return "live_coding"

    # --------------- Workflow manager ---------------
    workflow = _WorkflowManager(bridge, get_prompt_type_cb=get_selected_prompt_type)

    # --------------- Solve from screen (Ctrl+Alt+S or button) ---------------
    def on_solve_screen() -> None:
        if workflow.is_recording:
            # If recording, treat the "Solve" command exactly like stopping the recording
            # so it transcribes what was said up to now and screenshots.
            workflow.toggle_recording()
        else:
            workflow.solve_screen_only()

    keyboard.add_hotkey("ctrl+alt+s", on_solve_screen, suppress=False)
    window.solve_screen_clicked.connect(on_solve_screen)

    # --------------- Voice toggle (Ctrl+Alt+A or button) ---------------
    def on_voice_toggle() -> None:
        workflow.toggle_recording()

    keyboard.add_hotkey("ctrl+alt+a", on_voice_toggle, suppress=False)
    window.voice_toggle_clicked.connect(on_voice_toggle)

    # --------------- Show window on start ---------------
    window.show()

    print("✅ Assistant (OpenAI) uruchomiony")
    print("   Ctrl+Alt+S  →  Zrzut ekranu + analiza kodu")
    print("   Ctrl+Alt+A  →  Włącz nasłuchiwanie, po czym naciśnij ponownie aby wysłać do AI (Tekst + Ekran)")
    print("   Używaj przycisków w oknie lub skrótów klawiszowych.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
