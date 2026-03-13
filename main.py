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
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from capture import capture_screen
from audio_capture import SystemAudioCapture
from ai_service import (
    analyze_screenshot,
    analyze_screenshot_with_context,
    transcribe_audio
)
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


# ------------------------------------------------------------------
# REST API Workflow Manager
# ------------------------------------------------------------------

class _WorkflowManager:
    """Manages the audio recording and GPT-4o + Whisper pipeline."""

    def __init__(self, bridge: _SignalBridge) -> None:
        self._bridge = bridge
        self._audio_capture = SystemAudioCapture()
        self._processing = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._audio_capture.is_running

    def toggle_recording(self) -> None:
        if self._audio_capture.is_running:
            self._stop_and_solve()
        else:
            self._start()

    def solve_screen_only(self) -> None:
        """One-shot capture + solve without audio context."""
        if not self._processing.acquire(blocking=False):
            return

        def _worker() -> None:
            try:
                self._bridge.show_loading.emit()
                screenshot_bytes = capture_screen()
                answer = analyze_screenshot(screenshot_bytes)
                self._bridge.result_ready.emit(answer)
            except Exception as exc:
                self._bridge.error_occurred.emit(str(exc))
            finally:
                self._processing.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _start(self) -> None:
        self._audio_capture.start()
        self._bridge.voice_active.emit(True)

    def _stop_and_solve(self) -> None:
        if not self._processing.acquire(blocking=False):
            return

        def _worker() -> None:
            wav_path = None
            try:
                self._bridge.show_loading.emit()
                
                # Stop recording and save loopback to WAV
                self._audio_capture.stop()
                self._bridge.voice_active.emit(False)
                wav_path = self._audio_capture.save_wav()

                if wav_path:
                    # 1. Capture Screen immediately exact state when stopped
                    screenshot_bytes = capture_screen()

                    # 2. Transcribe audio
                    transcript = transcribe_audio(wav_path)
                    self._bridge.voice_text.emit(f"🗣️ Rekruter: {transcript}\n\nAnalizuję kod...")
                    
                    # 3. GPT-4o analysis
                    answer = analyze_screenshot_with_context(screenshot_bytes, transcript)
                    self._bridge.result_ready.emit(f"🗣️ Rekruter: {transcript}\n\n---\n{answer}")
                else:
                    self._bridge.error_occurred.emit("Brak nagrania audio.")

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

    # --------------- Workflow manager ---------------
    workflow = _WorkflowManager(bridge)

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
