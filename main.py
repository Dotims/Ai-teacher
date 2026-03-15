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
        self._audio_capture = SystemAudioCapture(on_audio_ready_callback=None)
        self._bridge.audio_ready.connect(self._process_audio_and_solve)
        self._processing = threading.Lock()
        
        # Start continuous background capture
        self._audio_capture.start()

    @property
    def is_recording(self) -> bool:
        return getattr(self._audio_capture, '_trigger_active', False)

    def on_trigger_press(self) -> None:
        """Called when toggle key is pressed."""
        if self._processing.locked():
            return
        if not self.is_recording:
            print("🔴 Nagrywanie rozpoczęte (Toggle ON)...")
            self._audio_capture.set_trigger(True)
            self._bridge.voice_active.emit(True)
        else:
            print("⏹️ Zakończono nagrywanie (Toggle OFF). Rozpoczynam przetwarzanie...")
            self._audio_capture.set_trigger(False)
            self._bridge.voice_active.emit(False)
            wav_path = self._audio_capture.save_and_clear()
            if wav_path:
                self._bridge.audio_ready.emit(wav_path)

    def on_trigger_release(self) -> None:
        """Left empty as we now use toggle on press, not push-to-talk."""
        pass

    def toggle_recording(self) -> None:
        """Fallback for GUI button toggle."""
        self.on_trigger_press()

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

    def _process_audio_and_solve(self, wav_path: str) -> None:
        if not self._processing.acquire(blocking=False):
            return

        prompt_type = self._get_prompt_type()
        def _worker() -> None:
            try:
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
        if text == "Rozmowa HR (PL/ENG)":
            return "hr"
        elif text == "Pytania Techniczne":
            return "technical"
        return "live_coding"

    # --------------- Workflow manager ---------------
    workflow = _WorkflowManager(bridge, get_prompt_type_cb=get_selected_prompt_type)

    # --------------- Solve from screen (Numpad 1 or button) ---------------
    def on_solve_screen() -> None:
        if workflow.is_recording:
            workflow.toggle_recording()
        else:
            workflow.solve_screen_only()

    keyboard.on_press_key(79, lambda e: on_solve_screen(), suppress=False) # 79 is scan code for Numpad 1
    window.solve_screen_clicked.connect(on_solve_screen)

    # --------------- Toggle Voice (Right Ctrl) ---------------
    # We want to use right ctrl as a toggle (Start / Stop)
    # Using strict scan code or name check to avoid Left Ctrl triggering it.
    def _is_right_ctrl(e) -> bool:
        return e.name == 'right ctrl' or e.scan_code == 285

    keyboard.on_press(lambda e: workflow.on_trigger_press() if _is_right_ctrl(e) else None, suppress=False)
    window.voice_toggle_clicked.connect(workflow.toggle_recording)

    # --------------- Show window on start ---------------
    window.show()

    print("✅ Assistant (OpenAI) uruchomiony")
    print("   Numpad 1    →  Zrzut ekranu + analiza kodu")
    print("   Prawy Ctrl  →  Włącz/Wyłącz nagrywanie mowy (Toggle)")
    print("   Używaj przycisków w oknie lub skrótów klawiszowych.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
