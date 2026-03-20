"""
main.py – Application entry point.

Boots the PyQt6 app, registers global hotkeys, and orchestrates:
  • Ctrl+Alt+S / [Rozwiąż z ekranu] → screenshot → GPT-4o → display
  • Ctrl+Alt+A / [Start/Stop]       → start audio recording / stop recording 
                                                                            → transcribe (Whisper)
                                                                            → GPT-4o (transkrypcja-only) → display
"""

import os
import sys
import threading
import traceback
import time
import re
from collections import Counter

# Force UTF-8 for console output to avoid cp1250 UnicodeEncodeError on Windows
if not sys.stdout.encoding or sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import keyboard

# FIX: Import ai_service (which imports faster-whisper -> onnxruntime) BEFORE PyQt6
# This prevents a known silent crash (access violation) due to OpenMP/dll conflicts on Windows.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ai_service import (
    analyze_screenshot,
    analyze_transcript_only,
    get_runtime_models,
    preload_transcriber,
    set_llm_model,
    transcribe_audio
)

_preload_ok, _preload_msg = preload_transcriber()
if _preload_ok:
    print(f"[Startup] {_preload_msg}")
else:
    lowered = _preload_msg.lower()
    if "pomin" in lowered or "przerwano preload" in lowered:
        print(f"[Startup] {_preload_msg}")
    else:
        print(f"[Startup] Whisper preload warning: {_preload_msg}")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from capture import capture_screen
from audio_capture import SystemAudioCapture
from gui import AssistantWindow


# ------------------------------------------------------------------
# Signal bridge  (thread-safe → GUI thread)
# ------------------------------------------------------------------

class _SignalBridge(QObject):
    result_ready = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)
    show_loading = pyqtSignal()
    voice_active = pyqtSignal(bool)
    voice_text = pyqtSignal(str)
    audio_ready = pyqtSignal(str)  # Emitted when VAD detects silence


def _is_meaningful_transcript(text: str) -> bool:
    """Return True if transcript has enough useful linguistic content."""
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    alnum_count = sum(1 for ch in stripped if ch.isalnum())
    if alnum_count < 3:
        return False

    lowered = stripped.lower()
    # Keep domain-like tokens intact (e.g. www.example.com).
    token_pattern = r"[a-z0-9ąćęłńóśźż-]+(?:\.[a-z0-9-]+)*"
    tokens = re.findall(token_pattern, lowered, flags=re.IGNORECASE)
    if len(tokens) < 3:
        return False

    domain_like = [t for t in tokens if t.startswith("www") or "." in t]
    if len(domain_like) >= 2 and (len(domain_like) / len(tokens)) > 0.35:
        return False

    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count >= 3 and (most_common_count / len(tokens)) > 0.55:
        return False

    unique_ratio = len(counts) / len(tokens)
    if len(tokens) >= 6 and unique_ratio < 0.35:
        return False

    return True


def _compose_audio_panel(transcript: str, answer: str | None = None, fallback: bool = False) -> str:
    """Format transcript and answer into one readable panel."""
    cleaned_transcript = (transcript or "").strip()
    transcript_block = cleaned_transcript if cleaned_transcript else "_Brak rozpoznanej mowy._"

    if answer is None:
        answer_block = (
            "_Generuję odpowiedź na podstawie transkrypcji..._"
            if cleaned_transcript
            else "_Brak odpowiedzi, bo nie wykryto transkrypcji._"
        )
    else:
        answer_block = answer.strip() or "_Model nie zwrócił treści odpowiedzi._"

    if fallback or not cleaned_transcript:
        return (
            "⚠️ Brak transkrypcji audio\n\n"
            f"---\n\n"
            f"### Odpowiedź\n\n{answer_block}"
        )

    return (
        f"**Transkrypcja**\n{transcript_block}\n\n"
        f"---\n\n"
        f"### Odpowiedź\n\n{answer_block}"
    )


# ------------------------------------------------------------------
# REST API Workflow Manager
# ------------------------------------------------------------------

class _WorkflowManager:
    """Manages the audio recording and GPT-4o + Whisper pipeline."""

    def __init__(self, bridge: _SignalBridge, get_prompt_type_cb, get_lang_override_cb=None) -> None:
        self._bridge = bridge
        self._get_prompt_type = get_prompt_type_cb
        self._get_lang_override = get_lang_override_cb or (lambda: None)
        self._audio_capture = SystemAudioCapture(on_audio_ready_callback=None)
        self._bridge.audio_ready.connect(self._process_audio_and_solve)
        
        self._current_task_id = 0
        self._is_processing = False
        
        # Start continuous background capture
        self._audio_capture.start()

    @property
    def is_recording(self) -> bool:
        return getattr(self._audio_capture, '_trigger_active', False)

    def cancel_task(self) -> None:
        self._current_task_id += 1
        self._is_processing = False
        print("❌ Operacja AI anulowana przez użytkownika.")
        self._bridge.result_ready.emit("❌ Analiza anulowana.", "")
        
    def on_trigger_press(self) -> None:
        """Called when toggle key is pressed."""
        if self._is_processing:
            self.cancel_task()
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
            else:
                msg = "❌ Nie zebrano danych audio. Sprawdź loopback i spróbuj ponownie."
                print(msg)
                self._bridge.error_occurred.emit(msg)

    def on_trigger_release(self) -> None:
        """Left empty as we now use toggle on press."""
        pass

    def toggle_recording(self) -> None:
        """Fallback for GUI button toggle."""
        self.on_trigger_press()

    def solve_screen_only(self) -> None:
        """One-shot capture + solve without audio context."""
        if self._is_processing:
            self.cancel_task()
            return

        self._current_task_id += 1
        task_id = self._current_task_id
        self._is_processing = True

        prompt_type = self._get_prompt_type()
        forced_lang = self._get_lang_override()
        def _worker() -> None:
            started = time.perf_counter()
            try:
                if task_id != self._current_task_id: return
                self._bridge.show_loading.emit()
                
                print("📷 Przechwytywanie ekranu...")
                screenshot_bytes = capture_screen()
                
                if task_id != self._current_task_id: return
                print("🤖 Wysyłanie zapytania do modelu (bez audio)...")
                answer, info = analyze_screenshot(screenshot_bytes, prompt_type, forced_language=forced_lang)
                
                if task_id != self._current_task_id: return
                self._bridge.result_ready.emit(answer, info)
            except Exception as exc:
                if task_id == self._current_task_id:
                    traceback.print_exc()
                    self._bridge.error_occurred.emit(str(exc))
            finally:
                if task_id == self._current_task_id:
                    self._is_processing = False
                    print(f"✅ Zakończono tryb 'zrzut ekranu' w {time.perf_counter() - started:.1f}s")

        threading.Thread(target=_worker, daemon=True).start()

    def _process_audio_and_solve(self, wav_path: str) -> None:
        self._current_task_id += 1
        task_id = self._current_task_id
        self._is_processing = True

        prompt_type = self._get_prompt_type()
        forced_lang = self._get_lang_override()
        def _worker() -> None:
            started = time.perf_counter()
            try:
                if task_id != self._current_task_id: return
                self._bridge.voice_active.emit(False)
                self._bridge.show_loading.emit()
                self._bridge.voice_text.emit("⏳ Przetwarzam nagranie...")

                # 1. Transcribe audio (audio-only fast path)
                if task_id != self._current_task_id: return
                self._bridge.voice_text.emit(
                    "### Transkrypcja\n\n_Trwa transkrypcja audio (Whisper)..._\n\n---\n\n### Odpowiedź\n\n_Oczekiwanie na transkrypcję..._"
                )
                print("📝 Transkrypcja audio...")
                transcript = transcribe_audio(wav_path)
                print(f"[Whisper] Transkrypt: '{transcript}'")
                
                if task_id != self._current_task_id: return
                if transcript.strip():
                    self._bridge.voice_text.emit(_compose_audio_panel(transcript))
                    print("🤖 Wysyłanie zapytania do modelu (transkrypcja-only)...")
                    answer, info = analyze_transcript_only(transcript, prompt_type, forced_language=forced_lang)
                    response_text = _compose_audio_panel(transcript, answer=answer)
                else:
                    info = ""
                    response_text = _compose_audio_panel("", answer=None, fallback=True)
                
                if task_id != self._current_task_id: return
                self._bridge.result_ready.emit(response_text, info)

            except Exception as exc:
                if task_id == self._current_task_id:
                    traceback.print_exc()
                    self._bridge.error_occurred.emit(str(exc))
            finally:
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                if task_id == self._current_task_id:
                    self._is_processing = False
                    print(f"✅ Zakończono tryb audio w {time.perf_counter() - started:.1f}s")

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
        lambda text, info: (window.set_loading(False), window.set_response(text, info))
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
    workflow = _WorkflowManager(
        bridge,
        get_prompt_type_cb=get_selected_prompt_type,
        get_lang_override_cb=lambda: window.get_language_override(),
    )

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
    window.model_changed.connect(set_llm_model)

    # --------------- Show window on start ---------------
    window.show()
    llm_model, whisper_model = get_runtime_models()

    print("✅ Assistant (OpenAI) uruchomiony")
    print(f"🧠 Model odpowiedzi: {llm_model}")
    print(f"🎧 Model transkrypcji: {whisper_model}")
    print("   Numpad 1    →  Zrzut ekranu + analiza kodu")
    print("   Prawy Ctrl  →  Włącz/Wyłącz nagrywanie mowy (Toggle)")
    print("   Używaj przycisków w oknie lub skrótów klawiszowych.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
