"""
ai_service.py – LLM and AI logic integration.

Provides:
  • transcribe_audio()               – local faster-whisper transcription
  • analyze_screenshot()             – one-shot image analysis (GitHub Models via OpenAI)
  • analyze_screenshot_with_context() – image + audio text context (GitHub Models via OpenAI)
"""

import os
import base64
import time
import re
import gc
import sys
import importlib
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

# GitHub Models Token
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not _GITHUB_TOKEN:
    raise EnvironmentError(
        "GITHUB_TOKEN is missing in .env. "
        "Create or update a .env file with GITHUB_TOKEN=your_github_personal_access_token"
    )


def _env_float(name: str, default: float) -> float:
    """Read float env var safely with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    """Read string env var safely with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default

# Standard OpenAI client pointing to GitHub Models endpoint
_LLM_MODEL = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
_LLM_TIMEOUT_SEC = _env_float("LLM_TIMEOUT_SEC", 60.0)

_github_client = None
_openai_load_error = None


def _get_github_client():
    """Lazy-load OpenAI/httpx so app startup is not blocked on those imports."""
    global _github_client, _openai_load_error

    if _github_client is not None:
        return _github_client

    if _openai_load_error is not None:
        raise RuntimeError(_openai_load_error)

    try:
        from openai import OpenAI
        _github_client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=_GITHUB_TOKEN.strip(),
            timeout=_LLM_TIMEOUT_SEC,
            max_retries=1,
        )
        return _github_client
    except Exception as exc:
        _openai_load_error = f"Nie mozna zaladowac klienta OpenAI: {exc}"
        raise RuntimeError(_openai_load_error) from exc


# Local Whisper Client
# Using "small" model which is very fast and reasonably accurate for Polish.
# device="cpu" avoids CUDA issues on most machines.
# compute_type="int8" reduces memory usage drastically with minimal accuracy loss.
_WHISPER_MODEL_NAME = (os.getenv("WHISPER_MODEL") or "tiny").strip() or "tiny"
_WHISPER_PRIMARY_LANGUAGE = _env_str("WHISPER_PRIMARY_LANGUAGE", "pl").lower()
_WHISPER_SECONDARY_LANGUAGE = _env_str("WHISPER_SECONDARY_LANGUAGE", "en").lower()
_WHISPER_RESCUE_MODEL_NAME = _env_str("WHISPER_RESCUE_MODEL", "small").lower()
_WHISPER_MIN_CONFIDENCE = _env_float("WHISPER_MIN_CONFIDENCE", -1.05)
_whisper_model = None
_whisper_load_error = None
_whisper_rescue_model = None
_whisper_rescue_load_error = None


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    """Remove partially initialized modules from sys.modules."""
    keys = [
        name
        for name in list(sys.modules.keys())
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    ]
    for key in keys:
        sys.modules.pop(key, None)
    if keys:
        importlib.invalidate_caches()
        gc.collect()


def _is_torch_partial_import_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "partially initialized module 'torch'" in msg
        or "has no attribute 'autograd'" in msg
        or ("torch" in msg and "circular import" in msg)
        or ("winerror 1114" in msg and "torch" in msg)
        or ("c10.dll" in msg and "torch" in msg)
    )


def _get_whisper_model(allow_download: bool = True):
    """Lazy-load faster-whisper only when transcription is actually requested."""
    global _whisper_model, _whisper_load_error

    if _whisper_model is not None:
        return _whisper_model

    load_start = time.perf_counter()
    mode = "download" if allow_download else "cache-only"
    print(f"[Whisper] Inicjalizacja modelu '{_WHISPER_MODEL_NAME}' ({mode})...")
    for attempt in range(2):
        try:
            from faster_whisper import WhisperModel
            kwargs = {}
            if not allow_download:
                kwargs["local_files_only"] = True

            _whisper_model = WhisperModel(
                _WHISPER_MODEL_NAME,
                device="cpu",
                compute_type="int8",
                **kwargs,
            )
            _whisper_load_error = None
            print(f"[Whisper] Model gotowy po {time.perf_counter() - load_start:.1f}s.")
            return _whisper_model
        except KeyboardInterrupt as exc:
            # Interrupted imports can leave half-initialized modules in sys.modules.
            _purge_modules(("torch", "faster_whisper"))
            raise RuntimeError("Przerwano inicjalizacje modelu Whisper.") from exc
        except Exception as exc:
            if attempt == 0 and _is_torch_partial_import_error(exc):
                print("[Whisper] Wykryto uszkodzony import torch, czyszcze moduly i probuje ponownie...")
                _purge_modules(("torch", "faster_whisper"))
                continue

            if allow_download:
                _whisper_load_error = (
                    f"Nie mozna zaladowac faster-whisper: {exc}. "
                    "Sprobuj ponownie lub uruchom aplikacje na Python 3.11/3.12."
                )
                raise RuntimeError(_whisper_load_error) from exc

            raise RuntimeError(
                f"Model Whisper '{_WHISPER_MODEL_NAME}' nie jest jeszcze w lokalnym cache. "
                "Uruchamiam aplikacje bez preload i pobiore model przy pierwszej transkrypcji."
            ) from exc

    raise RuntimeError("Nieoczekiwany blad inicjalizacji Whisper.")


def _get_whisper_rescue_model():
    """Load optional stronger rescue model used only for low-confidence audio."""
    global _whisper_rescue_model, _whisper_rescue_load_error

    if _WHISPER_RESCUE_MODEL_NAME == _WHISPER_MODEL_NAME:
        return _get_whisper_model()

    if _whisper_rescue_model is not None:
        return _whisper_rescue_model

    if _whisper_rescue_load_error is not None:
        raise RuntimeError(_whisper_rescue_load_error)

    load_start = time.perf_counter()
    print(f"[Whisper] Inicjalizacja modelu ratunkowego '{_WHISPER_RESCUE_MODEL_NAME}'...")
    try:
        from faster_whisper import WhisperModel
        _whisper_rescue_model = WhisperModel(_WHISPER_RESCUE_MODEL_NAME, device="cpu", compute_type="int8")
        print(f"[Whisper] Model ratunkowy gotowy po {time.perf_counter() - load_start:.1f}s.")
        return _whisper_rescue_model
    except Exception as exc:
        _whisper_rescue_load_error = f"Nie mozna zaladowac modelu ratunkowego: {exc}"
        raise RuntimeError(_whisper_rescue_load_error) from exc


def preload_transcriber() -> tuple[bool, str]:
    """Preload whisper model proactively (ideally before Qt import)."""
    preload_mode = _env_str("WHISPER_PRELOAD_MODE", "local").lower()
    if preload_mode in {"0", "false", "off", "no", "disabled"}:
        return False, "Whisper preload pominięty (WHISPER_PRELOAD_MODE=off)."

    try:
        # Default preload mode is local cache only to avoid long startup/download stalls.
        # Real model download is still allowed later on first real transcription.
        allow_download = preload_mode in {"download", "online", "force"}
        _get_whisper_model(allow_download=allow_download)
        return True, f"Whisper model '{_WHISPER_MODEL_NAME}' zaladowany."
    except KeyboardInterrupt:
        return False, "Przerwano preload Whisper. Aplikacja wystartuje bez preload."
    except Exception as exc:
        return False, str(exc)


# ---- System & User Prompts ----
_SYSTEM_PROMPT = (
    "Jesteś inżynierem oprogramowania i asystentem podczas technicznej rozmowy rekrutacyjnej.\n"
    "Twoim zadaniem jest dostarczanie naturalnych, wyczerpujących i bardzo trafnych odpowiedzi "
    "na podstawie słów rekrutera (audio) oraz zrzutów ekranu.\n\n"
    "Zasady komunikacji:\n"
    "- ZERO wstępów. Przechodź od razu do meritum.\n"
    "- KOD: Pisz w TypeScript. Generuj kod czytelny i realistyczny – unikaj przerośniętych optymalizacji.\n"
    "- TŁUMACZENIE WYJAŚNIEŃ: Pamiętaj, aby podawać wyjaśnienia w odpowiednim języku i tonie, zależnie od typu polecenia.\n"
    "- FORMAT: Używaj krótkich akapitów i wypunktowań, aby tekst był przejrzysty na małym nakładanym oknie."
)

_USER_PROMPTS = {
    "hr": (
        "Rozmowa HR. Wygeneruj naturalną, płynną odpowiedź na zadane w rozmowie pytanie. "
        "BARDZO WAŻNA ZASADA ODNOŚNIE JĘZYKA: Twoja odpowiedź ZAWSZE musi być w języku, w jakim zostało zadane pytanie. "
        "Jeżeli rekruter zadał pytanie po POLSKU - odpowiedz po POLSKU. "
        "Jeżeli rekruter zmienił język i zadał pytanie po ANGIELSKU w celu sprawdzenia poziomu (lub cała rozmowa jest po angielsku) - odpowiedz po ANGIELSKU (poziom B2/C1). "
        "Odpowiedź powinna brzmieć autentycznie, pokazując motywację i umiejętności miękkie. "
        "BARDZO WAŻNE STANOWISKO - Odpowiadasz jako kandydat, o poniższym życiorysie (CV):\n"
        "Imię: Radosław Marek, Software Developer.\n"
        "Edukacja: Applied Computer Science na Uniwersytecie Jagiellońskim (w trakcie), technik programista (ZST im. Tadeusza Kościuszki w Leżajsku).\n"
        "Umiejętności: React, Next.js, TypeScript, Tailwind, C#, .NET, Python, n8n.\n"
        "Doświadczenie:\n"
        "1. Divstack (Software Developer): Automatyzacja n8n/Airtable/ClickUp API.\n"
        "2. DevCodi (Software Developer): Optymalizacja React UI/UX, integracje API (Google Maps, płatności), e-commerce.\n"
        "3. Cetuspro (Frontend Developer): Projekty B2B w React, rozwój od stażysty do członka core teamu.\n"
        "Projekty:\n"
        "1. GrindVibe: Full-Stack (React 19, .NET 9, EF Core), autoryzacja OAuth/JWT, Redux Toolkit.\n"
        "2. USOS Registration Bot: Rozszerzenie przeglądarki (JS Manifest V3) do automatycznej rejestracji na zajęcia z milisekundową precyzją pod dużym obciążeniem.\n"
        "3. Stellar Journey: Projekt 3D na NASA Space Apps 2024 używający Three.js i Reacta.\n"
        "Jeśli rekruter pyta o coś z CV, buduj spójną 'swoją' historię. Pamiętaj: to Twój życiorys."
    ),
    "technical": "Wyczerpująco wyjaśnij techniczne zagadnienie z pytania rekrutera. Zamiast krótkiej odpowiedzi, wyczerp temat – zacznij od najważniejszych kwestii, a następnie rozbuduj odpowiedź o detale merytoryczne. Używaj poprawnej terminologii IT, aby pokazać bardzo głębokie zrozumienie tematu (odpowiadaj po polsku).",
    "live_coding": "Live coding. Rozwiąż zadanie ze zdjęcia. Podaj kod w TypeScript NA SAMYM POCZĄTKU OPDOQIEDZI. Kod ma być napisany 'po ludzku' – nie musi być to absolutnie optymalny algorytm, ma być przede wszystkim czytelny, naturalny i wyglądający tak, jak na prawdziwej rozmowie o pracę. Następnie pod spodem wyjaśnij logikę działania po polsku, podaj złożoność czasową i pamięciową, oraz wskaż przypadki brzegowe."
}


# ---------------------------------------------------------------------------
# Transcribe Audio (faster-whisper local)
# ---------------------------------------------------------------------------

def _looks_low_quality_transcript(text: str) -> bool:
    """Heuristic filter for repetitive/url-like garbage transcripts."""
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    tokens = re.findall(r"[a-z0-9ąćęłńóśźż-]+(?:\.[a-z0-9-]+)*", stripped.lower(), flags=re.IGNORECASE)
    if len(tokens) < 3:
        return True

    domain_like_count = sum(1 for t in tokens if t.startswith("www") or "." in t)
    if domain_like_count >= 2 and (domain_like_count / len(tokens)) > 0.35:
        return True

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    most_common = max(counts.values()) if counts else 0
    if most_common >= 3 and (most_common / len(tokens)) > 0.55:
        return True

    unique_ratio = len(counts) / len(tokens)
    if len(tokens) >= 6 and unique_ratio < 0.35:
        return True

    return False


def _transcript_quality_score(text: str) -> float:
    """Simple text quality score used to compare fallback hypotheses."""
    if not text:
        return -1e9

    stripped = text.strip()
    if not stripped:
        return -1e9

    tokens = re.findall(r"[a-z0-9ąćęłńóśźż-]+(?:\.[a-z0-9-]+)*", stripped.lower(), flags=re.IGNORECASE)
    if not tokens:
        return -1e9

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    unique_ratio = len(counts) / len(tokens)
    most_common = max(counts.values()) if counts else 0
    repeat_ratio = (most_common / len(tokens)) if tokens else 1.0
    domain_ratio = sum(1 for t in tokens if t.startswith("www") or "." in t) / len(tokens)

    # Higher is better.
    return (
        len(stripped) * 0.25
        + len(tokens) * 1.0
        + unique_ratio * 18.0
        - repeat_ratio * 14.0
        - domain_ratio * 20.0
    )


def _combined_transcript_score(text: str, confidence: float) -> float:
    """Combine lexical quality with acoustic confidence for hypothesis selection."""
    conf = max(-2.0, min(0.0, confidence))
    return _transcript_quality_score(text) + conf * 12.0


def _is_bad_hypothesis(text: str, confidence: float) -> bool:
    return _looks_low_quality_transcript(text) or confidence < _WHISPER_MIN_CONFIDENCE

def transcribe_audio(wav_path: str) -> str:
    """Transcribe audio using faster-whisper tuned for lower latency."""
    try:
        whisper_model = _get_whisper_model()
    except Exception as exc:
        print(f"[Whisper] Blad inicjalizacji: {exc}")
        return ""

    start = time.perf_counter()

    primary_lang = _WHISPER_PRIMARY_LANGUAGE
    if primary_lang in {"", "none", "auto"}:
        primary_lang = None

    secondary_lang = _WHISPER_SECONDARY_LANGUAGE
    if secondary_lang in {"", "none", "auto"}:
        secondary_lang = None
    if secondary_lang == primary_lang:
        secondary_lang = None

    def _run_once(model, language, beam_size, best_of, vad_filter) -> tuple[str, float]:
        segments, _ = model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
        )
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        if segment_list:
            conf = sum(float(getattr(segment, "avg_logprob", -2.0)) for segment in segment_list) / len(segment_list)
        else:
            conf = -99.0
        return text, conf

    try:
        transcript, transcript_conf = _run_once(
            whisper_model,
            language=primary_lang,
            beam_size=1,
            best_of=1,
            vad_filter=True,
        )
        best_transcript = transcript
        best_conf = transcript_conf
        best_score = _combined_transcript_score(transcript, transcript_conf)

        # If first pass looks weak, run stronger decoding and optional rescue model.
        if _is_bad_hypothesis(transcript, transcript_conf):
            strong_primary, strong_primary_conf = _run_once(
                whisper_model,
                language=primary_lang,
                beam_size=5,
                best_of=5,
                vad_filter=False,
            )
            strong_primary_score = _combined_transcript_score(strong_primary, strong_primary_conf)
            if strong_primary_score > best_score:
                best_transcript = strong_primary
                best_conf = strong_primary_conf
                best_score = strong_primary_score
                print("[Whisper] Uzyto fallback transkrypcji (dokladniejszy tryb).")

            if secondary_lang is not None:
                strong_secondary, strong_secondary_conf = _run_once(
                    whisper_model,
                    language=secondary_lang,
                    beam_size=5,
                    best_of=5,
                    vad_filter=False,
                )
                strong_secondary_score = _combined_transcript_score(strong_secondary, strong_secondary_conf)
                # Secondary language should clearly win to replace primary preference.
                if strong_secondary_score > (best_score + 6.0):
                    best_transcript = strong_secondary
                    best_conf = strong_secondary_conf
                    best_score = strong_secondary_score
                    print(f"[Whisper] Uzyto fallback transkrypcji (jezyk: {secondary_lang}).")

            if _WHISPER_RESCUE_MODEL_NAME and _is_bad_hypothesis(best_transcript, best_conf):
                try:
                    rescue_model = _get_whisper_rescue_model()
                    rescue_primary, rescue_primary_conf = _run_once(
                        rescue_model,
                        language=primary_lang,
                        beam_size=5,
                        best_of=5,
                        vad_filter=False,
                    )
                    rescue_primary_score = _combined_transcript_score(rescue_primary, rescue_primary_conf)
                    if rescue_primary_score > best_score:
                        best_transcript = rescue_primary
                        best_conf = rescue_primary_conf
                        best_score = rescue_primary_score
                        print(f"[Whisper] Uzyto modelu ratunkowego: {_WHISPER_RESCUE_MODEL_NAME}.")

                    if secondary_lang is not None:
                        rescue_secondary, rescue_secondary_conf = _run_once(
                            rescue_model,
                            language=secondary_lang,
                            beam_size=5,
                            best_of=5,
                            vad_filter=False,
                        )
                        rescue_secondary_score = _combined_transcript_score(rescue_secondary, rescue_secondary_conf)
                        if rescue_secondary_score > (best_score + 6.0):
                            best_transcript = rescue_secondary
                            best_conf = rescue_secondary_conf
                            best_score = rescue_secondary_score
                            print(f"[Whisper] Uzyto modelu ratunkowego (jezyk: {secondary_lang}).")
                except Exception as rescue_exc:
                    print(f"[Whisper] Pominieto model ratunkowy: {rescue_exc}")

            transcript = best_transcript

        took = time.perf_counter() - start
        print(f"[Whisper] Transkrypcja zakonczona w {took:.1f}s. Dlugosc: {len(transcript.strip())} znakow.")
        return transcript.strip()
    except Exception as exc:
        print(f"[Whisper] Blad transkrypcji: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Screenshot analysis (one-shot, no audio context, via GitHub Models)
# ---------------------------------------------------------------------------

def analyze_screenshot(image_bytes: bytes, prompt_type: str = "live_coding") -> tuple[str, str]:
    """Send a screenshot to LLM via GitHub Models and return the analysis text."""
    start = time.perf_counter()
    github_client = _get_github_client()
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    selected_prompt = _USER_PROMPTS.get(prompt_type, _USER_PROMPTS["live_coding"])
    
    # Use with_raw_response to access HTTP headers
    raw_response = github_client.chat.completions.with_raw_response.create(
        model=_LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": selected_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ],
            }
        ]
    )
    
    headers = raw_response.headers
    rem_tokens = headers.get('x-ratelimit-remaining-tokens', '?')
    rem_reqs = headers.get('x-ratelimit-remaining-requests', '?')
    
    # Get the actual JSON content
    completion = raw_response.parse()
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        text = "Model nie zwrócił treści odpowiedzi."
    
    info_str = f"API Limit: {rem_reqs} req | {rem_tokens} tok"
    print(f"[LLM] Odpowiedz modelu {_LLM_MODEL} w {time.perf_counter() - start:.1f}s.")
    return text, info_str


# ---------------------------------------------------------------------------
# Screenshot + audio transcript context (via GitHub Models)
# ---------------------------------------------------------------------------

def analyze_screenshot_with_context(
    image_bytes: bytes, audio_transcript: str, prompt_type: str = "live_coding"
) -> tuple[str, str]:
    """Send screenshot + text context to LLM via GitHub Models for combined analysis."""
    start = time.perf_counter()
    github_client = _get_github_client()
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    selected_prompt = _USER_PROMPTS.get(prompt_type, _USER_PROMPTS["live_coding"])
    
    final_prompt = (
        f"{selected_prompt}\n\n"
        f"Oto dodatkowy kontekst / co powiedział rekruter (transkrypcja):\n"
        f"«{audio_transcript}»\n\n"
        "Rozwiąż to zadanie uwzględniając powyższy kontekst z rozmowy."
    )
    
    raw_response = github_client.chat.completions.with_raw_response.create(
        model=_LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ],
            }
        ]
    )
    
    headers = raw_response.headers
    rem_tokens = headers.get('x-ratelimit-remaining-tokens', '?')
    rem_reqs = headers.get('x-ratelimit-remaining-requests', '?')
    
    completion = raw_response.parse()
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        text = "Model nie zwrócił treści odpowiedzi."
    
    info_str = f"API Limit: {rem_reqs} req | {rem_tokens} tok"
    print(f"[LLM] Odpowiedz modelu {_LLM_MODEL} (z kontekstem audio) w {time.perf_counter() - start:.1f}s.")
    return text, info_str


def get_runtime_models() -> tuple[str, str]:
    """Return active LLM and Whisper model names for diagnostics/UI."""
    return _LLM_MODEL, _WHISPER_MODEL_NAME
