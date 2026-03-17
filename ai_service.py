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


def _env_int(name: str, default: int) -> int:
    """Read int env var safely with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read bool env var safely with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default

# Standard OpenAI client pointing to GitHub Models endpoint
_LLM_MODEL = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
_LLM_TIMEOUT_SEC = _env_float("LLM_TIMEOUT_SEC", 60.0)
_LLM_MAX_TOKENS = max(200, _env_int("LLM_MAX_TOKENS", 700))

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
_WHISPER_FAST_MODE = _env_bool("WHISPER_FAST_MODE", True)
_WHISPER_ENABLE_FALLBACK = _env_bool("WHISPER_ENABLE_FALLBACK", not _WHISPER_FAST_MODE)
_WHISPER_ENABLE_SECONDARY_FALLBACK = _env_bool("WHISPER_ENABLE_SECONDARY_FALLBACK", False)
_WHISPER_ENABLE_RESCUE = _env_bool("WHISPER_ENABLE_RESCUE", False)
_WHISPER_INITIAL_VAD_FILTER = _env_bool("WHISPER_INITIAL_VAD_FILTER", True)
_WHISPER_STRONG_BEAM_SIZE = max(1, _env_int("WHISPER_STRONG_BEAM_SIZE", 3))
_WHISPER_STRONG_BEST_OF = max(1, _env_int("WHISPER_STRONG_BEST_OF", 3))
_WHISPER_USE_INITIAL_PROMPT = _env_bool("WHISPER_USE_INITIAL_PROMPT", True)
_WHISPER_INITIAL_PROMPT = _env_str(
    "WHISPER_INITIAL_PROMPT",
    "React, React Native, Virtual DOM, TypeScript, JavaScript, Next.js, Node.js, Redux, GraphQL, hooks, components, state, props",
)
_WHISPER_TECH_TERM_REWRITE = _env_bool("WHISPER_TECH_TERM_REWRITE", True)
_WHISPER_MIXED_QUICK_RETRY = _env_bool("WHISPER_MIXED_QUICK_RETRY", True)
_whisper_model = None
_whisper_load_error = None
_whisper_rescue_model = None
_whisper_rescue_load_error = None


_TECH_TERMS = (
    "react",
    "react native",
    "virtual",
    "virtual dom",
    "typescript",
    "javascript",
    "next.js",
    "node.js",
    "redux",
    "graphql",
    "zustand",
    "hooks",
    "component",
    "components",
)


_PHONETIC_TECH_REPLACEMENTS = (
    (r"\bwitu[\s-]*al\b", "virtual"),
    (r"\bwirtu[\s-]*al\b", "virtual"),
    (r"\bwiertu[\s-]*al\b", "virtual"),
    (r"\bri?akt\b", "React"),
    (r"\bri?akt[\s-]*nativ[e]?\b", "React Native"),
    (r"\btaip[\s-]*skript\b", "TypeScript"),
    (r"\btype[\s-]*skript\b", "TypeScript"),
    (r"\btypeskript\b", "TypeScript"),
    (r"\bjava[\s-]*skript\b", "JavaScript"),
    (r"\bd(?:z|ż)awa[\s-]*skript\b", "JavaScript"),
    (r"\bnext[\s-]*(?:js|d(?:z|ż)e?js)\b", "Next.js"),
    (r"\bnekst[\s-]*(?:js|d(?:z|ż)e?js)\b", "Next.js"),
    (r"\bnode[\s-]*(?:js|d(?:z|ż)e?js)\b", "Node.js"),
    (r"\bnod[\s-]*(?:js|d(?:z|ż)e?js)\b", "Node.js"),
    (r"\bgrap[\s-]*(?:kiu?el|ql)\b", "GraphQL"),
    (r"\bredaks\b", "Redux"),
)


_PHONETIC_TECH_HINTS = (
    r"\bwitu[\s-]*al\b",
    r"\bwirtu[\s-]*al\b",
    r"\bri?akt\b",
    r"\btaip[\s-]*skript\b",
    r"\bjava[\s-]*skript\b",
    r"\bd(?:z|ż)e?js\b",
)


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

            # In cache-only preload mode, missing model files can still trigger
            # partial imports of heavy native modules (torch/numpy/ctranslate2).
            # Purge them so later regular imports (e.g. audio capture) start clean.
            _purge_modules(("faster_whisper", "ctranslate2", "torch", "numpy"))

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
    "- JĘZYK: Używaj wyłącznie polskiego albo angielskiego. Jeśli pytanie / transkrypcja / treść na ekranie jest po polsku, odpowiadaj po polsku. Jeśli jest po angielsku, odpowiadaj po angielsku. Nie mieszaj języków.\n"
    "- KOD: Pisz w TypeScript. Generuj kod czytelny i realistyczny – unikaj przerośniętych optymalizacji.\n"
    "- TŁUMACZENIE WYJAŚNIEŃ: Pamiętaj, aby podawać wyjaśnienia w odpowiednim języku i tonie, zależnie od typu polecenia.\n"
    "- FORMAT: Używaj krótkich akapitów i wypunktowań, aby tekst był przejrzysty na małym nakładanym oknie."
)


def _detect_prompt_language(text: str) -> str | None:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None

    tokens = re.findall(r"[a-ząćęłńóśźż']+", lowered, flags=re.IGNORECASE)
    if not tokens:
        return None

    polish_words = {
        "jak", "dlaczego", "co", "czy", "jakie", "opowiedz", "proszę", "projekt", "doświadczenie",
        "slabe", "mocne", "strony", "rekruter", "pytanie", "zespole", "pracowales", "czym",
        "rozni", "roznica", "wyjasnij", "odpowiedz", "miedzy", "poniewaz", "pomocy", "polsku",
        "jest", "sa", "oraz", "ktory", "ktore", "dlatego", "wlasnie",
    }

    english_words = {
        "what", "why", "how", "tell", "describe", "project", "experience", "team", "question",
        "weaknesses", "weakness", "strengths", "strength", "your", "about", "worked", "implement",
        "while", "uses", "using", "use", "based", "over", "under", "through", "between", "difference",
        "from", "than", "with", "without", "when", "where", "which", "this", "that", "these", "those",
        "english", "answer", "explain", "compared",
    }

    english_tech_words = {
        "rest", "json", "http", "grpc", "protocol", "protocols", "buffer", "buffers", "protobuf",
        "virtual", "dom", "react", "native", "typescript", "javascript", "next", "node", "redux", "graphql",
    }

    polish_hits = sum(token in polish_words for token in tokens)
    english_hits = sum(token in english_words for token in tokens)
    english_tech_hits = sum(token in english_tech_words for token in tokens)

    polish_diacritics = sum(ch in lowered for ch in "ąćęłńóśźż")
    polish_score = polish_hits * 2 + polish_diacritics * 2
    english_score = english_hits * 2 + english_tech_hits * 3

    if polish_score == 0 and english_score == 0:
        return None

    if english_score >= polish_score + 2:
        return "en"
    if polish_score >= english_score + 2:
        return "pl"

    # Tie-breakers for mixed transcript with technical English terms.
    if english_tech_hits >= 2 and polish_diacritics == 0:
        return "en"
    if polish_diacritics >= 2 and english_tech_hits == 0:
        return "pl"
    if english_hits > polish_hits:
        return "en"
    if polish_hits > english_hits:
        return "pl"
    return None


def _language_directive(context_text: str = "") -> str:
    detected = _detect_prompt_language(context_text)
    if detected == "pl":
        return "Odpowiedz wyłącznie po polsku."
    if detected == "en":
        return "Answer only in English using natural spoken interview style (roughly B2/C1), clear and authentic."
    return (
        "Odpowiadaj wyłącznie w języku pytania. "
        "Jeśli pytanie jest po polsku, odpowiedz po polsku. "
        "If the question is in English, answer in natural spoken English (B2/C1). "
        "Używaj tylko polskiego albo angielskiego."
    )


_HR_COMPANY_NOTES = (
    "Kontekst firmy Software Mansion (uzyj tylko gdy pasuje do pytania):\n"
    "- Firma laczy software house i mocny open source dla devow.\n"
    "- Bardzo silna pozycja w React Native / Expo (np. reanimated, gesture-handler, screens).\n"
    "- Druga mocna domena: multimedia i streaming (Elixir + Membrane Framework).\n"
    "- Czlonkostwo w React Foundation (2026) i aktywny udzial w New Architecture React Native.\n"
    "- Kultura: spolecznosc, dzielenie sie wiedza, wydarzenia i meetupy.\n"
    "- W praktyce cenia mindset: smart, curious, versatile oraz solidne podstawy inzynierskie (algorytmy, struktury danych, live coding)."
)

_HR_PERSONAL_NOTES = (
    "Notatki o kandydacie (parafrazuj, nie cytuj 1:1):\n"
    "- Motywacja: rozwoj przez realne projekty zespolowe podczas wakacyjnego stazu i chec dalszej wspolpracy.\n"
    "- Inspiracja po spotkaniu meet.js i rozmowach z ludzmi z branzy.\n"
    "- Trzon stacku: React, TypeScript, JavaScript; dodatkowo Next.js, GraphQL, Zustand, Tailwind, shadcn/ui.\n"
    "- Rozwoj: React Native, React Query, Redux.\n"
    "- Odpowiedzi maja brzmiec naturalnie i ustnie, bez przesadnie ksiazkowego stylu."
)


def _contextual_hr_directive(prompt_type: str, transcript: str = "") -> str:
    if prompt_type != "hr":
        return ""

    lang = _detect_prompt_language(transcript)
    if lang == "en":
        style_line = "For English answers, keep a spoken B2/C1 level, 4-8 sentences, confident but natural."
    elif lang == "pl":
        style_line = "Dla odpowiedzi po polsku zachowaj naturalny, mówiony styl i konkret."
    else:
        style_line = (
            "Keep one language only: choose the dominant language from transcript terms. "
            "If dominant is English, answer in spoken B2/C1 English; otherwise use natural Polish."
        )

    return (
        "Dodatkowe instrukcje HR:\n"
        "- Nie kopiuj gotowych odpowiedzi slowo w slowo. Parafrazuj i rotuj argumenty.\n"
        "- W jednej odpowiedzi wspomnij maksymalnie 1-2 elementy z notatek firmy, tylko gdy sa trafne.\n"
        "- Pokaz swiadomosc firmy i motywacji, ale bez nadmiernego strategicznego tonu.\n"
        f"- {style_line}\n\n"
        f"{_HR_COMPANY_NOTES}\n\n"
        f"{_HR_PERSONAL_NOTES}"
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
    "technical": "Wyczerpująco wyjaśnij techniczne zagadnienie z pytania rekrutera. Zamiast krótkiej odpowiedzi, wyczerp temat – zacznij od najważniejszych kwestii, a następnie rozbuduj odpowiedź o detale merytoryczne. Używaj poprawnej terminologii IT, pokaż głębokie zrozumienie tematu i odpowiadaj w tym samym języku, w którym padło pytanie: po polsku dla polskiego pytania, po angielsku dla angielskiego.",
    "live_coding": "Live coding. Rozwiąż zadanie ze zdjęcia. Podaj kod w TypeScript na samym początku odpowiedzi. Kod ma być napisany naturalnie i czytelnie, jak podczas prawdziwej rozmowy o pracę. Następnie pod spodem wyjaśnij logikę działania, podaj złożoność czasową i pamięciową oraz wskaż przypadki brzegowe. Część opisową napisz w tym samym języku, w którym jest pytanie: po polsku dla polskiego pytania, po angielsku dla angielskiego."
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


def _count_tech_terms(text: str) -> int:
    lowered = (text or "").lower()
    if not lowered:
        return 0
    return sum(1 for term in _TECH_TERMS if term in lowered)


def _normalize_tech_terms(text: str) -> tuple[str, int]:
    """Normalize common phonetic renderings of technical EN words in PL speech."""
    if not text:
        return "", 0

    if not _WHISPER_TECH_TERM_REWRITE:
        return text, 0

    updated = text
    changes = 0
    for pattern, replacement in _PHONETIC_TECH_REPLACEMENTS:
        updated, n = re.subn(pattern, replacement, updated, flags=re.IGNORECASE)
        changes += n

    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    updated = re.sub(r"\s{2,}", " ", updated).strip()
    return updated, changes


def _needs_mixed_quick_retry(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _PHONETIC_TECH_HINTS)


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
    tech_hits = _count_tech_terms(stripped)

    # Higher is better.
    return (
        len(stripped) * 0.25
        + len(tokens) * 1.0
        + unique_ratio * 18.0
        - repeat_ratio * 14.0
        - domain_ratio * 20.0
        + tech_hits * 4.0
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
            initial_prompt=(_WHISPER_INITIAL_PROMPT if _WHISPER_USE_INITIAL_PROMPT else None),
        )
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        text, term_fix_count = _normalize_tech_terms(text)
        if term_fix_count:
            print(f"[Whisper] Korekta slownictwa technicznego: {term_fix_count}.")
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
            vad_filter=_WHISPER_INITIAL_VAD_FILTER,
        )
        best_transcript = transcript
        best_conf = transcript_conf
        best_score = _combined_transcript_score(transcript, transcript_conf)

        if _WHISPER_MIXED_QUICK_RETRY and _needs_mixed_quick_retry(transcript):
            retry_transcript, retry_conf = _run_once(
                whisper_model,
                language=None,
                beam_size=1,
                best_of=1,
                vad_filter=_WHISPER_INITIAL_VAD_FILTER,
            )
            retry_score = _combined_transcript_score(retry_transcript, retry_conf)
            if retry_score > (best_score + 1.5):
                best_transcript = retry_transcript
                best_conf = retry_conf
                best_score = retry_score
                transcript = retry_transcript
                transcript_conf = retry_conf
                print("[Whisper] Uzyto szybkiego retry auto-jezyk dla slow technicznych.")

        # Fast profile: if first pass returned any text, skip heavy fallback passes.
        if _WHISPER_FAST_MODE and transcript.strip():
            took = time.perf_counter() - start
            print(f"[Whisper] Fast mode: pomijam fallback. Czas {took:.1f}s.")
            return best_transcript.strip()

        # If first pass looks weak, run stronger decoding and optional rescue model.
        if _WHISPER_ENABLE_FALLBACK and _is_bad_hypothesis(transcript, transcript_conf):
            strong_primary, strong_primary_conf = _run_once(
                whisper_model,
                language=primary_lang,
                beam_size=_WHISPER_STRONG_BEAM_SIZE,
                best_of=_WHISPER_STRONG_BEST_OF,
                vad_filter=False,
            )
            strong_primary_score = _combined_transcript_score(strong_primary, strong_primary_conf)
            if strong_primary_score > best_score:
                best_transcript = strong_primary
                best_conf = strong_primary_conf
                best_score = strong_primary_score
                print("[Whisper] Uzyto fallback transkrypcji (dokladniejszy tryb).")

            if _WHISPER_ENABLE_SECONDARY_FALLBACK and secondary_lang is not None:
                strong_secondary, strong_secondary_conf = _run_once(
                    whisper_model,
                    language=secondary_lang,
                    beam_size=_WHISPER_STRONG_BEAM_SIZE,
                    best_of=_WHISPER_STRONG_BEST_OF,
                    vad_filter=False,
                )
                strong_secondary_score = _combined_transcript_score(strong_secondary, strong_secondary_conf)
                # Secondary language should clearly win to replace primary preference.
                if strong_secondary_score > (best_score + 6.0):
                    best_transcript = strong_secondary
                    best_conf = strong_secondary_conf
                    best_score = strong_secondary_score
                    print(f"[Whisper] Uzyto fallback transkrypcji (jezyk: {secondary_lang}).")

            if _WHISPER_ENABLE_RESCUE and _WHISPER_RESCUE_MODEL_NAME and _is_bad_hypothesis(best_transcript, best_conf):
                try:
                    rescue_model = _get_whisper_rescue_model()
                    rescue_primary, rescue_primary_conf = _run_once(
                        rescue_model,
                        language=primary_lang,
                        beam_size=_WHISPER_STRONG_BEAM_SIZE,
                        best_of=_WHISPER_STRONG_BEST_OF,
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
                            beam_size=_WHISPER_STRONG_BEAM_SIZE,
                            best_of=_WHISPER_STRONG_BEST_OF,
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
    extra_context = _contextual_hr_directive(prompt_type)
    final_prompt = f"{_language_directive()}\n\n{selected_prompt}"
    if extra_context:
        final_prompt += f"\n\n{extra_context}"
    
    # Use with_raw_response to access HTTP headers
    raw_response = github_client.chat.completions.with_raw_response.create(
        model=_LLM_MODEL,
        temperature=0.0,
        max_tokens=_LLM_MAX_TOKENS,
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
    
    # Get the actual JSON content
    completion = raw_response.parse()
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        text = "Model nie zwrócił treści odpowiedzi."
    
    info_str = f"API Limit: {rem_reqs} req | {rem_tokens} tok"
    print(f"[LLM] Odpowiedz modelu {_LLM_MODEL} w {time.perf_counter() - start:.1f}s.")
    return text, info_str


# ---------------------------------------------------------------------------
# Transcript-only analysis (fast audio path, no screenshot)
# ---------------------------------------------------------------------------

def analyze_transcript_only(audio_transcript: str, prompt_type: str = "live_coding") -> tuple[str, str]:
    """Send only transcript text to LLM for fastest voice-first response."""
    start = time.perf_counter()
    github_client = _get_github_client()

    transcript = (audio_transcript or "").strip()
    if not transcript:
        return "Brak transkrypcji mowy.", ""

    selected_prompt = _USER_PROMPTS.get(prompt_type, _USER_PROMPTS["live_coding"])
    language_directive = _language_directive(transcript)
    extra_context = _contextual_hr_directive(prompt_type, transcript)

    final_prompt = (
        f"{language_directive}\n\n"
        f"{selected_prompt}\n\n"
        f"{extra_context}\n\n"
        "Pytanie rekrutera (transkrypcja audio):\n"
        f"«{transcript}»\n\n"
        "Odpowiedz wyłącznie na podstawie tej transkrypcji."
    )

    raw_response = github_client.chat.completions.with_raw_response.create(
        model=_LLM_MODEL,
        temperature=0.0,
        max_tokens=_LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
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
    print(f"[LLM] Odpowiedz modelu {_LLM_MODEL} (transkrypcja-only) w {time.perf_counter() - start:.1f}s.")
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
    language_directive = _language_directive(audio_transcript)
    extra_context = _contextual_hr_directive(prompt_type, audio_transcript)
    
    final_prompt = (
        f"{language_directive}\n\n"
        f"{selected_prompt}\n\n"
        f"{extra_context}\n\n"
        f"Oto dodatkowy kontekst / co powiedział rekruter (transkrypcja):\n"
        f"«{audio_transcript}»\n\n"
        "Rozwiąż to zadanie uwzględniając powyższy kontekst z rozmowy."
    )
    
    raw_response = github_client.chat.completions.with_raw_response.create(
        model=_LLM_MODEL,
        temperature=0.0,
        max_tokens=_LLM_MAX_TOKENS,
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
