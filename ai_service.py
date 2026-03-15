"""
ai_service.py – LLM and AI logic integration.

Provides:
  • transcribe_audio()               – local faster-whisper transcription
  • analyze_screenshot()             – one-shot image analysis (GitHub Models via OpenAI)
  • analyze_screenshot_with_context() – image + audio text context (GitHub Models via OpenAI)
"""

import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel

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

# Standard OpenAI client pointing to GitHub Models endpoint
github_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=_GITHUB_TOKEN.strip(),
)
_LLM_MODEL = "gpt-4o" # or "gpt-4o-mini"


# Local Whisper Client
# Using "small" model which is very fast and reasonably accurate for Polish.
# device="cpu" avoids CUDA issues on most machines.
# compute_type="int8" reduces memory usage drastically with minimal accuracy loss.
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")


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

def transcribe_audio(wav_path: str) -> str:
    """Transcribe audio using local faster-whisper model."""
    segments, info = whisper_model.transcribe(wav_path, language="pl")
    transcript = " ".join([segment.text for segment in segments])
    return transcript.strip()


# ---------------------------------------------------------------------------
# Screenshot analysis (one-shot, no audio context, via GitHub Models)
# ---------------------------------------------------------------------------

def analyze_screenshot(image_bytes: bytes, prompt_type: str = "live_coding") -> str:
    """Send a screenshot to LLM via GitHub Models and return the analysis text."""
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    selected_prompt = _USER_PROMPTS.get(prompt_type, _USER_PROMPTS["live_coding"])
    
    response = github_client.chat.completions.create(
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
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Screenshot + audio transcript context (via GitHub Models)
# ---------------------------------------------------------------------------

def analyze_screenshot_with_context(
    image_bytes: bytes, audio_transcript: str, prompt_type: str = "live_coding"
) -> str:
    """Send screenshot + text context to LLM via GitHub Models for combined analysis."""
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    selected_prompt = _USER_PROMPTS.get(prompt_type, _USER_PROMPTS["live_coding"])
    
    final_prompt = (
        f"{selected_prompt}\n\n"
        f"Oto dodatkowy kontekst / co powiedział rekruter (transkrypcja):\n"
        f"«{audio_transcript}»\n\n"
        "Rozwiąż to zadanie uwzględniając powyższy kontekst z rozmowy."
    )
    
    response = github_client.chat.completions.create(
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
    return response.choices[0].message.content
