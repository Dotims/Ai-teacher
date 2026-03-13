"""
ai_service.py – OpenAI API integration.

Provides:
  • transcribe_audio()               – transcribes WAV file via whisper-1
  • analyze_screenshot()             – one-shot image analysis
  • analyze_screenshot_with_context() – image + audio transcript context
"""

import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

_API_KEY = os.getenv("OPENAI_API_KEY")
if _API_KEY:
    _API_KEY = _API_KEY.strip()

if not _API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is missing or empty. "
        "Create or update a .env file with OPENAI_API_KEY=your_key"
    )

client = OpenAI(api_key=_API_KEY)

# ---- GPT-4o system prompt ----
_SYSTEM_PROMPT = (
    "Jesteś ekspertem IT. Odpowiadasz na pytania rekrutera na podstawie "
    "zrzutu ekranu i transkrypcji pytania z audio. Podawaj optymalny kod i "
    "max 3 zwięzłe punkty o złożoności. Brak wstępów."
)

_MODEL_NAME = "gpt-4o"


# ---------------------------------------------------------------------------
# Transcribe Audio (whisper-1)
# ---------------------------------------------------------------------------

def transcribe_audio(wav_path: str) -> str:
    """Send WAV file to Whisper API and return transcription."""
    with open(wav_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f,
            language="pl" # Hint to Whisper for Polish interviews, can be changed
        )
    return response.text


# ---------------------------------------------------------------------------
# Screenshot analysis (one-shot, no audio context)
# ---------------------------------------------------------------------------

def analyze_screenshot(image_bytes: bytes) -> str:
    """Send a screenshot to GPT-4o and return the analysis text."""
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Przeanalizuj ten zrzut ekranu. Rozwiąż zadanie widoczne na ekranie."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ],
            }
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Screenshot + audio transcript context
# ---------------------------------------------------------------------------

def analyze_screenshot_with_context(
    image_bytes: bytes, audio_transcript: str
) -> str:
    """Send screenshot + audio transcript for combined GPT-4o analysis."""
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = (
        "Oto co powiedział rekruter:\n"
        f"«{audio_transcript}»\n\n"
        "Rozwiąż zadanie widoczne na ekranie uwzględniając to co właśnie powiedział."
    )
    
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ],
            }
        ]
    )
    return response.choices[0].message.content
