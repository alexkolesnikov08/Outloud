"""OutLoud Cloud — Groq API with timeouts and fallback."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI, APITimeoutError, APIStatusError
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from outloud.config import (
    CLOUD_WHISPER_MODEL,
    CLOUD_SUMMARY_MODELS,
    CLOUD_GRAMMAR_MODELS,
    API_TIMEOUT,
    RATE_LIMIT_RETRY_DELAY,
)
from outloud.logger import get_logger
from outloud.exceptions import (
    RateLimitError,
    QuotaExceededError,
    NetworkError,
    APIKeyError,
)

log = get_logger("cloud")

KEYS_FILE = Path.home() / ".outloud" / "api_keys.json"


# ─── Key Management ───────────────────────────────────────────────────────

def _ensure_keys_dir():
    """Ensure the keys directory exists."""
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)


def save_api_keys(groq_key: str):
    """Save API keys to ~/.outloud/api_keys.json."""
    _ensure_keys_dir()
    data = {
        "groq_api_key": groq_key,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(KEYS_FILE, 0o600)
    log.info("API keys saved")


def load_api_keys() -> Optional[dict]:
    """Load API keys."""
    if not KEYS_FILE.exists():
        return None
    with open(KEYS_FILE) as f:
        return json.load(f)


def check_keys() -> bool:
    """Check if API keys are configured."""
    return load_api_keys() is not None


def _get_client() -> OpenAI:
    """Get Groq OpenAI client with timeout."""
    keys = load_api_keys()
    if not keys:
        raise APIKeyError("Groq")
    return OpenAI(
        api_key=keys["groq_api_key"],
        base_url="https://api.groq.com/openai/v1",
        timeout=API_TIMEOUT,
    )


# ─── Error Classification ─────────────────────────────────────────────────

def _classify_error(err: Exception) -> Exception:
    """Classify an API error into a specific exception type."""
    msg = str(err).lower()

    if isinstance(err, APITimeoutError):
        return NetworkError("Groq", "request timed out")

    if isinstance(err, APIStatusError):
        status = getattr(err, "status_code", 0)
        if status == 429:
            return RateLimitError("Groq", RATE_LIMIT_RETRY_DELAY)
        if status == 401:
            return APIKeyError("Groq")
        if status in (500, 502, 503):
            return NetworkError("Groq", f"server error {status}")

    if any(k in msg for k in ["rate_limit", "too_many", "429"]):
        return RateLimitError("Groq", RATE_LIMIT_RETRY_DELAY)
    if any(k in msg for k in ["quota", "insufficient_quota"]):
        return QuotaExceededError("Groq")
    if any(k in msg for k in ["timeout", "timed out"]):
        return NetworkError("Groq", "request timed out")

    return err


# ─── Chat with Fallback ───────────────────────────────────────────────────

def _chat_with_fallback(client: OpenAI, model_list: list[str],
                        messages: list[dict], **kwargs) -> str:
    """Try models in order, move to next on rate limits."""
    last_error = None

    for model in model_list:
        try:
            log.info("Trying cloud model: %s", model)
            resp = client.chat.completions.create(
                model=model, messages=messages,
                timeout=API_TIMEOUT, **kwargs)
            return resp.choices[0].message.content.strip()

        except Exception as e:
            classified = _classify_error(e)
            last_error = classified

            if isinstance(classified, RateLimitError):
                log.warning("Rate limit on %s, trying next model", model)
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                continue
            if isinstance(classified, QuotaExceededError):
                log.error("Quota exceeded — trying next model")
                continue
            if isinstance(classified, APIKeyError):
                raise classified
            # Other errors — try next model
            log.warning("Model %s failed: %s", model, classified)
            continue

    # All cloud models exhausted
    log.error("All cloud models failed: %s", last_error)
    return ""


# ─── Chunked Transcription ────────────────────────────────────────────────

def _transcribe_chunks(client: OpenAI, audio_path: str) -> str:
    """Transcribe large audio file in chunks."""
    from pydub import AudioSegment
    import tempfile

    audio = AudioSegment.from_mp3(audio_path)
    chunk_ms = 300_000  # 5 min
    total_chunks = max(1, len(audio) // chunk_ms + (1 if len(audio) % chunk_ms else 0))

    log.info("Splitting large file into %d chunks", total_chunks)
    print(f"Large file — splitting into {total_chunks} chunks")

    parts = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        chunk_num = i // chunk_ms + 1

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[white]chunk {chunk_num}/{total_chunks}"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            transient=True,
        ) as prog:
            prog.add_task("work", total=None)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                chunk.export(tmp.name, format="mp3", bitrate="64k")
                try:
                    with open(tmp.name, "rb") as f:
                        result = client.audio.transcriptions.create(
                            model=CLOUD_WHISPER_MODEL,
                            file=f,
                            response_format="text",
                            language="ru",
                        )
                finally:
                    os.unlink(tmp.name)

            text = result.strip() if isinstance(result, str) else result.text.strip()
            parts.append(text)

    full_text = " ".join(parts)
    log.info("Chunked transcription done: %d words", len(full_text.split()))
    return full_text


# ─── Public API ───────────────────────────────────────────────────────────

def transcribe_cloud(audio_path: str) -> str:
    """Transcription: Whisper Large v3 Turbo.

    Automatically chunks files > 20MB.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    log.info("Cloud transcription: %s", audio_path)
    client = _get_client()

    # Check size
    file_size = os.path.getsize(audio_path)
    is_large = file_size > 20 * 1024 * 1024

    if is_large:
        return _transcribe_chunks(client, audio_path)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[white]transcribing (whisper-turbo)"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            transient=True,
        ) as prog:
            prog.add_task("work", total=None)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=CLOUD_WHISPER_MODEL,
                    file=f,
                    response_format="text",
                    language="ru",
                )

        text = result.strip() if isinstance(result, str) else result.text.strip()
        log.info("Cloud transcription done: %d words", len(text.split()))
        return text

    except Exception as e:
        classified = _classify_error(e)
        if "413" in str(e) or "too large" in str(e).lower():
            return _transcribe_chunks(client, audio_path)
        raise classified


def summarize_cloud(text: str) -> str:
    """Summarization: GPT-OSS 20B → Qwen 32B → Llama 70B → Llama 8B."""
    if not text.strip():
        return ""
    words = text.split()
    if len(words) < 20:
        return text

    log.info("Cloud summarizing: %d words", len(words))
    client = _get_client()

    messages = [
        {"role": "system", "content": (
            "You are a student's study assistant. Convert the text into lecture notes.\n"
            "Format (markdown):\n"
            "## 📝 Study Notes\n\n"
            "**Text:** brief overview\n\n"
            "> **Main idea:** one sentence\n\n"
            "**Key points:**\n"
            "- Point 1\n"
            "- Point 2\n"
            "- Point 3\n\n"
            "> **Takeaway:** one sentence\n\n"
            "Rules: facts only, no filler, same language as source."
        )},
        {"role": "user", "content": f"Create study notes:\n\n{text}"}
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]summarizing (cloud)"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        prog.add_task("work", total=None)
        result = _chat_with_fallback(
            client, CLOUD_SUMMARY_MODELS, messages,
            temperature=0.3, top_p=0.9, max_tokens=512)

    if result:
        log.info("Cloud summary done: %d words", len(result.split()))
        return result

    log.warning("All cloud summary models failed, falling back to local")
    print("Cloud unavailable — using local extractive summary")
    from outloud.summarizer import summarize_extractive
    return summarize_extractive(text)


def correct_grammar_cloud(text: str) -> str:
    """Grammar: Llama 3.1 8B → Llama 4 Scout."""
    if not text.strip():
        return text

    log.info("Cloud grammar correction: %d chars", len(text))
    client = _get_client()

    messages = [
        {"role": "system", "content": (
            "Fix grammatical and punctuation errors. "
            "DO NOT change the meaning. DO NOT add anything. "
            "Output only the corrected text."
        )},
        {"role": "user", "content": f"Fix errors:\n\n{text}"}
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]correcting grammar (cloud)"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        prog.add_task("work", total=None)
        result = _chat_with_fallback(
            client, CLOUD_GRAMMAR_MODELS, messages,
            temperature=0.2, top_p=0.9, max_tokens=len(text.split()) * 2)

    if result:
        log.info("Cloud grammar done: %d chars", len(result))
        return result

    log.warning("All cloud grammar models failed")
    print("Cloud unavailable — returning original text")
    return text


def verify_keys() -> bool:
    """Verify API keys are valid."""
    try:
        client = _get_client()
        models = client.models.list()
        return len(models.data) > 0
    except Exception as e:
        log.error("Key verification failed: %s", e)
        return False
