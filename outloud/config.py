"""Configuration for OutLoud."""

import os
import platform
import subprocess
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path.home() / "Desktop"

# ─── Vosk ASR models ────────────────────────────────────────────────────────

VOSK_MODELS = {
    # Russian
    "vosk-small-ru": {
        "name": "vosk-model-small-ru-0.22",
        "lang": "ru",
        "size": "70MB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
    },
    "vosk-medium-ru": {
        "name": "vosk-model-ru-0.42",
        "lang": "ru",
        "size": "800MB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
    },
    # English
    "vosk-small-en": {
        "name": "vosk-model-small-en-us-0.15",
        "lang": "en",
        "size": "50MB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    },
    "vosk-medium-en": {
        "name": "vosk-model-en-us-0.22",
        "lang": "en",
        "size": "1.6GB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
    },
}

# ─── Local LLM models (MLX 4-bit) ──────────────────────────────────────────

LOCAL_LLM_MODELS = {
    # Default summarization
    "qwen3-0.6b": {
        "mlx_name": "mlx-community/Qwen3-0.6B-4bit",
        "size": "400MB",
        "langs": ["ru", "en"],
        "task": "summary",
    },
    # Alternative summarization
    "gemma2-1b": {
        "mlx_name": "mlx-community/gemma-2-1b-it-4bit",
        "size": "800MB",
        "langs": ["ru", "en"],
        "task": "summary",
    },
    # Reasoning (more accurate, slower)
    "qwen3-1.8b-reasoning": {
        "mlx_name": "mlx-community/Qwen3-1.8B-4bit",
        "size": "1.2GB",
        "langs": ["ru", "en"],
        "task": "summary",
    },
    # English-only, very small
    "lmf2.5-350m": {
        "mlx_name": "mlx-community/LFM2-5-350M-4bit",
        "size": "250MB",
        "langs": ["en"],
        "task": "summary",
    },
}

# ─── Cloud models (Groq) ────────────────────────────────────────────────────

CLOUD_WHISPER_MODEL = "whisper-large-v3-turbo"

CLOUD_SUMMARY_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

CLOUD_GRAMMAR_MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# ─── Language detection ─────────────────────────────────────────────────────

LANG_TO_VOSK = {
    "ru": "vosk-small-ru",
    "en": "vosk-small-en",
}

LANG_TO_LLM = {
    "en": ["lmf2.5-350m", "qwen3-0.6b"],
    "ru": ["qwen3-0.6b", "gemma2-1b"],
}

# ─── Recording settings ─────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# ─── Timeouts ───────────────────────────────────────────────────────────────

API_TIMEOUT = 120  # seconds for cloud API calls
DOWNLOAD_TIMEOUT = 300  # seconds for model downloads
YTDLP_TIMEOUT = 120  # seconds for YouTube downloads
RATE_LIMIT_RETRY_DELAY = 5  # seconds before retry on 429

# ─── Helpers ─────────────────────────────────────────────────────────────────


def get_models_dir() -> Path:
    """Get the models directory."""
    return Path(__file__).parent.parent / "models"


def get_output_dir() -> Path:
    """Get the default output directory."""
    return OUTPUT_DIR


def model_exists(model_key: str, model_type: str = "vosk") -> bool:
    """Check if a model is already downloaded.

    Args:
        model_key: Model identifier (e.g. 'vosk-small-ru')
        model_type: 'vosk', 'llm', or 'custom'
    """
    if model_type == "custom":
        # Custom model — check if path exists
        return os.path.exists(model_key)

    if model_type == "vosk":
        model_info = VOSK_MODELS.get(model_key)
        if not model_info:
            return False
        model_dir = get_models_dir() / model_info["name"]
        return (model_dir / "am" / "final.mdl").exists()

    if model_type == "llm":
        model_info = LOCAL_LLM_MODELS.get(model_key)
        if not model_info:
            return False
        # MLX models stored in huggingface cache
        try:
            from mlx_lm import load
            model, tok = load(model_info["mlx_name"])
            del model, tok
            return True
        except Exception:
            return False

    return False


def detect_hardware() -> dict:
    """Detect hardware capabilities."""
    chip = platform.processor() or "unknown"
    try:
        mem = subprocess.check_output(
            "sysctl -n hw.memsize 2>/dev/null || echo 4294967296", shell=True
        ).decode().strip()
        mem_gb = int(mem) // (1024**3)
    except Exception:
        mem_gb = 4

    if mem_gb >= 16:
        ai_model = "qwen3-1.8b-reasoning"
    elif mem_gb >= 8:
        ai_model = "qwen3-0.6b"
    else:
        ai_model = "lmf2.5-350m"

    return {
        "chip": chip,
        "ram_gb": mem_gb,
        "ai_model": ai_model,
    }
