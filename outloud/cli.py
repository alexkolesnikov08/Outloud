"""CLI interface for OutLoud."""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from outloud import __version__
from outloud.config import (
    get_output_dir,
    get_models_dir,
    detect_hardware,
    VOSK_MODELS,
    LOCAL_LLM_MODELS,
    model_exists,
)
from outloud.transcriber import transcribe_vosk, check_vosk_model, download_vosk_model
from outloud.summarizer import summarize_extractive
from outloud.cloud import (
    transcribe_cloud,
    summarize_cloud,
    correct_grammar_cloud,
    save_api_keys,
    check_keys,
    verify_keys,
)
from outloud.downloader import download_audio, get_video_info
from outloud.recorder import record_audio, save_audio
from outloud.router import ProviderRouter
from outloud.llm_pipeline import LLMPipeline
from outloud.logger import get_logger

console = Console()
log = get_logger("cli")


def _save(filename: str, content: str, folder: Path):
    """Save text to a file."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding="utf-8")


def _stats(start: float, text: str, summary: str, folder: Path):
    """Print processing stats."""
    secs = time.time() - start
    words = len(text.split())
    print()
    print(f"words: {words} | time: {secs:.0f}s | saved: {folder}")


@click.group()
@click.version_option(version=__version__)
def main():
    """OutLoud — record, transcribe, and summarize audio."""
    pass


# ─── setup ──────────────────────────────────────────────────────────────────

@main.command()
@click.option('--model', type=str, default=None,
              help='Specific model to download (e.g. vosk-small-en)')
@click.option('--all', 'download_all', is_flag=True, default=False,
              help='Download all available models')
def setup(model: str = None, download_all: bool = False):
    """Set up OutLoud (download models)."""
    hw = detect_hardware()
    print(f"OutLoud v{__version__}")
    print(f"  Chip: {hw['chip']}")
    print(f"  RAM: {hw['ram_gb']}GB")
    print(f"  Recommended LLM: {hw['ai_model']}")
    print()

    # ffmpeg check
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("ffmpeg: ready")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg: NOT FOUND — run: brew install ffmpeg")

    print()

    # Download specific model
    if model:
        _download_model(model)
        return

    # Download all
    if download_all:
        _setup_all_models()
        return

    # Default setup
    _setup_default_models()


def _download_model(model_key: str):
    """Download a single model by key."""
    if model_key in VOSK_MODELS:
        if model_exists(model_key, "vosk"):
            print(f"Model {model_key} already exists")
            return
        download_vosk_model(model_key)
    elif model_key in LOCAL_LLM_MODELS:
        if model_exists(model_key, "llm"):
            print(f"Model {model_key} already exists")
            return
        print(f"Downloading {model_key} ({LOCAL_LLM_MODELS[model_key]['size']})...")
        pipeline = LLMPipeline(model_key)
        pipeline._load()
        pipeline.cleanup()
        print(f"Model {model_key} ready")
    else:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(list(VOSK_MODELS.keys()) + list(LOCAL_LLM_MODELS.keys()))}")


def _setup_default_models():
    """Download default models based on hardware."""
    hw = detect_hardware()

    # Vosk Russian small (always)
    if not model_exists("vosk-small-ru", "vosk"):
        download_vosk_model("vosk-small-ru")
    else:
        print("Model vosk-small-ru already exists")

    # LLM based on hardware
    llm_key = hw["ai_model"]
    if not model_exists(llm_key, "llm"):
        print(f"Downloading {llm_key} ({LOCAL_LLM_MODELS[llm_key]['size']})...")
        pipeline = LLMPipeline(llm_key)
        pipeline._load()
        pipeline.cleanup()
        print(f"Model {llm_key} ready")
    else:
        print(f"Model {llm_key} already exists")

    print()
    print("Ready. Try: outloud record")


def _setup_all_models():
    """Download all available models."""
    print("Downloading all models (this will take a while)...")
    print()

    for key in VOSK_MODELS:
        if not model_exists(key, "vosk"):
            download_vosk_model(key)
        else:
            print(f"Vosk {key}: already exists")

    for key in LOCAL_LLM_MODELS:
        if not model_exists(key, "llm"):
            info = LOCAL_LLM_MODELS[key]
            print(f"Downloading {key} ({info['size']})...")
            pipeline = LLMPipeline(key)
            pipeline._load()
            pipeline.cleanup()
            print(f"Model {key} ready")
        else:
            print(f"LLM {key}: already exists")

    print()
    print("All models ready")


# ─── record ─────────────────────────────────────────────────────────────────

@main.command()
@click.option('--cloud', is_flag=True, default=False, help='Use cloud models')
@click.option('--grammar', is_flag=True, default=False, help='Fix grammar')
@click.option('--lang', type=click.Choice(['ru', 'en']), default=None,
              help='Language (auto-detect if not set)')
@click.option('--model', type=str, default=None,
              help='Custom local model path (GGUF/MLX)')
def record(cloud: bool, grammar: bool, lang: str = None, model: str = None):
    """Record from microphone → text → summary."""
    if cloud and not check_keys():
        print("Cloud not configured. Run: outloud cloud-setup")
        return

    folder = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = folder / f"outloud_{ts}"
    sess.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Recording
    print("Recording... (Ctrl+C to stop)")
    audio = record_audio()
    if len(audio) == 0:
        print("Recording failed — no audio captured")
        return

    save_audio(audio, str(sess / "audio.wav"))
    print(f"Audio saved: {sess / 'audio.wav'}")

    # Router
    router = ProviderRouter(language=lang or "ru", cloud=cloud)

    # Transcription
    try:
        text = router.transcribe(str(sess / "audio.wav"))
        _save("transcription.txt", text, sess)
        print("Transcription done")
    except Exception as e:
        print(f"Transcription failed: {e}")
        return

    # Auto-detect language from transcription
    if not lang:
        detected_lang = router.detect_language(text)
        router.language = detected_lang
        if detected_lang != "ru":
            print(f"Language detected: {detected_lang}")

    # Summarization
    try:
        summary = router.summarize(text)
        _save("summary.txt", summary, sess)
        print("Summary done")
    except Exception as e:
        print(f"Summarization failed: {e}")
        summary = text

    # Grammar
    if grammar:
        try:
            fixed = router.correct_grammar(summary)
            _save("corrected.txt", fixed, sess)
            final = fixed
            print("Grammar done")
        except Exception as e:
            print(f"Grammar failed: {e}")
            final = summary
    else:
        final = summary

    router.cleanup()
    _stats(start, text, final, sess)


# ─── file ───────────────────────────────────────────────────────────────────

@main.command("file")
@click.argument("filepath", type=click.Path(exists=True))
@click.option('--cloud', is_flag=True, default=False, help='Use cloud models')
@click.option('--grammar', is_flag=True, default=False, help='Fix grammar')
@click.option('--lang', type=click.Choice(['ru', 'en']), default=None,
              help='Language (auto-detect if not set)')
def transcribe_file(filepath: str, cloud: bool, grammar: bool, lang: str = None):
    """Process an audio file."""
    if cloud and not check_keys():
        print("Cloud not configured. Run: outloud cloud-setup")
        return

    folder = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = folder / f"outloud_{ts}"
    sess.mkdir(parents=True, exist_ok=True)
    start = time.time()

    router = ProviderRouter(language=lang or "ru", cloud=cloud)

    # Transcription
    try:
        text = router.transcribe(filepath)
        _save("transcription.txt", text, sess)
        print("Transcription done")
    except Exception as e:
        print(f"Transcription failed: {e}")
        return

    if not lang:
        detected_lang = router.detect_language(text)
        router.language = detected_lang
        if detected_lang != "ru":
            print(f"Language detected: {detected_lang}")

    # Summarization
    try:
        summary = router.summarize(text)
        _save("summary.txt", summary, sess)
        print("Summary done")
    except Exception as e:
        print(f"Summarization failed: {e}")
        summary = text

    if grammar:
        try:
            fixed = router.correct_grammar(summary)
            _save("corrected.txt", fixed, sess)
            final = fixed
            print("Grammar done")
        except Exception as e:
            print(f"Grammar failed: {e}")
            final = summary
    else:
        final = summary

    router.cleanup()
    _stats(start, text, final, sess)


# ─── url ────────────────────────────────────────────────────────────────────

@main.command("url")
@click.argument("url")
@click.option('--cloud', is_flag=True, default=False, help='Use cloud models')
@click.option('--grammar', is_flag=True, default=False, help='Fix grammar')
@click.option('--lang', type=click.Choice(['ru', 'en']), default=None,
              help='Language (auto-detect if not set)')
def process_url(url: str, cloud: bool, grammar: bool, lang: str = None):
    """Process audio from any URL (YouTube, Vimeo, etc.)."""
    if cloud and not check_keys():
        print("Cloud not configured. Run: outloud cloud-setup")
        return

    folder = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = folder / f"outloud_{ts}"
    sess.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Get info
    info = get_video_info(url)
    print(f"Title: {info['title']}")
    print(f"Duration: {info['duration_min']} min")

    # Download
    audio_path, title = download_audio(url, sess)
    print(f"Audio downloaded: {title}")

    router = ProviderRouter(language=lang or "ru", cloud=cloud)

    # Transcription
    try:
        text = router.transcribe(audio_path)
        _save("transcription.txt", text, sess)
        print("Transcription done")
    except Exception as e:
        print(f"Transcription failed: {e}")
        return

    if not lang:
        detected_lang = router.detect_language(text)
        router.language = detected_lang
        if detected_lang != "ru":
            print(f"Language detected: {detected_lang}")

    # Summarization
    try:
        summary = router.summarize(text)
        _save("summary.txt", summary, sess)
        print("Summary done")
    except Exception as e:
        print(f"Summarization failed: {e}")
        summary = text

    if grammar:
        try:
            fixed = router.correct_grammar(summary)
            _save("corrected.txt", fixed, sess)
            final = fixed
            print("Grammar done")
        except Exception as e:
            print(f"Grammar failed: {e}")
            final = summary
    else:
        final = summary

    router.cleanup()
    _stats(start, text, final, sess)


# ─── cloud-setup ────────────────────────────────────────────────────────────

@main.command("cloud-setup")
def cloud_setup():
    """Configure cloud API (Groq)."""
    print("OutLoud Cloud Setup")
    print()
    print("Get a free Groq key: https://console.groq.com/keys")
    print()
    print("Cloud models:")
    print("  Transcription: Whisper Large v3 Turbo")
    print("  Summary: GPT-OSS 20B → Qwen 32B → Llama 70B → Llama 8B")
    print("  Grammar: Llama 3.1 8B → Llama 4 Scout")
    print()

    key = input("Groq API key: ").strip()
    if not key:
        print("Key is required")
        return

    print()
    print("Saving...")
    save_api_keys(key)

    print("Verifying...")
    if verify_keys():
        print("Ready. Use: outloud record --cloud")
    else:
        print("Warning: verification failed. Check your key.")


# ─── cloud-status ───────────────────────────────────────────────────────────

@main.command("cloud-status")
def cloud_status():
    """Check cloud API key status."""
    if not check_keys():
        print("Cloud not configured. Run: outloud cloud-setup")
        return
    print("Keys: configured")
    if verify_keys():
        print("Status: OK")
    else:
        print("Status: error (check limits at Groq dashboard)")


# ─── models ─────────────────────────────────────────────────────────────────

@main.command("models")
def list_models():
    """List available models and their status."""
    print("Available models:")
    print()

    print("  Vosk (ASR):")
    for key, info in VOSK_MODELS.items():
        exists = model_exists(key, "vosk")
        status = "✓" if exists else "✗"
        print(f"    {status} {key:25s} ({info['lang']}, {info['size']})")

    print()
    print("  Local LLM (MLX 4-bit):")
    for key, info in LOCAL_LLM_MODELS.items():
        exists = model_exists(key, "llm")
        langs = ",".join(info["langs"])
        status = "✓" if exists else "✗"
        print(f"    {status} {key:25s} ({langs}, {info['size']})")

    print()
    print("Download a model: outloud setup --model <key>")


if __name__ == "__main__":
    main()
