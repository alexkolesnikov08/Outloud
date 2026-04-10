"""Vosk transcription with language support."""

import json
import os

from vosk import Model, KaldiRecognizer
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from outloud.config import get_models_dir, VOSK_MODELS, LANG_TO_VOSK
from outloud.logger import get_logger
from outloud.exceptions import ModelNotFoundError, AudioError

log = get_logger("transcriber")

CHUNK_SIZE = 4000  # Samples per chunk


def _get_model_path(model_key: str) -> str:
    """Get the model path for a given key."""
    model_info = VOSK_MODELS.get(model_key)
    if not model_info:
        raise ModelNotFoundError(model_key, "vosk")
    model_dir = get_models_dir() / model_info["name"]
    return str(model_dir)


def check_vosk_model(model_key: str = "vosk-small-ru") -> bool:
    """Check if a Vosk model exists.

    Args:
        model_key: Model identifier (e.g. 'vosk-small-ru', 'vosk-small-en')
    """
    model_info = VOSK_MODELS.get(model_key)
    if not model_info:
        return False
    model_dir = get_models_dir() / model_info["name"]
    return (model_dir / "am" / "final.mdl").exists()


def download_vosk_model(model_key: str = "vosk-small-ru"):
    """Download a Vosk model.

    Args:
        model_key: Model identifier
    """
    import urllib.request
    import zipfile

    model_info = VOSK_MODELS.get(model_key)
    if not model_info:
        raise ModelNotFoundError(model_key, "vosk")

    url = model_info["url"]
    name = model_info["name"]

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already exists
    if (models_dir / name / "am" / "final.mdl").exists():
        log.info("Model %s already exists, skipping download", model_key)
        print(f"Model {model_key} already exists")
        return

    zip_path = models_dir / f"{name}.zip"

    print(f"Downloading {model_key} ({model_info['size']})...")
    log.info("Downloading %s from %s", model_key, url)

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)

    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(models_dir)
        zip_path.unlink()
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        raise AudioError(f"Failed to download model: {e}")
    finally:
        os.dup2(old_err, 2)
        os.close(devnull)

    log.info("Model %s downloaded", model_key)
    print(f"Model {model_key} ready")


def transcribe_vosk(audio, language: str = "ru") -> str:
    """Transcribe audio with Vosk.

    Args:
        audio: numpy array of int16 or path to audio file
        language: 'ru' or 'en'

    Returns:
        Transcribed text
    """
    model_key = LANG_TO_VOSK.get(language, "vosk-small-ru")
    model_path = _get_model_path(model_key)

    log.info("Loading Vosk model: %s (%s)", model_key, language)

    # Suppress Vosk logs
    import logging
    logging.getLogger("vosk").setLevel(logging.WARNING)

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)

    try:
        model = Model(model_path)
    except Exception as e:
        raise AudioError(f"Failed to load Vosk model: {e}")
    finally:
        os.dup2(old_err, 2)
        os.close(devnull)

    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    # Handle numpy array or file path
    if isinstance(audio, np.ndarray):
        audio_data = audio.tobytes()
    else:
        # Convert to WAV if not already
        from outloud.utils import convert_to_wav
        audio_path = audio if isinstance(audio, str) else str(audio)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            wav_path = convert_to_wav(audio_path)
            import wave
            wf = wave.open(wav_path, "rb")
            audio_data = wf.readframes(wf.getnframes())
            wf.close()
        except Exception as e:
            raise AudioError(f"Failed to load audio: {e}")

    # Process in chunks
    total = len(audio_data)
    if total == 0:
        raise AudioError("Audio data is empty")

    processed = 0
    result = []
    log.info("Transcribing %d samples (%.1fs)", total, total / 16000)

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[white]transcribing (vosk {language})"),
        BarColumn(bar_width=30),
        TextColumn("[white]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("work", total=total)

        for i in range(0, total, CHUNK_SIZE * 2):  # int16 = 2 bytes
            chunk = audio_data[i:i + CHUNK_SIZE * 2]
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                if "text" in res and res["text"]:
                    result.append(res["text"])
            processed += len(chunk)
            prog.update(task, completed=min(processed, total))

        # Final result
        res = json.loads(rec.FinalResult())
        if "text" in res and res["text"]:
            result.append(res["text"])

    text = " ".join(r for r in result if r)
    if not text:
        log.warning("No speech detected")
        print("Warning: no speech detected in audio")

    log.info("Vosk transcription done: %d words", len(text.split()))
    return text
