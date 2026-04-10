"""Vosk transcription."""

import json
import os

from vosk import Model, KaldiRecognizer
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from outloud.config import get_models_dir, VOSK_MODELS
from outloud.logger import get_logger

log = get_logger("transcriber")

CHUNK_SIZE = 4000  # Samples per chunk


def _get_model_path(size: str = "small") -> str:
    """Get the model path."""
    model_name = VOSK_MODELS[size]["name"]
    return str(get_models_dir() / model_name)


def check_vosk_model(size: str = "small") -> bool:
    """Check if the model exists."""
    model_path = _get_model_path(size)
    return os.path.exists(os.path.join(model_path, "am", "final.mdl"))


def download_vosk_model(size: str = "small"):
    """Download the Vosk model."""
    import urllib.request
    import zipfile

    model_info = VOSK_MODELS[size]
    url = model_info["url"]
    name = model_info["name"]

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    zip_path = models_dir / f"{name}.zip"

    print(f"Downloading {name} ({model_info['size']})...")

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)

    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(models_dir)
        zip_path.unlink()
    finally:
        os.dup2(old_err, 2)
        os.close(devnull)

    print(f"{name} downloaded")


def transcribe_vosk(audio) -> str:
    """Transcribe audio with Vosk.

    Args:
        audio: numpy array of int16 or path to audio file
    """
    model_path = _get_model_path("small")
    log.info("Loading Vosk model: %s", model_path)

    # Suppress Vosk logs
    import logging
    logging.getLogger("vosk").setLevel(logging.WARNING)

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)

    try:
        model = Model(model_path)
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
        wav_path = convert_to_wav(audio_path)
        import wave
        wf = wave.open(wav_path, "rb")
        audio_data = wf.readframes(wf.getnframes())
        wf.close()

    # Process in chunks
    total = len(audio_data)
    processed = 0
    result = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]transcribing (vosk)"),
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
    log.info("Vosk transcription done: %d words", len(text.split()))
    return text
