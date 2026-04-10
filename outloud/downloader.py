"""Audio downloader from any URL (YouTube, Vimeo, etc.) via yt-dlp."""

import os
import tempfile
from pathlib import Path

import yt_dlp

from outloud.config import YTDLP_TIMEOUT
from outloud.logger import get_logger
from outloud.exceptions import InvalidURLError, DownloadError, NetworkError

log = get_logger("downloader")


def get_video_info(url: str) -> dict:
    """Get media metadata without downloading.

    Args:
        url: Any URL supported by yt-dlp

    Returns:
        dict with title, duration_min, duration_sec
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": YTDLP_TIMEOUT,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown"),
                "duration_min": round(info.get("duration", 0) / 60),
                "duration_sec": info.get("duration", 0),
            }
    except yt_dlp.utils.DownloadError as e:
        if "unsupported" in str(e).lower():
            raise InvalidURLError(url)
        raise NetworkError("yt-dlp", str(e))
    except Exception as e:
        raise NetworkError("yt-dlp", str(e))


def download_audio(url: str, output_dir: Path) -> tuple[str, str]:
    """Download audio from any URL supported by yt-dlp.

    Args:
        url: Media URL (YouTube, Vimeo, Dailymotion, etc.)
        output_dir: Directory to save the WAV file

    Returns:
        (wav_path, title)

    Raises:
        InvalidURLError: If the URL is not supported
        DownloadError: If the download fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate URL
    if not url.startswith(("http://", "https://")):
        raise InvalidURLError(url)

    out_template = str(output_dir / "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": YTDLP_TIMEOUT,
    }

    # Suppress yt-dlp console output
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown")
            except yt_dlp.utils.DownloadError as e:
                raise DownloadError(url, str(e))
            except Exception as e:
                raise DownloadError(url, str(e))
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)

    # Find the downloaded WAV
    wav_path = str(output_dir / "audio.wav")
    if os.path.exists(wav_path):
        log.info("Audio downloaded: %s (%s)", wav_path, title)
        return wav_path, title

    raise DownloadError(url, "Output file not found after download")
