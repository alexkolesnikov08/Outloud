"""Custom exceptions for OutLoud."""


class OutLoudError(Exception):
    """Base exception for OutLoud."""
    pass


class ModelNotFoundError(OutLoudError):
    """Raised when a required model is not found."""

    def __init__(self, model_key: str, model_type: str = "unknown"):
        self.model_key = model_key
        self.model_type = model_type
        super().__init__(
            f"Model '{model_key}' ({model_type}) not found. "
            f"Run: outloud setup --model {model_key}"
        )


class DownloadError(OutLoudError):
    """Raised when a download fails."""

    def __init__(self, url: str, reason: str = "unknown"):
        self.url = url
        super().__init__(
            f"Failed to download: {url}\nReason: {reason}"
        )


class NetworkError(OutLoudError):
    """Raised when a network request fails."""

    def __init__(self, service: str, reason: str = "unknown"):
        self.service = service
        super().__init__(
            f"Network error ({service}): {reason}\n"
            f"Check your internet connection."
        )


class RateLimitError(OutLoudError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, service: str, retry_after: int = None):
        self.service = service
        self.retry_after = retry_after
        hint = f"\nRetry after {retry_after}s" if retry_after else ""
        super().__init__(
            f"Rate limit exceeded on {service}.{hint}\n"
            f"Switching to local models..."
        )


class QuotaExceededError(OutLoudError):
    """Raised when API quota is exhausted."""

    def __init__(self, service: str):
        self.service = service
        super().__init__(
            f"API quota exceeded on {service}.\n"
            f"Check your limits at the provider's dashboard."
        )


class AudioError(OutLoudError):
    """Raised when audio processing fails."""

    def __init__(self, reason: str):
        super().__init__(f"Audio error: {reason}")


class ConversionError(OutLoudError):
    """Raised when audio format conversion fails."""

    def __init__(self, input_path: str, target_format: str, reason: str):
        super().__init__(
            f"Failed to convert '{input_path}' to {target_format}: {reason}"
        )


class RecordingError(OutLoudError):
    """Raised when microphone recording fails."""

    def __init__(self, reason: str):
        super().__init__(f"Recording error: {reason}")


class InvalidURLError(OutLoudError):
    """Raised when a URL is not valid or not supported."""

    def __init__(self, url: str):
        super().__init__(
            f"Invalid or unsupported URL: {url}\n"
            f"Supported: YouTube, Vimeo, Dailymotion, and any site yt-dlp supports."
        )


class ConfigError(OutLoudError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, key: str, reason: str = ""):
        hint = f" ({reason})" if reason else ""
        super().__init__(f"Configuration error: '{key}' not set.{hint}")


class APIKeyError(ConfigError):
    """Raised when API key is missing or invalid."""

    def __init__(self, service: str):
        super().__init__(
            f"{service}_api_key",
            f"Run: outloud cloud-setup"
        )
