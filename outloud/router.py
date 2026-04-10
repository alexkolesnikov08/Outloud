"""ProviderRouter — unified model selection with fallback chains."""

import time
from typing import Optional, Callable

from outloud.config import (
    VOSK_MODELS,
    LOCAL_LLM_MODELS,
    CLOUD_SUMMARY_MODELS,
    CLOUD_GRAMMAR_MODELS,
    CLOUD_WHISPER_MODEL,
    LANG_TO_VOSK,
    LANG_TO_LLM,
    RATE_LIMIT_RETRY_DELAY,
    API_TIMEOUT,
    model_exists,
    get_models_dir,
)
from outloud.exceptions import (
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    NetworkError,
    APIKeyError,
)
from outloud.logger import get_logger

log = get_logger("router")


class ProviderRouter:
    """Routes requests between local and cloud providers with fallback."""

    def __init__(self, language: str = "ru", cloud: bool = False):
        self.language = language
        self.cloud = cloud
        self._cloud_client = None
        self._local_pipeline = None

    # ─── ASR (Speech-to-Text) ────────────────────────────────────────────

    def get_asr_model(self) -> str:
        """Get the best ASR model key for the detected language."""
        return LANG_TO_VOSK.get(self.language, "vosk-small-ru")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio — cloud or local based on mode.

        Args:
            audio_path: Path to WAV audio file

        Returns:
            Transcribed text
        """
        if self.cloud:
            return self._transcribe_cloud(audio_path)
        return self._transcribe_local(audio_path)

    def _transcribe_local(self, audio_path: str) -> str:
        """Local Vosk transcription."""
        model_key = self.get_asr_model()

        if not model_exists(model_key, "vosk"):
            raise ModelNotFoundError(model_key, "vosk")

        from outloud.transcriber import transcribe_vosk
        return transcribe_vosk(audio_path)

    def _transcribe_cloud(self, audio_path: str) -> str:
        """Cloud Whisper transcription with retry."""
        from outloud.cloud import transcribe_cloud
        return transcribe_cloud(audio_path)

    # ─── Summarization ───────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """Summarize text — cloud or local with fallback chain."""
        if self.cloud:
            return self._summarize_cloud(text)
        return self._summarize_local(text)

    def _summarize_local(self, text: str) -> str:
        """Local LLM summarization with fallback."""
        model_keys = LANG_TO_LLM.get(self.language, ["qwen3-0.6b"])

        for key in model_keys:
            if not model_exists(key, "llm"):
                log.debug("Skipping %s (not downloaded)", key)
                continue

            try:
                return self._run_local_summary(key, text)
            except Exception as e:
                log.warning("Model %s failed: %s", key, e)
                continue

        # All local models failed — extractive fallback
        log.info("All LLMs failed, falling back to extractive summary")
        from outloud.summarizer import summarize_extractive
        return summarize_extractive(text)

    def _run_local_summary(self, model_key: str, text: str) -> str:
        """Run a single local model."""
        pipeline = self._get_local_pipeline(model_key)
        return pipeline.summarize(text)

    def _summarize_cloud(self, text: str) -> str:
        """Cloud summarization with fallback chain."""
        from outloud.cloud import summarize_cloud
        return summarize_cloud(text)

    # ─── Grammar Correction ──────────────────────────────────────────────

    def correct_grammar(self, text: str) -> str:
        """Correct grammar — cloud or local."""
        if self.cloud:
            return self._correct_grammar_cloud(text)
        return self._correct_grammar_local(text)

    def _correct_grammar_local(self, text: str) -> str:
        """Local grammar correction."""
        model_keys = LANG_TO_LLM.get(self.language, ["qwen3-0.6b"])

        for key in model_keys:
            if not model_exists(key, "llm"):
                continue

            try:
                pipeline = self._get_local_pipeline(key)
                return pipeline.correct_grammar(text)
            except Exception as e:
                log.warning("Grammar model %s failed: %s", key, e)
                continue

        log.info("No local grammar model available, returning original text")
        return text

    def _correct_grammar_cloud(self, text: str) -> str:
        """Cloud grammar correction."""
        from outloud.cloud import correct_grammar_cloud
        return correct_grammar_cloud(text)

    # ─── Language Detection ──────────────────────────────────────────────

    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection based on character analysis.

        Returns: 'ru' or 'en'
        """
        if not text:
            return "ru"

        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin = sum(1 for c in text if c.isalpha())

        if cyrillic > latin * 0.3:
            return "ru"
        return "en"

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _get_local_pipeline(self, model_key: str):
        """Get or create local MLX pipeline for the given model."""
        if self._local_pipeline is not None and self._local_pipeline.model_key == model_key:
            return self._local_pipeline

        from outloud.llm_pipeline import LLMPipeline
        self._local_pipeline = LLMPipeline(model_key)
        self._local_pipeline._language = self.language
        return self._local_pipeline

    def cleanup(self):
        """Release model resources."""
        if self._local_pipeline is not None:
            self._local_pipeline.cleanup()
            self._local_pipeline = None
