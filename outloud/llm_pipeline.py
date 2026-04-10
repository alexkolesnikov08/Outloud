"""LLMPipeline — unified MLX inference for multiple models."""

import gc
import re

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

from outloud.config import LOCAL_LLM_MODELS
from outloud.logger import get_logger

log = get_logger("llm")


class LLMPipeline:
    """Unified MLX pipeline for any 4-bit model."""

    def __init__(self, model_key: str):
        self.model_key = model_key
        self._model = None
        self._tokenizer = None

    @property
    def model_info(self) -> dict:
        """Get model configuration."""
        info = LOCAL_LLM_MODELS.get(self.model_key)
        if not info:
            raise ValueError(f"Unknown model: {self.model_key}")
        return info

    # ─── Loading ─────────────────────────────────────────────────────────

    def _load(self):
        """Lazy load model."""
        if self._model is not None:
            return

        mlx_name = self.model_info["mlx_name"]
        log.info("Loading %s (4-bit MLX)", mlx_name)
        self._model, self._tokenizer = load(mlx_name)
        log.info("%s loaded (%s)", self.model_key, self.model_info["size"])

    # ─── Generation ──────────────────────────────────────────────────────

    def _run(self, messages: list[dict], max_tokens: int = 512,
             temp: float = 0.3, top_p: float = 0.9) -> str:
        """Run generation."""
        self._load()

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)

        # Disable thinking for reasoning models
        kwargs = {}
        if "reasoning" in self.model_key.lower():
            kwargs["enable_thinking"] = False

        sampler = make_sampler(temp=temp, top_p=top_p)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

        return response.strip()

    # ─── Tasks ───────────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """Summarize text."""
        if not text.strip():
            return ""

        words = text.split()
        if len(words) < 20:
            return text

        log.info("%s summarizing: %d words", self.model_key, len(words))

        # Truncate if too long for small models
        max_chars = 6000
        if len(text) > max_chars:
            return self._summarize_long(text)

        messages = [
            {"role": "system", "content": (
                "Briefly summarize the text. 2-4 sentences. "
                "Write only the result, no filler words."
            )},
            {"role": "user", "content": f"Summarize briefly:\n\n{text}"}
        ]

        return self._run(messages, max_tokens=256)

    def _summarize_long(self, text: str) -> str:
        """Summarize long text in batches."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        batches = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 6000 and current:
                batches.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        if current:
            batches.append(current)

        log.info("Long text: %d batches", len(batches))

        partials = []
        for batch in batches:
            messages = [
                {"role": "system", "content": (
                    "Briefly summarize the text. 2-3 sentences. "
                    "Only the result."
                )},
                {"role": "user", "content": f"Summarize:\n\n{batch}"}
            ]
            partials.append(self._run(messages, max_tokens=200))

        if len(partials) > 1:
            combined = "\n".join(partials)
            messages = [
                {"role": "system", "content": (
                    "Combine these partial summaries into one. 2-4 sentences. "
                    "Only the result."
                )},
                {"role": "user", "content": f"Combine:\n\n{combined}"}
            ]
            return self._run(messages, max_tokens=256)

        return partials[0] if partials else ""

    def correct_grammar(self, text: str) -> str:
        """Fix grammar and punctuation."""
        if not text.strip():
            return text

        log.info("%s grammar correction: %d chars", self.model_key, len(text))

        sentences = re.split(r'(?<=[.!?])\s+', text)

        batches = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 1500 and current:
                batches.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        if current:
            batches.append(current)

        corrected = []
        for batch in batches:
            messages = [
                {"role": "system", "content": (
                    "Fix grammatical and punctuation errors. "
                    "DO NOT change the meaning. DO NOT add anything. "
                    "Output only the corrected text."
                )},
                {"role": "user", "content": f"Fix errors:\n\n{batch}"}
            ]
            c = self._run(messages, max_tokens=len(batch.split()) * 2)
            corrected.append(c)

        result = ' '.join(corrected)
        return result

    # ─── Cleanup ─────────────────────────────────────────────────────────

    def cleanup(self):
        """Free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        mx.clear_cache()
        gc.collect()
