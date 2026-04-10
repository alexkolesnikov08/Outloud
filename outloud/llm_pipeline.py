"""LLMPipeline — unified MLX inference for multiple models."""

import gc
import re

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

from outloud.config import LOCAL_LLM_MODELS
from outloud.logger import get_logger

log = get_logger("llm")

# ─── System Prompts ────────────────────────────────────────────────────────

SUMMARY_PROMPT_RU = """Ты — помощник студента. Сделай краткий конспект текста.

Формат (markdown):
## 📝 Конспект
- **Главная мысль:** одно предложение
- **Ключевые факты:** 3-5 пунктов
- **Вывод:** одно предложение

Правила:
- Пиши на русском
- Только факты, без воды
- Используй **жирный** для важных слов
- Не выдумывай ничего
- Не добавляй вступлений и заключений кроме формата выше"""

SUMMARY_PROMPT_EN = """You are a student's study assistant. Create brief lecture notes from the text.

Format (markdown):
## 📝 Study Notes
- **Main idea:** one sentence
- **Key points:** 3-5 bullets
- **Takeaway:** one sentence

Rules:
- Write in English
- Facts only, no filler
- Use **bold** for key terms
- Do not invent anything
- No extra text beyond the format above"""

GRAMMAR_PROMPT_RU = """Исправь грамматические и пунктуационные ошибки. НЕ меняй смысл. НЕ добавляй текст. Выведи ТОЛЬКО исправленный текст."""

GRAMMAR_PROMPT_EN = """Fix all grammatical and punctuation errors. Do NOT change the meaning. Do NOT add any text. Output ONLY the corrected text."""


class LLMPipeline:
    """Unified MLX pipeline for any 4-bit model."""

    def __init__(self, model_key: str):
        self.model_key = model_key
        self._model = None
        self._tokenizer = None
        self._language = None

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

    # ─── Language Detection ──────────────────────────────────────────────

    def _detect(self, text: str) -> str:
        """Detect language of the text."""
        if self._language:
            return self._language
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        total = sum(1 for c in text if c.isalpha())
        return "ru" if cyrillic > total * 0.3 else "en"

    # ─── Generation ──────────────────────────────────────────────────────

    def _run(self, messages: list[dict], max_tokens: int = 512,
             temp: float = 0.3, top_p: float = 0.9) -> str:
        """Run generation."""
        self._load()

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)

        sampler = make_sampler(temp=temp, top_p=top_p)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Clean up thinking tags
        result = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # If model started thinking but never closed — cut at </think>
        if not result and '</think>' in response:
            result = response.split('</think>', 1)[-1].strip()

        # If still empty or starts with <think> — take text after it
        if not result:
            # Find first non-</think> line
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('<') and len(line) > 10:
                    result = line
                    break

        return result

    # ─── Tasks ───────────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """Summarize text as student-style lecture notes."""
        if not text.strip():
            return ""

        words = text.split()
        if len(words) < 20:
            return text

        log.info("%s summarizing: %d words", self.model_key, len(words))
        lang = self._detect(text)

        # Small models (< 1B params) are bad at generation — use extractive
        model_size = self.model_key.split("-")[-1]  # "0.6b", "1b", "350m"
        is_small = False
        try:
            num = float(model_size.replace("b", "").replace("m", ""))
            is_small = num < 1.5
        except (ValueError, IndexError):
            pass

        if is_small:
            return self._format_extractive(text)

        # For larger models — direct LLM summary
        system_prompt = SUMMARY_PROMPT_RU if lang == "ru" else SUMMARY_PROMPT_EN

        # Truncate if too long
        max_chars = 4000
        if len(text) > max_chars:
            return self._summarize_long(text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text:\n\n{text}"}
        ]

        try:
            result = self._run(messages, max_tokens=300, temp=0.2)
            if result and len(result) > 20:
                log.info("LLM summary done")
                return result
        except Exception as e:
            log.warning("LLM summarization failed: %s", e)

        # Fallback
        return self._format_extractive(text)

    def _format_extractive(self, text: str) -> str:
        """Format extractive summary as markdown."""
        from outloud.summarizer import _split_sentences

        sentences = _split_sentences(text)
        if not sentences:
            return "## 📝 Конспект\n\nNo content to summarize."

        # Score sentences by position and key words
        scored = []
        for i, s in enumerate(sentences):
            score = 0
            # First and last sentences are important
            if i == 0:
                score += 3
            elif i == 1:
                score += 2
            elif i == len(sentences) - 1:
                score += 1
            # Longer sentences have more info
            score += min(len(s.split()) / 10, 2)
            scored.append((i, score, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Take top sentences
        n_take = max(3, len(sentences) // 4)
        top = sorted(scored[:n_take], key=lambda x: x[0])

        md = ["## 📝 Конспект", ""]
        md.append(f"**Текст:** {len(sentences)} предложений, {len(text.split())} слов")
        md.append("")

        # Main idea — first high-scored sentence
        md.append(f"> **Главная мысль:** {top[0][2]}")
        md.append("")

        # Key facts — rest
        if len(top) > 1:
            md.append("**Ключевые моменты:**")
            md.append("")
            for _, _, s in top[1:]:
                md.append(f"- {s}")
            md.append("")

        # Takeaway
        md.append(f"> **Итог:** {top[-1][2]}")

        return "\n".join(md)

    def _summarize_long(self, text: str) -> str:
        """Summarize long text in batches."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        batches = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 4000 and current:
                batches.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        if current:
            batches.append(current)

        log.info("Long text: %d batches", len(batches))

        partials = []
        for batch in batches:
            lang = self._detect(batch)
            system = SUMMARY_PROMPT_RU if lang == "ru" else SUMMARY_PROMPT_EN
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Текст:\n\n{batch}"}
            ]
            partials.append(self._run(messages, max_tokens=200, temp=0.2))

        if len(partials) > 1:
            combined = "\n".join(partials)
            lang = self._detect(combined)
            system = SUMMARY_PROMPT_RU if lang == "ru" else SUMMARY_PROMPT_EN
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Объедини:\n\n{combined}"}
            ]
            return self._run(messages, max_tokens=300, temp=0.2)

        return partials[0] if partials else ""

    def correct_grammar(self, text: str) -> str:
        """Fix grammar and punctuation."""
        if not text.strip():
            return text

        log.info("%s grammar correction: %d chars", self.model_key, len(text))

        # Small models are unreliable for grammar — use rule-based
        model_size = self.model_key.split("-")[-1]
        is_small = False
        try:
            num = float(model_size.replace("b", "").replace("m", ""))
            is_small = num < 1.5
        except (ValueError, IndexError):
            pass

        if is_small:
            return self._rule_based_grammar(text)

        lang = self._detect(text)
        system = GRAMMAR_PROMPT_RU if lang == "ru" else GRAMMAR_PROMPT_EN

        # Process in chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)

        batches = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 1000 and current:
                batches.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        if current:
            batches.append(current)

        corrected = []
        for batch in batches:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": batch}
            ]
            try:
                c = self._run(messages, max_tokens=len(batch.split()) * 2, temp=0.1)
                if c and len(c) > len(batch) * 0.5:  # At least 50% of original
                    corrected.append(c)
                else:
                    corrected.append(batch)
            except Exception:
                corrected.append(batch)

        result = ' '.join(corrected)
        return result

    def _rule_based_grammar(self, text: str) -> str:
        """Rule-based grammar correction for small models."""
        # Basic fixes
        rules = [
            (r'\b([а-яё])\s+\1\b', r'\1'),  # duplicate words
            (r'\s+,', ','),                   # space before comma
            (r'\s+\.', '.'),                   # space before period
            (r',\s*,', ','),                   # double commas
            (r'\.{2,}', '.'),                  # multiple periods
            (r'\s{2,}', ' '),                  # multiple spaces
        ]
        result = text
        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, result)

        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]

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
