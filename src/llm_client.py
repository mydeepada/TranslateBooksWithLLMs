"""LLM client module for interacting with various language model APIs.

Supports OpenAI-compatible APIs, Anthropic, and local models via Ollama.
"""

import logging
from typing import Optional

import anthropic
import openai

from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for interacting with different LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on the configured provider."""
        provider = self.config.provider.lower()

        if provider == "anthropic":
            return anthropic.Anthropic(api_key=self.config.api_key)

        elif provider in ("openai", "openai_compatible", "ollama"):
            base_url = self.config.base_url or (
                "http://localhost:11434/v1" if provider == "ollama" else None
            )
            return openai.OpenAI(
                api_key=self.config.api_key or "ollama",
                base_url=base_url,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Translate text from source language to target language.

        Args:
            text: The text to translate.
            source_language: The language of the input text.
            target_language: The language to translate into.
            system_prompt: Optional custom system prompt override.

        Returns:
            The translated text as a string.
        """
        prompt = self._build_system_prompt(source_language, target_language, system_prompt)
        user_message = f"Translate the following text:\n\n{text}"

        logger.debug(
            "Sending translation request to %s (model: %s)",
            self.config.provider,
            self.config.model,
        )

        provider = self.config.provider.lower()

        if provider == "anthropic":
            return self._call_anthropic(prompt, user_message)
        else:
            return self._call_openai_compatible(prompt, user_message)

    def _build_system_prompt(
        self,
        source_language: str,
        target_language: str,
        override: Optional[str] = None,
    ) -> str:
        """Build the system prompt for translation."""
        if override:
            return override
        return (
            f"You are a professional literary translator. "
            f"Translate the provided text from {source_language} to {target_language}. "
            f"Preserve the original tone, style, and formatting. "
            # Added: explicitly ask the model not to translate proper nouns like character
            # names, place names, and titles — avoids awkward auto-translations.
            f"Do not translate proper nouns such as character names, place names, or titles. "
            # Personal note: also asking it to keep paragraph breaks intact — the default
            # behavior was collapsing them which made epub output look wrong.
            f"Preserve all paragraph breaks and blank lines exactly as they appear in the original text. "
            f"Output only the translated text, with no commentary or explanation."
        )
