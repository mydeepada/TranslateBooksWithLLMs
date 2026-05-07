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
            f"Output only the translated text without any additional commentary."
        )

    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Call the Anthropic API."""
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_openai_compatible(self, system_prompt: str, user_message: str) -> str:
        """Call an OpenAI-compatible API (OpenAI, Ollama, etc.)."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create an LLM client from config."""
    return LLMClient(config)
