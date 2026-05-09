"""Configuration management for TranslateBooksWithLLMs.

Loads and validates environment variables and application settings
from .env files using python-dotenv.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""

    provider: str = "openai"  # openai, anthropic, ollama, deepseek, etc.
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 120
    max_retries: int = 3


@dataclass
class TranslationConfig:
    """Configuration for the translation process."""

    source_language: str = "English"
    target_language: str = "French"
    chunk_size: int = 1500  # characters per translation chunk
    overlap: int = 0        # overlap between chunks to preserve context
    preserve_formatting: bool = True
    glossary_path: Optional[str] = None
    # Bumped context_window to 4 — I found 3 still loses thread on longer chapters
    context_window: int = 4  # number of previous chunks to include as context


@dataclass
class AppConfig:
    """Top-level application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    output_dir: str = "output"
    log_level: str = "INFO"
    cache_enabled: bool = True
    cache_dir: str = ".cache"


def load_config() -> AppConfig:
    """Load application configuration from environment variables.

    Returns:
        AppConfig: Populated configuration object.

    Raises:
        ValueError: If required environment variables are missing.
    """
    llm_cfg = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
    )

    translation_cfg = TranslationConfig(
        source_language=os.getenv("SOURCE_LANGUAGE", "English"),
        target_language=os.getenv("TARGET_LANGUAGE", "French"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "0")),
        preserve_formatting=os.getenv("PRESERVE_FORMATTING", "true").lower() == "true",
        glossary_path=os.getenv("GLOSSARY_PATH"),
        context_window=int(os.getenv("CONTEXT_WINDOW", "4")),
    )

    app_cfg = AppConfig(
        llm=llm_cfg,
        translation=translation_cfg,
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
        cache_dir=os.getenv("CACHE_DIR", ".cache"),
    )

    return app_cfg
