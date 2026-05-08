"""Core translation pipeline for processing book files."""

import os
import time
import logging
from pathlib import Path
from typing import Optional

from src.config import AppConfig
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


class Translator:
    """Handles the end-to-end translation of book files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".epub"}

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = LLMClient(config.llm)
        self._translated_chunks: list[str] = []

    def translate_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Translate a book file and write the result to disk.

        Args:
            input_path: Path to the source file.
            output_path: Optional destination path. If omitted, a path is
                         derived from ``input_path`` and the target language.

        Returns:
            The path where the translated file was written.
        """
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        ext = source.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if output_path is None:
            target_lang = self.config.translation.target_language.lower()
            output_path = str(
                source.parent / f"{source.stem}_{target_lang}{source.suffix}"
            )

        logger.info("Reading source file: %s", input_path)
        text = self._read_file(source)

        chunks = self._split_into_chunks(
            text, self.config.translation.chunk_size
        )
        logger.info("Split into %d chunk(s) for translation.", len(chunks))

        self._translated_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("Translating chunk %d / %d …", idx, len(chunks))
            translated = self._translate_chunk_with_retry(chunk)
            self._translated_chunks.append(translated)

            # Respect rate limits between requests
            if idx < len(chunks):
                time.sleep(self.config.translation.request_delay)

        result = "\n".join(self._translated_chunks)
        self._write_file(Path(output_path), result)
        logger.info("Translation complete. Output written to: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_chunk_with_retry(self, chunk: str) -> str:
        """Attempt to translate *chunk*, retrying on transient errors."""
        max_retries = self.config.translation.max_retries
        for attempt in range(1, max_retries + 1):
            try:
                return self.client.translate(chunk)
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries:
                    logger.error(
                        "All %d retry attempts exhausted. Last error: %s",
                        max_retries,
                        exc,
                    )
                    raise
                wait = 2 ** attempt  # exponential back-off
                logger.warning(
                    "Attempt %d failed (%s). Retrying in %ds …",
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
        # Should never reach here
        raise RuntimeError("Translation failed unexpectedly.")

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
        """Split *text* into chunks of at most *chunk_size* characters,
        preferring paragraph boundaries where possible."""
        if chunk_size <= 0:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para) + 2  # account for the '\n\n' separator
            if current_len + para_len > chunk_size and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a plain-text or Markdown file."""
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _write_file(path: Path, content: str) -> None:
        """Write *content* to *path*, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
