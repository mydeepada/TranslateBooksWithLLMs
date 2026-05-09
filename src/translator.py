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
            # Put translated files in a dedicated 'translated/' subdirectory
            # to keep the source folder tidy.
            output_dir = source.parent / "translated"
            output_dir.mkdir(exist_ok=True)
            output_path = str(
                output_dir / f"{source.stem}_{target_lang}{source.suffix}"
            )

        logger.info("Reading source file: %s", input_path)
        text = self._read_file(source)

        chunks = self._split_into_chunks(
            text, self.config.translation.chunk_size
        )
        logger.info("Split into %d chunk(s) for translation.", len(chunks))

        self._translated_chunks = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            # Print progress to stdout as well so I can monitor long books
            # without tailing the log file.
            print(f"[{idx}/{total}] Translating chunk {idx} …")
            logger.info("Translating chunk %d / %d …", idx, total)
            translated = self._translate_chunk_with_retry(chunk)
            self._translated_chunks.append(translated)

            # Respect rate limits between requests
            if idx < total:
                time.sleep(self.config.translation.request_delay)

        # Use double newline between chunks so paragraph breaks are preserved
        result = "\n\n".join(self._translated_chunks)
        self._write_file(Path(output_path), result)
        logger.info("Translation complete. Output written to: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # --------------------------
