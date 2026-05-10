#!/usr/bin/env python3
"""Main entry point for TranslateBooksWithLLMs.

Handles CLI argument parsing and orchestrates the translation workflow.
"""

import argparse
import sys
import logging
from pathlib import Path

from config import load_config
from translator import Translator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="translate-books",
        description="Translate books and documents using LLMs.",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input file to translate (e.g., .txt, .epub, .pdf).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path for the translated output file. Defaults to <input>_translated.<ext>.",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=None,
        help="Source language (e.g., 'English'). Overrides config value.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default=None,
        help="Target language (e.g., 'French'). Overrides config value.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name to use. Overrides config value.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".env"),
        help="Path to the .env configuration file (default: .env).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Derive a default output path if none is provided."""
    if output_path is not None:
        return output_path
    stem = input_path.stem
    suffix = input_path.suffix
    return input_path.with_name(f"{stem}_translated{suffix}")


def main() -> int:
    """Run the translation pipeline.

    Returns:
        Exit code (0 for success, non-zero on error).
    """
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input file
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    if not args.input.is_file():
        logger.error("Input path is not a file: %s", args.input)
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    # Apply CLI overrides
    if args.source_lang:
        config.translation.source_language = args.source_lang
    if args.target_lang:
        config.translation.target_language = args.target_lang
    if args.model:
        config.llm.model = args.model

    output_path = resolve_output_path(args.input, args.output)

    logger.info(
        "Translating '%s' from %s to %s using model '%s'.",
        args.input,
        config.translation.source_language,
        config.translation.target_language,
        config.llm.model,
    )

    # Run translation
    try:
        translator = Translator(config)
        translator.translate_file(args.input, output_path)
        logger.info("Translation complete. Output saved to '%s'.", output_path)
    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user.")
        return 130
    except Exception as exc:
        logger.error("Translation failed: %s", exc, exc_info=args.verbose)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
