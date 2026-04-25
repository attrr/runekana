#!/usr/bin/env python3
"""
runekana — Add furigana (ruby) annotations to Japanese EPUB files.

Uses Sudachi for morphological analysis, Yomitan dictionaries for frequency-based
filtering, preserves original HTML structure, skips already-annotated text,
and optionally verifies difficult readings via Gemini or OpenAI.
"""

import argparse
import logging
import os

from runekana.epub import process_epub
from runekana.tokenizer import build_skip_set, load_local_dict, save_local_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def main(args):
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.freq_dict:
        from runekana.tokenizer import import_yomitan_dict

        import_yomitan_dict(args.freq_dict)

    skip_words = build_skip_set(args.skip_top)

    if args.dict:
        dict_path = args.dict
    else:
        base, _ = os.path.splitext(args.input)
        dict_path = f"{base}_dict.tsv"

    local_dict = load_local_dict(dict_path)

    try:
        process_epub(
            input_path=args.input,
            output_path=args.output,
            skip_words=skip_words,
            local_dict=local_dict,
            dict_path=dict_path,
            verify=args.verify,
            contextual=args.contextual,
            provider=args.provider,
            model=args.model,
        )
    except KeyboardInterrupt:
        log.warning(
            "\nProcess interrupted by user (Ctrl-C). Gracefully shutting down..."
        )
        log.info("Saving progress to %s before exiting...", dict_path)
        save_local_dict(dict_path, local_dict)
        os._exit(130)


def cli():
    p = argparse.ArgumentParser(
        description="Add furigana annotations to Japanese EPUB files.",
    )
    p.add_argument("input", help="Input EPUB file")
    p.add_argument("output", help="Output EPUB file")
    p.add_argument(
        "--skip-top",
        type=int,
        default=1500,
        help="Skip the top N most frequent Japanese words (default: 1500).",
    )
    p.add_argument(
        "--freq-dict",
        default=None,
        help="Path to Yomitan frequency dictionary (ZIP or folder) to import into local cache.",
    )
    p.add_argument(
        "--dict",
        default=None,
        help="Local dictionary file (TSV: word<TAB>reading).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify readings with LLM (needs GCP_PROJECT for gemini, or OPENAI_API_KEY for openai).",
    )
    p.add_argument(
        "--contextual",
        action="store_true",
        help="Deduplicate candidates by context (sends identical words in different contexts to LLM).",
    )
    p.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="API provider for verification (default: gemini).",
    )
    p.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        help="LLM model name to use for verification.",
    )
    p.add_argument("--verbose", "-v", action="store_true")

    parsed_args = p.parse_args()
    main(parsed_args)


if __name__ == "__main__":
    cli()
