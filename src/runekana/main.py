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
import sys

from rich.logging import RichHandler

from runekana import console
from runekana.document import EpubArchive
from runekana.llm import Gemini, Vertex, OpenAI, LLM
from runekana.tokenizer import (
    load_local_dict,
    save_local_dict,
    Tokenizer,
    YomitanDB,
)

log = logging.getLogger("runekana")


def _build_llm(args) -> LLM:
    """Construct the appropriate LLM client from CLI arguments and env vars."""
    kwargs = {"model_name": args.model}
    if args.canary_url:
        kwargs["canary_url"] = args.canary_url

    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        gcp_project = os.environ.get("GCP_PROJECT")
        if api_key:
            return Gemini(api_key=api_key, base_url=args.base_url, **kwargs)
        elif gcp_project:
            location = os.environ.get("GCP_LOCATION", "global")
            return Vertex(project=gcp_project, location=location, **kwargs)
        else:
            log.error(
                "Either GEMINI_API_KEY or GCP_PROJECT env var is required for gemini provider"
            )
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set. Assuming proxy/setup handles auth.")
        return OpenAI(api_key=api_key or "", base_url=args.base_url, **kwargs)


def setup_logging(verbosity: int):
    """Configure logging levels for runekana and its dependencies."""
    # Default to WARNING; each -v drops one level (INFO, DEBUG)
    level = min(logging.WARNING, max(logging.DEBUG, logging.WARNING - verbosity * 10))

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=verbosity > 1,
        show_time=False,
        markup=False,
    )
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def main(args):
    setup_logging(args.verbose)
    db = YomitanDB()

    if args.freq_dict:
        console.print(
            f"[bold blue]Importing frequency dictionary:[/bold blue] {args.freq_dict}"
        )
        db.import_dict(args.freq_dict)

    skip_words = db.get_top_n(args.skip_top)

    if args.dict:
        dict_path = args.dict
    else:
        base, _ = os.path.splitext(args.input)
        dict_path = f"{base}_dict.tsv"

    local_dict = load_local_dict(dict_path)

    llm = None
    if args.verify:
        llm = _build_llm(args)

    tokenizer = Tokenizer(skip_words, local_dict)

    console.print(f"[bold green]Processing EPUB:[/bold green] {args.input}")
    try:
        with EpubArchive(
            input_path=args.input, output_path=args.output, tokenizer=tokenizer
        ) as archive:
            archive.process(
                dict_path=dict_path,
                llm=llm,
                contextual=args.contextual,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                price_input=args.price_input,
                price_output=args.price_output,
            )
    except KeyboardInterrupt:
        log.warning(
            "\nProcess interrupted by user (Ctrl-C). Gracefully shutting down..."
        )
        log.info("Saving progress to %s before exiting...", dict_path)
        save_local_dict(dict_path, local_dict)
        sys.exit(130)


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
    p.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for OpenAI/Gemini-compatible providers (e.g. DeepSeek, vLLM).",
    )
    p.add_argument(
        "--canary-url",
        default="http://connectivitycheck.gstatic.com/generate_204",
        help="URL to check for internet connectivity (default: Google 204). Try Cloudflare or GrapheneOS one if blocked.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max parallel LLM requests during verification (default: 5). Lower for rate-limited APIs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of words to send in a single LLM request (default: 100). Lower for faster feedback.",
    )
    p.add_argument(
        "--price-input",
        type=float,
        default=0.0,
        help="Price per 1M input tokens (USD). Used for cost estimation.",
    )
    p.add_argument(
        "--price-output",
        type=float,
        default=0.0,
        help="Price per 1M output tokens (USD). Used for cost estimation.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vvv)",
    )

    parsed_args = p.parse_args()
    main(parsed_args)


if __name__ == "__main__":
    cli()
