import os
import logging
import zipfile
import tempfile
from collections import defaultdict

from lxml import etree

from .tokenizer import Tokenizer, save_local_dict
from .llm import verify_candidates
from .xhtml import (
    prepare_ruby_injection,
    apply_ruby_injections,
    serialize_xhtml,
)

log = logging.getLogger(__name__)


def unpack_epub(epub_path: str, dest_dir: str):
    """Extract an EPUB file into a directory."""
    with zipfile.ZipFile(epub_path, "r") as zf:
        zf.extractall(dest_dir)


def repack_epub(src_dir: str, epub_path: str):
    """Repack directory into EPUB. mimetype first, uncompressed (EPUB spec)."""
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        mimetype_path = os.path.join(src_dir, "mimetype")
        if os.path.exists(mimetype_path):
            zf.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)

        for root, _dirs, files in os.walk(src_dir):
            for f in sorted(files):
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, src_dir)
                if arcname == "mimetype":
                    continue
                zf.write(filepath, arcname)


def find_xhtml_files(epub_dir: str) -> list[str]:
    """Find all XHTML/HTML files in the unpacked EPUB directory."""
    result = []
    for root, _dirs, files in os.walk(epub_dir):
        for f in sorted(files):
            if f.endswith((".xhtml", ".html", ".htm")):
                result.append(os.path.join(root, f))
    return result


def process_epub(
    input_path: str,
    output_path: str,
    skip_words: set[str],
    local_dict: dict[str, str],
    dict_path: str,
    verify: bool = False,
    contextual: bool = False,
    provider: str = "gemini",
    model: str = "gemini-3.1-flash-lite-preview",
):
    """
    Main orchestration pipeline to unrar an EPUB, tokenize text and defer ruby injections,
    optionally verify difficult readings with an LLM by mutating tokens in-memory,
    apply the injections, and repackage the result.
    """
    log.info("Processing: %s -> %s", input_path, output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        epub_dir = os.path.join(tmpdir, "epub")
        unpack_epub(input_path, epub_dir)
        log.info("Unpacked EPUB")

        tokenizer = Tokenizer(skip_words, local_dict)

        xhtml_files = find_xhtml_files(epub_dir)
        log.info("Found %d XHTML files", len(xhtml_files))

        all_trees = {}
        all_original_bytes = {}
        all_injection_tasks = []
        grouped_candidates = defaultdict(list)

        for xhtml_path in xhtml_files:
            rel = os.path.relpath(xhtml_path, epub_dir)
            parser = etree.XMLParser(
                remove_blank_text=False, strip_cdata=False, resolve_entities=False
            )
            try:
                tree = etree.parse(xhtml_path, parser)
                all_trees[xhtml_path] = tree

                with open(xhtml_path, "rb") as f:
                    all_original_bytes[xhtml_path] = f.read()

                tasks, candidates = prepare_ruby_injection(tree.getroot(), tokenizer)
                all_injection_tasks.extend(tasks)

                for c in candidates:
                    if contextual:
                        k = (c["word"], c["reading"], c["context"])
                    else:
                        k = (c["word"], c["reading"])

                    # Store a tuple of (token_ref, context) so we can log it later
                    grouped_candidates[k].append((c["token_ref"], c["context"]))

            except etree.XMLSyntaxError as e:
                log.warning("Skipping %s: %s", rel, e)
                continue

        if verify and grouped_candidates:
            gcp_project = None
            gcp_location = None
            if provider == "gemini":
                gcp_project = os.environ.get("GCP_PROJECT")
                gcp_location = os.environ.get("GCP_LOCATION", "global")
                if not gcp_project:
                    log.error("GCP_PROJECT env var required when using gemini provider")
                    return
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    log.warning(
                        "OPENAI_API_KEY env var not set. Assuming your proxy/setup handles auth..."
                    )

            log.info(
                "Pass 1: Verifying %d unique candidate groups...",
                len(grouped_candidates),
            )
            verify_candidates(
                grouped_candidates,
                tokenizer.local_dict,
                dict_path,
                provider,
                model,
                save_fn=save_local_dict,
                gcp_project=gcp_project,
                gcp_location=gcp_location,
            )
        elif verify:
            log.info("No words found that require verification.")

        log.info("Pass 2: Applying deferred ruby injections...")
        apply_ruby_injections(all_injection_tasks)

        for xhtml_path, tree in all_trees.items():
            rel = os.path.relpath(xhtml_path, epub_dir)
            log.info("Writing: %s", rel)
            output = serialize_xhtml(tree, all_original_bytes[xhtml_path])
            with open(xhtml_path, "w", encoding="utf-8", newline="") as f:
                f.write(output)

        repack_epub(epub_dir, output_path)
        log.info("Output written to %s", output_path)
