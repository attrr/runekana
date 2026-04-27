from __future__ import annotations
import os
import re
import tempfile
import zipfile
import logging
from typing import Iterator, Optional
from runekana import console

from lxml import etree

from runekana.tokenizer import Tokenizer, save_local_dict
from runekana.llm import verify_candidates, VerificationJob, LLM
from runekana.inject import DomTraverser

log = logging.getLogger("runekana.io")

# Replicating original xhtml.py constants related to serialization
XHTML_NS = "http://www.w3.org/1999/xhtml"
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class XhtmlDocument:
    """Represents an XHTML document parsed from an EPUB archive."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, "rb") as f:
            self.original_bytes = f.read()

        parser = etree.XMLParser(
            remove_blank_text=False, strip_cdata=False, resolve_entities=False
        )
        self.tree = etree.parse(filepath, parser)

    @property
    def root(self):
        return self.tree.getroot()

    def get_block_text(self, elem: etree._Element) -> str:
        """Find the nearest block-level container and extract all its text."""
        block_tags = {
            "p",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "body",
            "html",
        }
        current = elem
        while current is not None:
            # Use QName to safely handle namespaced tags like {http://...}p
            if isinstance(current.tag, str):
                tag_name = etree.QName(current.tag).localname.lower()
                if tag_name in block_tags:
                    break
            current = current.getparent()

        if current is None:
            current = elem

        text = "".join(current.itertext())
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_empty_tags(tree: etree._ElementTree | etree._Element):
        """Ensure non-void elements are not self-closed by setting text to empty string."""
        for elem in tree.iter():
            if isinstance(elem.tag, str):
                local_name = etree.QName(elem.tag).localname.lower()
                if local_name not in VOID_ELEMENTS:
                    if elem.text is None and len(elem) == 0:
                        elem.text = ""

    @staticmethod
    def _extract_header(content: bytes) -> bytes:
        """Extract document prolog (XML declaration, DOCTYPE, and comments before the root element)."""
        match = re.search(b"<(?!\\?|!|!--)", content)
        if match:
            return content[: match.start()]
        return b""

    def save(self):
        """Serialize the DOM back to disk, preserving original header and XHTML compatibility."""
        self._normalize_empty_tags(self.tree)

        # Serialize ONLY the root to avoid lxml re-inserting DOCTYPE from the tree object.
        # NOTE: This ignores any trailing content (comments, whitespace) after the root 
        # element in the original file, which is usually acceptable for EPUB documents.
        body_bytes = etree.tostring(
            self.tree.getroot(),
            encoding="utf-8",
            xml_declaration=False,
            method="xml",
            pretty_print=False,
        )

        header = self._extract_header(self.original_bytes)
        output = header.decode("utf-8") + body_bytes.decode("utf-8")
        with open(self.filepath, "w", encoding="utf-8", newline="") as f:
            f.write(output)


class EpubArchive:
    """Context manager for extracting and repacking EPUB files."""

    def __init__(self, input_path: str, output_path: str, tokenizer: Tokenizer):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer = tokenizer
        self._temp_dir = tempfile.TemporaryDirectory()
        self.epub_dir = os.path.join(self._temp_dir.name, "epub")

    def __enter__(self):
        self.unpack()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.repack()
        self._temp_dir.cleanup()

    def unpack(self):
        with zipfile.ZipFile(self.input_path, "r") as zip_ref:
            zip_ref.extractall(self.epub_dir)
        log.info("Unpacked EPUB to temporary directory")

    def repack(self):
        with zipfile.ZipFile(self.output_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            # Add mimetype first (uncompressed)
            mimetype_path = os.path.join(self.epub_dir, "mimetype")
            if os.path.exists(mimetype_path):
                zip_ref.write(
                    mimetype_path, arcname="mimetype", compress_type=zipfile.ZIP_STORED
                )

            # Add everything else
            for root, dirs, files in os.walk(self.epub_dir):
                for file in files:
                    if file == "mimetype" and root == self.epub_dir:
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.epub_dir)
                    zip_ref.write(file_path, arcname=arcname)
        log.info("Output written to %s", self.output_path)

    def xhtml_documents(self) -> Iterator[XhtmlDocument]:
        for root, dirs, files in os.walk(self.epub_dir):
            for file in files:
                if file.endswith((".xhtml", ".html")):
                    yield XhtmlDocument(os.path.join(root, file))

    def process(
        self,
        dict_path: str,
        llm: Optional[LLM] = None,
        contextual: bool = False,
        concurrency: int = 5,
        batch_size: int = 100,
        price_input: float = 0.0,
        price_output: float = 0.0,
    ) -> int:
        """
        Orchestrate the annotation pipeline: Scan, Verify, and Inject.
        Returns total number of corrections applied.
        """
        console.print("[bold blue]Scanning XHTML documents...[/bold blue]")
        all_tasks = []
        jobs_map = {}

        # Scan all documents to collect candidates and injection tasks
        docs = list(self.xhtml_documents())
        for doc in docs:
            rel_path = os.path.relpath(doc.filepath, self.epub_dir)
            log.info("Scanning: %s", rel_path)

            traverser = DomTraverser(self.tokenizer)
            tasks, candidates = traverser.traverse(doc)
            all_tasks.extend(tasks)

            for c in candidates:
                word, reading, context = c["word"], c["reading"], c["context"]
                key = (word, reading, context if contextual else None)

                if key not in jobs_map:
                    jobs_map[key] = VerificationJob(
                        word=word,
                        proposed_reading=reading,
                        context=context,
                        token_refs=[],
                    )
                jobs_map[key].token_refs.append(c["token_ref"])

        all_jobs = list(jobs_map.values())
        corrections = 0

        # Verify difficult readings with LLM if provided
        if llm and all_jobs:
            console.print(
                f"[bold cyan]Verifying {len(all_jobs)} candidate groups via {llm.provider}...[/bold cyan]"
            )
            corrections = verify_candidates(
                all_jobs,
                self.tokenizer.local_dict,
                dict_path,
                llm,
                save_fn=save_local_dict,
                concurrency=concurrency,
                batch_size=batch_size,
                price_input=price_input,
                price_output=price_output,
            )
        elif llm:
            console.print("[yellow]No words found that require verification.[/yellow]")

        # Execute ruby injections into the DOM
        if all_tasks:
            console.print("[bold magenta]Applying ruby injections...[/bold magenta]")
            for task in all_tasks:
                task.apply()

        # Save all modified documents back to the temporary directory
        for doc in docs:
            rel_path = os.path.relpath(doc.filepath, self.epub_dir)
            log.info("Writing: %s", rel_path)
            doc.save()

        console.print(
            f"[bold green]Success![/bold green] Applied {corrections} corrections."
        )
        return corrections
