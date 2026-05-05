from __future__ import annotations
import os
import tempfile
import zipfile
import logging
from typing import Iterator, Optional

from runekana import console
from runekana.tokenizer import Tokenizer, save_local_dict
from runekana.llm import Verifier, VerificationJob, LLM
from runekana.text import has_kanji
from runekana.document import XhtmlDocument, TextNode, TokenizedText

log = logging.getLogger("runekana.io")


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
        generated_dir: Optional[str] = None,
    ) -> int:
        """
        Orchestrate the annotation pipeline: Tokenise, Verify, and Inject.
        Returns total number of LLM corrections applied.
        """
        console.print("[bold blue]Scanning XHTML documents...[/bold blue]")
        all_tokenized: list[TokenizedText] = []
        jobs_map: dict = {}

        # Tokenise every TextNode in every paragraph of every document
        docs = list(self.xhtml_documents())
        for doc in docs:
            rel_path = os.path.relpath(doc.filepath, self.epub_dir)
            log.info("Scanning: %s", rel_path)

            for para in doc.paragraphs():
                for seg in para.segments:
                    if not isinstance(seg, TextNode):
                        continue

                    tokenized = seg.tokenize(self.tokenizer)
                    all_tokenized.append(tokenized)

                    # collect kanji readings as LLM verification candidates
                    for yomi in tokenized.annotations:
                        if yomi.reading is None or not has_kanji(yomi.base):
                            continue
                        if not yomi.to_verify:
                            continue
                        yomi_ctx = tokenized.get_context(
                            yomi, forward_max=60, backward_max=60
                        )
                        ctx_str = yomi_ctx.get_nearest_clause().to_highlighted_string()
                        key = (yomi.base, yomi.reading, ctx_str if contextual else None)
                        if key not in jobs_map:
                            jobs_map[key] = VerificationJob(
                                word=yomi.base,
                                proposed_reading=yomi.reading,
                                context=ctx_str,
                                token_refs=[],
                            )
                        # Yomi.reading is mutable; Verifier sets .reading on each ref
                        jobs_map[key].token_refs.append(yomi)

        all_jobs = list(jobs_map.values())
        corrections = 0

        # Verify uncertain readings with LLM if provided
        if llm and all_jobs:
            console.print(
                f"[bold cyan]Verifying {len(all_jobs)} candidate groups via {llm.provider}...[/bold cyan]"
            )
            verifier = Verifier(
                llm=llm,
                local_dict=self.tokenizer.local_dict,
                dict_path=dict_path,
                save_fn=save_local_dict,
                concurrency=concurrency,
                batch_size=batch_size,
                price_input=price_input,
                price_output=price_output,
                generated_dir=generated_dir,
                book_name=os.path.splitext(os.path.basename(self.input_path))[0],
            )
            with verifier:
                corrections = verifier.verify(all_jobs)
        elif llm:
            console.print("[yellow]No words found that require verification.[/yellow]")

        # Inject all tokenised annotations into the DOM
        console.print("[bold magenta]Applying ruby injections...[/bold magenta]")
        for tokenized in all_tokenized:
            tokenized.inject()

        # Save all modified documents
        for doc in docs:
            rel_path = os.path.relpath(doc.filepath, self.epub_dir)
            log.info("Writing: %s", rel_path)
            doc.save()

        console.print(
            f"[bold green]Success![/bold green] Applied {corrections} corrections."
        )
        return corrections
