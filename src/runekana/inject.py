from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING, Union
from dataclasses import dataclass

from lxml import etree
from lxml.builder import ElementMaker
import jaconv

from .text import has_kanji, split_okurigana
from .tokenizer import Tokenizer, Token

if TYPE_CHECKING:
    from .document import XhtmlDocument

XHTML_NS = "http://www.w3.org/1999/xhtml"
SKIP_TAGS = {"ruby", "rt", "rp", "nav", "title", "head"}


log = logging.getLogger(__name__)
E = ElementMaker(namespace=XHTML_NS)


@dataclass
class InjectionTask:
    """A command object representing a deferred ruby injection."""

    elem: etree._Element
    attr: str  # "text" or "tail"
    tokens: list[Token]

    def apply(self):
        """Convert tokens to nodes and inject them into the DOM."""
        nodes = self._tokens_to_nodes()
        if not nodes:
            return

        if self.attr == "text":
            # Clear old text and insert nodes at the beginning
            self.elem.text = None
            self._inject_at(self.elem, 0, nodes)
        elif self.attr == "tail":
            # Clear old tail and insert nodes after this element
            parent = self.elem.getparent()
            if parent is not None:
                idx = list(parent).index(self.elem)
                self.elem.tail = None
                self._inject_at(parent, idx + 1, nodes)

    def _inject_at(
        self,
        container: etree._Element,
        start_idx: int,
        nodes: list[Union[str, etree._Element]],
    ):
        """Helper to inject a mix of strings and Elements into a container at a specific index."""
        current_idx = start_idx

        # If the first node is text, it becomes the .text of the container (at start_idx)
        # or the .tail of the element before start_idx.
        # But wait, lxml's indexing is about children, not text.

        first_node = nodes[0]
        nodes_to_process = nodes

        if isinstance(first_node, str):
            if start_idx == 0:
                container.text = first_node
            else:
                # Attach to the tail of the element before the insertion point
                prev_sibling = container[start_idx - 1]
                prev_sibling.tail = first_node
            nodes_to_process = nodes[1:]

        # Insert remaining nodes
        current_pos = start_idx
        last_elem = None
        for node in nodes_to_process:
            if isinstance(node, str):
                if last_elem is not None:
                    last_elem.tail = node
                else:
                    # This case happens if the first node was an element,
                    # and the second is text.
                    if start_idx == 0:
                        container.text = (container.text or "") + node
                    else:
                        prev = container[start_idx - 1]
                        prev.tail = (prev.tail or "") + node
            else:
                container.insert(current_pos, node)
                last_elem = node
                current_pos += 1

    def _segments_to_nodes(
        self, segments: list[tuple[str, Optional[str]]]
    ) -> list[Union[str, etree._Element]]:
        nodes = []
        for text, ruby in segments:
            if ruby and ruby != jaconv.kata2hira(text):
                # Using E builder for clean, namespaced element creation
                nodes.append(E.ruby(text, E.rt(ruby)))
            else:
                # Merge consecutive text nodes
                if nodes and isinstance(nodes[-1], str):
                    nodes[-1] += text
                else:
                    nodes.append(text)
        return nodes

    def _tokens_to_nodes(self) -> list[Union[str, etree._Element]]:
        """Convert all tokens into a normalized list of text strings and Elements."""
        if not any(token.reading is not None for token in self.tokens):
            return []

        all_nodes = []
        for token in self.tokens:
            surface = token.surface
            reading = token.reading
            if reading is not None:
                all_nodes.extend(
                    self._segments_to_nodes(split_okurigana(surface, reading))
                )
            else:
                if all_nodes and isinstance(all_nodes[-1], str):
                    all_nodes[-1] += surface
                else:
                    all_nodes.append(surface)
        return all_nodes


class DomTraverser:
    """Scans an XhtmlDocument to produce deferred injection tasks and LLM candidates."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.tasks: list[InjectionTask] = []
        self.candidates: list[dict] = []
        self.doc: Optional[XhtmlDocument] = None

    def traverse(self, doc: XhtmlDocument) -> tuple[list[InjectionTask], list[dict]]:
        """
        Traverses the document's XML tree.
        Returns:
            tasks: List of InjectionTask to apply later.
            candidates: List of dictionary candidates for LLM verification.
        """
        self.doc = doc
        self.tasks.clear()
        self.candidates.clear()
        self._traverse_node(doc.root, inside_ruby=False)
        return self.tasks, self.candidates

    def _should_skip_element(self, elem: etree._Element) -> bool:
        """Check if the element should be skipped (e.g., ruby, script, style tags)."""
        if not isinstance(elem.tag, str):
            return True
        return etree.QName(elem.tag).localname.lower() in SKIP_TAGS

    def _process_node_text(
        self,
        target_elem: etree._Element,
        attr: str,
        text: str,
        context_node: etree._Element,
    ):
        """Tokenize text and create injection tasks and LLM candidates."""
        if not text or not text.strip() or not has_kanji(text):
            return

        tokens = self.tokenizer.tokenize(text)
        self.tasks.append(InjectionTask(elem=target_elem, attr=attr, tokens=tokens))

        if not self.doc:
            return

        full_context = self.doc.get_block_text(context_node)
        search_start_idx = 0
        for token in tokens:
            if token.to_verify and token.reading is not None:
                surface = token.surface

                # gathering context for llm
                idx = full_context.find(surface, search_start_idx)
                if idx == -1:
                    idx = full_context.find(surface)
                if idx != -1:
                    start = max(0, idx - 30)
                    end = min(len(full_context), idx + len(surface) + 30)
                    ctx_snippet = full_context[start:end]
                    search_start_idx = idx + len(surface)
                else:
                    ctx_snippet = surface

                self.candidates.append(
                    {
                        "word": surface,
                        "reading": token.reading,
                        "context": ctx_snippet,
                        "token_ref": token,
                    }
                )

    def _traverse_node(self, elem: etree._Element, inside_ruby: bool):
        """Recursive traversal of the DOM tree to find text nodes."""
        is_ruby = (
            isinstance(elem.tag, str) and etree.QName(elem.tag).localname == "ruby"
        )
        current_inside_ruby = inside_ruby or is_ruby

        if not current_inside_ruby and not self._should_skip_element(elem):
            if elem.text:
                self._process_node_text(
                    target_elem=elem,
                    attr="text",
                    text=elem.text,
                    context_node=elem,
                )

        for child in list(elem):
            if isinstance(child.tag, str):
                self._traverse_node(child, current_inside_ruby)

            if not current_inside_ruby and child.tail:
                self._process_node_text(
                    target_elem=child,
                    attr="tail",
                    text=child.tail,
                    context_node=elem,
                )
