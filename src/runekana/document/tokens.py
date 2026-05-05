from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

from lxml import etree
from lxml.builder import ElementMaker

from runekana.text import split_okurigana, normalize_kana

from .context import Context

if TYPE_CHECKING:
    from .nodes import TextNode

XHTML_NS = "http://www.w3.org/1999/xhtml"
E = ElementMaker(namespace=XHTML_NS)


@dataclass
class Yomi:
    """
    A single reading unit within a TokenizedText.

    Produced by tokenising a TextNode and splitting okurigana.
    Decoupled from Sudachi Token objects: all fields are plain Python values.

    Attributes:
        base: Surface text for this unit (kanji compound or kana run).
        reading: Kana reading for ``base``, or ``None`` if ``base`` is
            already pure kana and needs no ruby annotation.
        begin: Start character offset of ``base`` within the owning TextNode.
        end: End character offset of ``base`` within the owning TextNode
            (exclusive, i.e. ``textnode.text[begin:end] == base``).
    """

    base: str
    reading: Optional[str]
    begin: int
    end: int
    to_verify: bool = False


class TokenizedText:
    def __init__(self, node: TextNode, annotations: list[Yomi]) -> None:
        self.node = node
        self.annotations = annotations

    def get_context(
        self,
        yomi: Yomi,
        forward_max: Optional[int] = None,
        backward_max: Optional[int] = None,
    ) -> Context:
        """
        Return a Context zoomed in on ``yomi``'s position.

        The caller is responsible for passing a ``ctx`` that is already
        windowed to the desired character limits (e.g. obtained via
        ``Paragraph.get_context(textnode, forward_max=N, backward_max=N)``).
        This method only re-focuses ``start_idx``/``end_idx`` onto the yomi's
        span; no additional trimming is applied.

        Args:
            yomi: The specific Yomi unit to focus on.
            forward_max: Max characters to include after this yomi's span.
            backward_max: Max characters to include before this yomi's span.

        Returns:
            A ``Context`` with ``start_idx``/``end_idx`` pointing to the
            yomi's span, ready for ``get_nearest_clause()`` and LLM use.
        """
        ctx = self.node.paragraph.get_context(
            self.node, forward_max=forward_max, backward_max=backward_max
        )
        yomi_start = ctx.start_idx + yomi.begin
        yomi_end = ctx.start_idx + yomi.end
        return Context(
            start_idx=yomi_start,
            end_idx=yomi_end,
            segment=self.node,
            context=ctx.context,
            offsets=ctx.offsets,
            size=ctx.size,
        )

    def modify(self, yomi: Yomi, reading: str) -> None:
        """
        Update the reading for a specific Yomi after LLM correction.

        Args:
            yomi: The Yomi unit to update (must belong to this TokenizedText).
            reading: The corrected kana reading to assign.
        """
        yomi.reading = reading

    @staticmethod
    def _create_ruby(base: str, annotation: str) -> Optional[etree._Element]:
        if not base or not annotation:
            return None
        return E.ruby(base, E.rt(annotation))

    @staticmethod
    def _inject_at(
        container: etree._Element,
        start_idx: int,
        nodes: list[Union[str, etree._Element]],
    ) -> None:
        """Insert a mix of text strings and Elements into ``container`` at ``start_idx``."""
        nodes_to_process = nodes

        if nodes and isinstance(nodes[0], str):
            if start_idx == 0:
                container.text = nodes[0]
            else:
                container[start_idx - 1].tail = nodes[0]
            nodes_to_process = nodes[1:]

        current_pos = start_idx
        last_elem: Optional[etree._Element] = None
        for node in nodes_to_process:
            if isinstance(node, str):
                if last_elem is not None:
                    last_elem.tail = node
                elif start_idx == 0:
                    container.text = (container.text or "") + node
                else:
                    prev = container[start_idx - 1]
                    prev.tail = (prev.tail or "") + node
            else:
                container.insert(current_pos, node)
                last_elem = node
                current_pos += 1

    def inject(self) -> None:
        """
        Inject the tokenised annotations into the DOM, replacing the owning
        TextNode's text (or tail) with the corresponding mix of plain text
        and ``<ruby>`` elements.

        Calls ``split_okurigana`` on each Yomi to separate the kanji root
        from any kana suffix before creating ruby nodes.
        """
        nodes: list[Union[str, etree._Element]] = []

        for yomi in self.annotations:
            if yomi.reading is None:
                # kana-only: plain text, merge adjacent text nodes
                if nodes and isinstance(nodes[-1], str):
                    nodes[-1] += yomi.base
                else:
                    nodes.append(yomi.base)
            else:
                for base, reading in split_okurigana(yomi.base, yomi.reading):
                    if reading and normalize_kana(reading) != normalize_kana(base):
                        nodes.append(E.ruby(base, E.rt(reading)))
                    else:
                        if nodes and isinstance(nodes[-1], str):
                            nodes[-1] += base
                        else:
                            nodes.append(base)

        if not nodes:
            return

        elem = self.node.element
        attr = self.node.attr

        if attr == "text":
            elem.text = None
            self._inject_at(elem, 0, nodes)
        elif attr == "tail":
            parent = elem.getparent()
            if parent is not None:
                idx = list(parent).index(elem)
                elem.tail = None
                self._inject_at(parent, idx + 1, nodes)
