from __future__ import annotations
import re
from typing import Union, Optional, TYPE_CHECKING
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from lxml import etree

from runekana.text import split_okurigana, normalize_kana

from .utils import find_overlap_range, E

if TYPE_CHECKING:
    from .dom import TextNode, Ruby


@dataclass
class Context:
    start_idx: int
    end_idx: int
    segment: Union[TextNode, Ruby]
    context: list[tuple[str, Optional[str]]]
    size: int
    offsets: list[int]
    end_offsets: list[int] = field(init=False)

    def __post_init__(self):
        self.end_offsets = self.offsets[1:] + [self.size]

    def __str__(self) -> str:
        return "".join(base for base, _ in self.context)

    def to_highlighted_string(self) -> str:
        s = str(self)
        return f"{s[:self.start_idx]}【{s[self.start_idx:self.end_idx]}】{s[self.end_idx:]}"

    def get_nearest_clause(self) -> Context:
        return self._get_nearest_clause(self)

    @staticmethod
    def _get_nearest_clause(ctx: Context) -> Context:
        """
        Narrows a Context to the clause containing the target segment.

        Clauses are delimited by Japanese punctuation (、。！？『』「」…).
        Returns a new Context whose tuples and indices cover only that clause.
        Ruby pairs are always preserved whole; only TextNode tuples are sliced
        at clause boundaries.
        """
        text = str(ctx)
        matches = re.finditer(r"[、。！？『』「」…]", text)
        puncts = [0] + [m.end() for m in matches]

        # ensure end exists
        if puncts[-1] != len(text):
            puncts.append(len(text))

        # find clause boundaries in character space
        clause_start_idx = puncts[bisect_right(puncts, ctx.start_idx) - 1]
        clause_end_idx = puncts[bisect_left(puncts, ctx.end_idx)]

        # map clause boundaries to tuple indices
        tuple_start, tuple_end = find_overlap_range(
            ctx.offsets, ctx.end_offsets, clause_start_idx, clause_end_idx
        )

        # slice tuples, truncating boundary TextNodes only
        clause_tuples: list[tuple[str, Optional[str]]] = []
        clause_offsets: list[int] = [0]
        for i in range(tuple_start, tuple_end):
            base, ann = ctx.context[i]
            t_start = ctx.offsets[i]

            if ann is not None:
                # Ruby pair — always include whole
                clause_tuples.append((base, ann))
            else:
                # TextNode — trim at clause boundaries
                lo = max(0, clause_start_idx - t_start)
                hi = min(len(base), clause_end_idx - t_start)
                base = base[lo:hi]
                if not base:
                    continue
                clause_tuples.append((base, None))
            clause_offsets.append(clause_offsets[-1] + len(base))

        clause_size = clause_offsets.pop()

        return Context(
            start_idx=ctx.start_idx - clause_start_idx,
            end_idx=ctx.end_idx - clause_start_idx,
            segment=ctx.segment,
            context=clause_tuples,
            offsets=clause_offsets,
            size=clause_size,
        )

    def tuples_in_range(self, begin: int, end: int) -> list[tuple[str, Optional[str]]]:
        """Return context tuples overlapping with character range [begin, end)."""
        tuple_start, tuple_end = find_overlap_range(
            self.offsets, self.end_offsets, begin, end
        )
        return self.context[tuple_start:tuple_end]

    def get_overlap_annotation_map(self, begin: int, end: int) -> dict[str, str]:
        tuples = self.tuples_in_range(begin, end)
        annotations = {}
        for base, anno in tuples:
            if anno:
                annotations[base] = anno
        return annotations


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
