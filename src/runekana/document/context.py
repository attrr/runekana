from __future__ import annotations
import re
from typing import Union, Optional, TYPE_CHECKING
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from .utils import find_overlap_range

if TYPE_CHECKING:
    from .nodes import TextNode, Ruby


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
