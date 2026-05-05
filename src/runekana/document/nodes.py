from __future__ import annotations
import logging
from bisect import bisect_right
from typing import Literal, Optional, Self, TYPE_CHECKING


from lxml import etree

from runekana.tokenizer import Token, Tokenizer
from runekana.text import chunk_by_kanji, normalize_kana

from .utils import get_tag_from_element, generate_offsets, find_overlap_range
from .tokens import Yomi, TokenizedText

log = logging.getLogger("runekana.document")

if TYPE_CHECKING:
    from .xhtml import Paragraph


class ImpossibleToAlignException(Exception):
    pass


class TextNode:
    """
    A data object, that only hold the smallest lxml element,
    in other word, an element without subtree
    """

    def __init__(
        self,
        element: etree._Element,
        attr: Literal["text", "tail"],
        paragraph: Paragraph,
    ) -> None:
        self.element = element
        self.attr = attr
        self.tag = get_tag_from_element(element)
        self.text = getattr(self.element, self.attr) or ""
        self.paragraph = paragraph

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"TextNode({self.tag=}, {self.attr=}, {self.text=})".replace("self.", "")

    def __len__(self) -> int:
        return len(self.text)

    @staticmethod
    def _in_range(boundary: int, begin: int, end: int) -> bool:
        return begin < boundary < end

    def _consume_ltr(
        self,
        chunks: list[tuple[str, bool]],
        overlap_chunk: tuple[str, bool],
        reading: str,
    ) -> str:
        """
        Consume the reading for all surface chunks that fall entirely before
        the overlap chunk, advancing the reading pointer to the start of the
        overlap chunk's portion.

        Args:
            chunks: Surface chunks produced by ``chunk_by_kanji`` that lie
                *before* the overlap chunk.  They alternate strictly between
                kanji blocks (``is_kanji=True``) and kana blocks
                (``is_kanji=False``), e.g. [kanji, kana, kanji, kana, ...].
                Every kanji block is therefore immediately followed by a kana
                block (either the next element in ``chunks`` or
                ``overlap_chunk`` itself).
            overlap_chunk: The first chunk that straddles or starts at the
                boundary offset.  By the alternating guarantee, when
                ``chunks[-1]`` is a kanji block, ``overlap_chunk`` is the
                subsequent kana block and its text is used as the right-hand
                kana anchor to locate where the kanji's reading ends.
            reading: Normalised kana reading for the full token surface
                (i.e. ``normalize_kana(tok.reading)``).

        Returns:
            The suffix of ``reading`` that corresponds to the overlap chunk
            and everything after it.  The caller is responsible for further
            trimming this suffix at ``inner_offset`` inside the overlap chunk.

        Raises:
            ValueError: If a kana chunk's text does not match the current
                prefix of ``reading`` (indicates a bug in the caller or in
                ``chunk_by_kanji``).
        """

        for idx, chunk in enumerate(chunks):
            text, is_kanji = chunk

            if not is_kanji:
                if reading.startswith(text):
                    reading = reading.removeprefix(text)
                    continue
                else:
                    raise ImpossibleToAlignException(
                        f"kana in reading not match kana in surface: {text!r} not in {reading!r}"
                    )

            if idx == len(chunks) - 1:
                next_kana = overlap_chunk[0]
            else:
                next_kana, _ = chunks[idx + 1]

            end = reading.find(next_kana, len(text) - 1)
            reading = reading[end:]
        return reading

    def _consume_rtl(
        self,
        chunks: list[tuple[str, bool]],
        overlap_chunk: tuple[str, bool],
        reading: str,
    ) -> str:
        """
        Consume the reading for all surface chunks that fall entirely after
        the overlap chunk, trimming the reading pointer from the right to the
        end of the overlap chunk's portion.

        Args:
            chunks: Surface chunks produced by ``chunk_by_kanji`` that lie
                *after* the overlap chunk.  They alternate strictly between
                kanji blocks (``is_kanji=True``) and kana blocks
                (``is_kanji=False``), e.g. [kana, kanji, kana, kanji, ...].
                Every kanji block is therefore immediately preceded by a kana
                block (either the previous element in ``chunks`` or
                ``overlap_chunk`` itself).
            overlap_chunk: The last chunk that straddles or starts at the
                boundary offset.  By the alternating guarantee, when
                ``chunks[0]`` is a kanji block, ``overlap_chunk`` is the
                preceding kana block and its text is used as the left-hand
                kana anchor to locate where the kanji's reading begins.
            reading: Normalised kana reading for the full token surface
                (i.e. ``normalize_kana(tok.reading)``).

        Returns:
            The prefix of ``reading`` that corresponds to the overlap chunk
            and everything before it.  The caller is responsible for further
            trimming this prefix at ``inner_offset`` inside the overlap chunk.

        Raises:
            ValueError: If a kana chunk's text does not match the current
                suffix of ``reading`` (indicates a bug in the caller or in
                ``chunk_by_kanji``).
        """
        for idx in range(len(chunks) - 1, -1, -1):
            text, is_kanji = chunks[idx]

            if not is_kanji:
                if not reading.endswith(text):
                    # FIXME: only return this for kana-only chunk(no number/en/etc)
                    raise ImpossibleToAlignException(
                        f"kana in reading not match kana in surface: {text!r} not in {reading!r}"
                    )
                reading = reading.removesuffix(text)
                continue

            # kanji: use the preceding kana (in original order) as left anchor
            if idx == 0:
                prev_kana = overlap_chunk[0]
            else:
                prev_kana, _ = chunks[idx - 1]

            anchor = reading.rfind(prev_kana, 0, len(reading) - (len(text) - 1))
            assert (
                anchor != -1
            ), f"bug: kana anchor {prev_kana!r} not found in {reading!r}"
            reading = reading[: anchor + len(prev_kana)]

        return reading

    def _align_token_to_boundary(
        self,
        tok: Token,
        boundary: int,
        annotations: Optional[dict[str, str]] = None,
        direction: Literal["ltr", "rtl"] = "ltr",
    ) -> Optional[str]:
        if boundary < 0:
            raise ValueError("boundary should not be negative number")

        # convert abs position to tok surface relative index
        offset = boundary - tok.begin
        if offset < 0 or offset >= len(tok.surface):
            raise ValueError("invalid boundary")

        if not tok.reading:
            return None

        annotations = annotations or {}
        reading = normalize_kana(tok.reading)
        chunks = chunk_by_kanji(tok.surface)

        chunks_start_offsets, chunks_end_offsets = generate_offsets(
            [t for t, _ in chunks]
        )

        # bisect_right on end offsets: first chunk whose end > offset = overlap chunk
        overlap_idx = bisect_right(chunks_end_offsets, offset)
        overlap_chunk = chunks[overlap_idx]
        chunk_text, is_kanji = overlap_chunk
        inner_offset = offset - chunks_start_offsets[overlap_idx]

        if direction == "ltr":
            reading = self._consume_ltr(chunks[:overlap_idx], overlap_chunk, reading)
            # discard overlap chunk's prefix (surface[:inner_offset]) from reading
            if not is_kanji:
                reading = reading[inner_offset:]
            else:
                discarded = chunk_text[:inner_offset]
                r = annotations.get(discarded)
                if r and reading.startswith(r):
                    reading = reading[len(r) :]
                else:
                    for ch in discarded:
                        r = annotations.get(ch)
                        if not r or not reading.startswith(r):
                            raise ImpossibleToAlignException(
                                f"cannot split reading at left boundary: "
                                f"surface={tok.surface!r} reading={tok.reading!r} "
                                f"char={ch!r} remaining={reading!r}"
                            )
                        reading = reading[len(r) :]
        else:
            reading = self._consume_rtl(
                chunks[overlap_idx + 1 :], overlap_chunk, reading
            )
            # discard overlap chunk's suffix (surface[inner_offset:]) from reading
            trim = len(chunk_text) - inner_offset
            if not is_kanji:
                reading = reading[:-trim] if trim else reading
            else:
                discarded = chunk_text[inner_offset:]
                r = annotations.get(discarded)
                if r and reading.endswith(r):
                    reading = reading[: -len(r)]
                else:
                    for ch in reversed(discarded):
                        r = annotations.get(ch)
                        if not r or not reading.endswith(r):
                            raise ImpossibleToAlignException(
                                f"cannot split reading at right boundary: "
                                f"surface={tok.surface!r} reading={tok.reading!r} "
                                f"char={ch!r} remaining={reading!r}"
                            )
                        reading = reading[: -len(r)]

        return reading

    def tokenize(self, tok: Tokenizer) -> TokenizedText:
        """
        Tokenise this TextNode within its clause context, aligning boundary
        tokens to the TextNode's exact character limits.

        Falls back to tokenising ``self.text`` in isolation when boundary
        alignment fails (e.g. an unknown kanji at a cross-node token split).
        """
        ctx = self.paragraph.get_context(self).get_nearest_clause()
        tokens = tok.tokenize(str(ctx))

        # trim to tokens overlapping with the target segment
        tok_starts = [t.begin for t in tokens]
        tok_ends = tok_starts[1:] + [tokens[-1].end]
        start_idx, end_idx = find_overlap_range(
            tok_starts, tok_ends, ctx.start_idx, ctx.end_idx
        )
        relevant = tokens[start_idx:end_idx]

        try:
            # align first boundary token
            first = relevant[0]
            first_reading = normalize_kana(first.reading) if first.reading else None
            if first.begin != ctx.start_idx:
                exist_reading = ctx.get_overlap_annotation_map(
                    begin=first.begin, end=ctx.start_idx
                )
                first_reading = self._align_token_to_boundary(
                    first,
                    boundary=ctx.start_idx,
                    annotations=exist_reading,
                    direction="ltr",
                )

            # align last boundary token
            last = relevant[-1]
            last_reading = normalize_kana(last.reading) if last.reading else None
            if last.end != ctx.end_idx:
                exist_reading = ctx.get_overlap_annotation_map(
                    begin=ctx.end_idx, end=last.end
                )
                last_reading = self._align_token_to_boundary(
                    last,
                    boundary=ctx.end_idx,
                    annotations=exist_reading,
                    direction="rtl",
                )

        except ImpossibleToAlignException:
            log.warning(
                "boundary alignment failed for %r, falling back to isolated tokenisation",
                self.text,
            )
            return TokenizedText(
                node=self,
                annotations=[
                    Yomi(
                        base=t.surface,
                        reading=(
                            normalize_kana(t.reading)
                            if t.reading
                            and normalize_kana(t.reading) != normalize_kana(t.surface)
                            else None
                        ),
                        begin=t.begin,
                        end=t.end,
                        to_verify=t.to_verify,
                    )
                    for t in tok.tokenize(self.text)
                ],
            )

        # build Yomi list from aligned tokens
        yomis: list[Yomi] = []
        for t in relevant:
            is_first = t is first
            is_last = t is last

            if is_first and is_last:
                # TODO: single token spans both boundaries; needs two-sided trim
                surface = t.surface[ctx.start_idx - t.begin : ctx.end_idx - t.begin]
                reading = first_reading  # approximation: ltr-trimmed reading
            elif is_first:
                surface = t.surface[ctx.start_idx - t.begin :]
                reading = first_reading
            elif is_last:
                surface = t.surface[: ctx.end_idx - t.begin]
                reading = last_reading
            else:
                surface = t.surface
                reading = normalize_kana(t.reading) if t.reading else None

            # kana-only: no ruby needed
            if reading and normalize_kana(surface) == reading:
                reading = None

            begin = max(0, t.begin - ctx.start_idx)
            end = min(len(self.text), t.end - ctx.start_idx)
            yomis.append(
                Yomi(
                    base=surface,
                    reading=reading,
                    begin=begin,
                    end=end,
                    to_verify=t.to_verify,
                )
            )

        return TokenizedText(node=self, annotations=yomis)


class Ruby:
    """
    input a element, resolve a object?
    """

    def __init__(
        self,
        element: etree._Element,
        pairs: list[tuple[str, str]],
        paragraph: Paragraph,
    ) -> None:
        self.element = element
        self.pairs = pairs
        self.text = "".join(base for base, _ in self.pairs)
        self.paragraph = paragraph

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Ruby({self.pairs})"

    def __len__(self) -> int:
        return len(self.text)

    @staticmethod
    def _parse(ruby: etree._Element) -> list[tuple[str, str]]:
        """
        Parse an exist ruby etree to Ruby object
        dont bother to create a mapping between tag/result, as we wont inject this back anyway
        WARNING: this will produce malform result, if rt is not direct child of ruby
        """
        results = []
        buffer = []

        buffer.append(ruby.text or "")
        for elem in ruby:
            tag = get_tag_from_element(elem)
            if tag == "rt":
                base = "".join(buffer).strip()
                annotation = "".join(elem.itertext()).strip()
                if base:
                    results.append((base, annotation))
                buffer.clear()
            elif tag == "rp":
                pass
            else:
                # rb or other tag, treat as text only
                buffer.append("".join(elem.itertext()).strip())
            if elem.tail:
                buffer.append(elem.tail)
        return results

    @classmethod
    def parse(cls, ruby: etree._Element, paragraph: Paragraph) -> Self:
        pairs = cls._parse(ruby)
        return cls(element=ruby, pairs=pairs, paragraph=paragraph)
