from __future__ import annotations
import logging
import re
from bisect import bisect_right
from typing import Optional, Union, Generator, Self, Literal

import jaconv
from lxml import etree

from runekana.tokenizer import Token, Tokenizer
from runekana.text import chunk_by_kanji, has_small_kana, is_kana, normalize_kana

from .utils import (
    BLOCK_TAGS,
    VOID_ELEMENTS,
    get_tag_from_element,
    find_overlap_range,
    generate_offsets,
)
from .annotations import Yomi, TokenizedText, Context


log = logging.getLogger("runekana.document")


class XhtmlDocument:
    """Represents an XHTML document parsed from disk."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        with open(filepath, "rb") as f:
            self.original_bytes = f.read()

        parser = etree.XMLParser(
            remove_blank_text=False, strip_cdata=False, resolve_entities=False
        )
        self.tree = etree.parse(filepath, parser)

    @property
    def root(self) -> etree._Element:
        return self.tree.getroot()

    def paragraphs(self) -> Generator[Paragraph, None, None]:
        """Iterate over leaf-level block elements, yielding ``Paragraph`` objects.

        Only yields block elements whose direct children contain no other block
        elements (i.e. genuine text containers, not layout wrappers).

        .. note::
            Text appearing as the ``.tail`` of block elements (between sibling
            blocks inside a container) is currently ignored.
        """
        # iter() is DFS pre-order, so parents are visited before children
        for elem in self.root.iter():
            if not isinstance(elem.tag, str):
                continue
            tag = etree.QName(elem.tag).localname.lower()
            if tag not in BLOCK_TAGS:
                continue
            # skip layout containers that have block children
            child_tags = {
                etree.QName(c.tag).localname.lower()
                for c in elem
                if isinstance(c.tag, str)
            }
            if child_tags & BLOCK_TAGS:
                continue
            yield Paragraph(elem)

    def save(self) -> None:
        """Serialise the DOM back to disk, preserving the original XML header."""
        self._normalize_empty_tags(self.tree)

        # Serialize ONLY the root to avoid lxml re-inserting DOCTYPE from the tree object.
        # NOTE: This ignores any trailing content (comments, whitespace) after the root
        # element in the original file, which is usually acceptable for EPUB documents.
        body_bytes = etree.tostring(
            self.root,
            encoding="utf-8",
            xml_declaration=False,
            method="xml",
            pretty_print=False,
        )
        header = self._extract_header(self.original_bytes)
        output = header.decode("utf-8") + body_bytes.decode("utf-8")
        with open(self.filepath, "w", encoding="utf-8", newline="") as f:
            f.write(output)

    @staticmethod
    def _normalize_empty_tags(tree: etree._ElementTree) -> None:
        """Ensure non-void elements are never self-closed (set text='' when empty)."""
        for elem in tree.iter():
            if isinstance(elem.tag, str):
                local = etree.QName(elem.tag).localname.lower()
                if local not in VOID_ELEMENTS and elem.text is None and len(elem) == 0:
                    elem.text = ""

    @staticmethod
    def _extract_header(content: bytes) -> bytes:
        """Extract XML declaration / DOCTYPE / comments that precede the root element."""
        match = re.search(b"<(?!\\?|!|!--)", content)
        return content[: match.start()] if match else b""


class Paragraph:
    def __init__(self, element: etree._Element) -> None:
        self.element = element
        self.segments = list(self.parse())
        # build offsets array at class level, so it can be used in tokenizing
        self.start_offsets, self.total_length = self._compute_segment_start_offsets()
        self.end_offsets = self._start_offset_to_end_offset()

    def __str__(self) -> str:
        return self.getstr(self.segments)

    def __repr__(self) -> str:
        return f"{repr(self.segments)}"

    def __getitem__(self, index):
        return self.segments[index]

    def __len__(self):
        return len(self.segments)

    @classmethod
    def getstr(cls, segments: list[Union[TextNode, Ruby]]) -> str:
        return "".join(str(segment) for segment in segments)

    def parse(self) -> Generator[Union[TextNode, Ruby]]:
        context = etree.iterwalk(self.element, events=("start", "end"))
        for event, elem in context:
            tag = get_tag_from_element(elem)
            if event == "start":
                if tag == "ruby":
                    context.skip_subtree()
                    yield Ruby.parse(elem, paragraph=self)
                else:
                    text = elem.text or ""
                    if text.strip():
                        yield TextNode(elem, "text", paragraph=self)
            elif event == "end":
                tail = elem.tail or ""
                if tail.strip():
                    yield TextNode(elem, "tail", paragraph=self)

    def _compute_segment_start_offsets(self) -> tuple[list[int], int]:
        """
        Computes the cumulative character offsets for each segment in the paragraph.

        This generates an array of start indices suitable for binary search (bisect),
        mapping character positions back to their containing segments.

        Returns:
            A tuple of (offsets, total_length), where 'offsets' contains the
            start character index for each segment, and 'total_length' is the
            total character length of the paragraph.
        """
        offsets = [0] * (len(self.segments) + 1)
        for i, seg in enumerate(self.segments):
            offsets[i + 1] = offsets[i] + len(seg)
        total_length = offsets.pop()
        return offsets, total_length

    def _start_offset_to_end_offset(self) -> list[int]:
        return self.start_offsets[1:] + [self.total_length]

    def get_context(
        self,
        segment: Union[TextNode, Ruby],
        forward_max: Optional[int] = None,
        backward_max: Optional[int] = None,
    ) -> Context:
        """
        Retrieves the structured context surrounding a segment within the paragraph.

        Args:
            segment: The segment (TextNode or Ruby) to find context for.
            forward_max: Max characters to include from subsequent segments. If None, include all.
            backward_max: Max characters to include from preceding segments. If None, include all.

        Returns:
            Context: A data object containing the segment's start/end indices relative to the
            reconstructed context string, the original segment, and a structured list of
            (text, yomi) pairs representing the surrounding context.

            Note: The context window "snaps" to Ruby boundaries. If a Ruby object overlaps
            the requested character limits (forward_max/backward_max), it is included
            in its entirety to maintain semantic integrity, and the returned indices
            are adjusted to reflect this expanded window.

        Raises:
            ValueError: If segment is not part of this paragraph.
        """
        # init
        try:
            idx = self.segments.index(segment)
        except ValueError:
            raise ValueError(f"segment {segment!r} not in Paragraph")

        target_start = self.start_offsets[idx]
        target_end = self.end_offsets[idx]
        limit_b = (target_start - backward_max) if backward_max is not None else 0
        limit_f = (
            (target_end + forward_max) if forward_max is not None else self.total_length
        )

        backward_idx, forward_idx = find_overlap_range(
            self.start_offsets,
            self.end_offsets,
            begin=max(0, limit_b),
            end=min(self.total_length, limit_f),
        )

        segments = self.segments[backward_idx:forward_idx]
        context_tuples: list[tuple[str, Optional[str]]] = []
        context_offsets: list[int] = [0]
        for i, seg in enumerate(segments):
            # we are not going to truncate ruby, as it's undoable
            # good luck when some malform epub put the whole book inside ruby tags, lol
            if isinstance(seg, Ruby):
                context_tuples.extend(seg.pairs)
                for base, _ in seg.pairs:
                    context_offsets.append(context_offsets[-1] + len(base))
            elif 0 < i < len(segments) - 1:
                # just append text for non-first and non-last textNode
                context_tuples.append((str(seg), None))
                context_offsets.append(context_offsets[-1] + len(seg))
            else:
                text = str(seg)
                if not i:
                    # clamp: limit_b may fall before this segment's start
                    first_seg_start = max(0, limit_b - self.start_offsets[backward_idx])
                    context_tuples.append((text[first_seg_start:], None))
                    context_offsets.append(
                        context_offsets[-1] + len(text) - first_seg_start
                    )
                else:
                    last_seg_end = limit_f - self.start_offsets[forward_idx - 1]
                    context_tuples.append((text[:last_seg_end], None))
                    context_offsets.append(context_offsets[-1] + last_seg_end)
        context_size = context_offsets.pop()

        # calculate idx
        snap_at_start = isinstance(segments[0], Ruby)
        if snap_at_start:
            start_idx = target_start - self.start_offsets[backward_idx]
        else:
            # actual context start = max(segment start, limit_b floor)
            context_char_start = max(self.start_offsets[backward_idx], max(0, limit_b))
            start_idx = target_start - context_char_start
        start_idx = max(0, start_idx)

        return Context(
            start_idx=start_idx,
            end_idx=start_idx + (target_end - target_start),
            segment=segment,
            context=context_tuples,
            offsets=context_offsets,
            size=context_size,
        )


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
                    if is_kana(text) and is_kana(reading):
                        raise ValueError(
                            f"kana in reading not match kana in surface: {text!r} not in {reading!r}"
                        )
                    else:
                        raise ImpossibleToAlignException(
                            f"surface {text!r} or reading {reading!r} is not pure kana"
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
                    if is_kana(text) and is_kana(reading):
                        raise ValueError(
                            f"kana in reading not match kana in surface: {text!r} not in {reading!r}"
                        )
                    else:
                        raise ImpossibleToAlignException(
                            f"surface {text!r} or reading {reading!r} is not pure kana"
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

    def _lookup_anno_match_reading(
        self,
        reading: str,
        word: str,
        annotations: dict[str, str],
        direction: Literal["ltr", "rtl"] = "ltr",
    ) -> Optional[str]:
        """
        Look up a word's reading in annotations and match it against the reading prefix/suffix.

        This method handles small kana normalization (e.g., matching 'きや' in annotations
        with 'きゃ' in the token reading).

        Args:
            reading: The token reading string to match against.
            word: The kanji word to look up in annotations.
            annotations: Dictionary of existing word-to-reading mappings.
            direction: 'ltr' to match from start, 'rtl' to match from end.

        Returns:
            The matched substring from the input `reading` (possibly containing small kana),
            or None if the word is not found in annotations.

        Raises:
            ImpossibleToAlignException: If the annotation exists but the readings are
                incompatible even after small kana normalization.
        """
        r = annotations.get(word)
        if not r:
            return None

        is_match = reading.startswith(r) if direction == "ltr" else reading.endswith(r)
        if is_match:
            return r

        # in annotations not has common prefix/suffix with reading
        # most like exist ruby normalized small kana
        exist_r = reading[: len(r)] if direction == "ltr" else reading[-len(r) :]
        if has_small_kana(exist_r):
            normalized = jaconv.enlargesmallkana(exist_r)
            if normalized == r:
                return exist_r
            raise ImpossibleToAlignException(
                f"normalized reading {normalized!r} (from {exist_r!r}) still does not match annotation {r!r} for word {word!r}"
            )
        else:
            raise ImpossibleToAlignException(
                f"reading {direction} part {exist_r!r} does not match annotation {r!r} for word {word!r}"
            )

    def _trim_reading_by_lookup_anno(
        self,
        reading: str,
        word: str,
        annotations: Optional[dict[str, str]] = None,
        direction: Literal["ltr", "rtl"] = "ltr",
    ) -> str:
        """
        Trim a reading string by removing parts that match existing annotations for a kanji word.

        If a match for the whole word is found, it is trimmed. Otherwise, it falls back to
        trimming character by character from the specified direction.

        Args:
            reading: The reading string to be trimmed.
            word: The kanji word (surface text) that corresponds to the reading to be removed.
            annotations: Existing annotations to guide the trimming.
            direction: 'ltr' to trim from the beginning, 'rtl' to trim from the end.

        Returns:
            The trimmed reading string.

        Raises:
            ImpossibleToAlignException: If the reading cannot be aligned with the annotations.
        """

        if not word:
            return reading
        annotations = annotations or {}
        if not annotations:
            raise ImpossibleToAlignException(
                f"unable to trim reading from word {word!r} with empty annotations"
            )

        r = self._lookup_anno_match_reading(
            reading, word, annotations, direction=direction
        )
        if r is not None:
            return reading[len(r) :] if direction == "ltr" else reading[: -len(r)]
        else:
            chars = word if direction == "ltr" else reversed(word)
            for ch in chars:
                r = self._lookup_anno_match_reading(
                    reading, ch, annotations, direction=direction
                )
                if r is None:
                    raise ImpossibleToAlignException(
                        f"cannot split reading at {direction} boundary: "
                        f"word={word!r} char={ch!r} remaining={reading!r}"
                    )
                reading = (
                    reading[len(r) :] if direction == "ltr" else reading[: -len(r)]
                )
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
                reading = self._trim_reading_by_lookup_anno(
                    reading, discarded, annotations, direction="ltr"
                )
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
                reading = self._trim_reading_by_lookup_anno(
                    reading, discarded, annotations, direction="rtl"
                )

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

        except ImpossibleToAlignException as e:
            log.info(
                "boundary alignment failed for %r: %s. falling back to isolated tokenisation",
                self.text,
                e,
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
                surface = t.surface[ctx.start_idx - t.begin : ctx.end_idx - t.begin]
                if t.reading and first_reading is not None and last_reading is not None:
                    diff = len(last_reading) - len(normalize_kana(t.reading))
                    reading = first_reading[:diff] if diff < 0 else first_reading
                else:
                    reading = None
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
