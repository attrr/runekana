from __future__ import annotations
import logging
import re
from typing import Optional, Union, Generator

from lxml import etree

from .nodes import TextNode, Ruby
from .context import Context
from .utils import get_tag_from_element, find_overlap_range


log = logging.getLogger("runekana.document")

BLOCK_TAGS = {
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "section",
}
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
