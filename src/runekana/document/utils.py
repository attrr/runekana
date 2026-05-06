from typing import Optional, Any
from bisect import bisect_left, bisect_right

from lxml import etree
from lxml.builder import ElementMaker

XHTML_NS = "http://www.w3.org/1999/xhtml"
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

E = ElementMaker(namespace=XHTML_NS)


def get_tag_from_element(elem: etree.Element | None):
    if elem is None:
        return ""
    if not isinstance(elem.tag, str):
        return ""
    return etree.QName(elem).localname.lower()


def generate_offsets(l: list[Any]) -> tuple[list[int], list[int]]:  # noqa: E741
    start_offsets = [0] * (len(l) + 1)
    for i, ele in enumerate(l):
        start_offsets[i + 1] = start_offsets[i] + len(ele)
    total_length = start_offsets.pop()
    return (
        start_offsets,
        derive_end_offsets_from_start(start_offsets, total_size=total_length),
    )


def derive_end_offsets_from_start(
    start_offsets: list[int],
    total_size: Optional[int] = None,
    last_size: Optional[int] = None,
) -> list[int]:
    if not total_size and not last_size:
        raise ValueError("input at least one of total_size & last_size")
    if last_size:
        total_size = start_offsets[-1] + last_size
    return start_offsets[1:] + [total_size]  # type: ignore


def derive_start_offsets_from_end(end_offsets: list[int]) -> tuple[int, list[int]]:
    total_size = end_offsets.pop()
    return total_size, [0] + end_offsets


def find_overlap_range(
    start_offsets: list[int], end_offsets: list[int], begin: int, end: int
) -> tuple[int, int]:
    """
    Find the index range [i, j) of intervals overlapping with [begin, end).

    Uses bisect on paired start/end offset arrays (1:1 index-mapped):
    - bisect_right(end_offsets, begin): first interval whose end > begin
    - bisect_left(start_offsets, end): first interval whose start >= end
      (this matches python's exclusive-end slicing)
    """
    if begin < 0 or end < 0:
        raise ValueError("begin and end can't be negative")

    if begin >= end or not start_offsets:
        return (0, 0)

    start_idx = bisect_right(end_offsets, begin)
    end_idx = bisect_left(start_offsets, end)
    return (start_idx, end_idx)
