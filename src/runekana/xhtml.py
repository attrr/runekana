import re
import html
import logging
from typing import Optional, cast

from lxml import etree
import jaconv
from .text import has_kanji, split_okurigana
from .tokenizer import Tokenizer, Token

log = logging.getLogger(__name__)


XHTML_NS = "http://www.w3.org/1999/xhtml"

# Tags whose text content we should never process
SKIP_TAGS = frozenset(
    [
        f"{{{XHTML_NS}}}rt",
        f"{{{XHTML_NS}}}rp",
        f"{{{XHTML_NS}}}ruby",
        f"{{{XHTML_NS}}}script",
        f"{{{XHTML_NS}}}style",
        "rt",
        "rp",
        "ruby",
        "script",
        "style",
    ]
)

# XHTML void elements — should remain self-closing
VOID_ELEMENTS = frozenset(
    [
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
        "param",
        "source",
        "track",
        "wbr",
    ]
)


def segments_to_xml(segments: list[tuple[str, Optional[str]]]) -> str:
    """Convert segments to XHTML ruby markup string (with proper HTML escaping)."""
    parts = []
    for text, ruby in segments:
        escaped_text = html.escape(text)
        if ruby and ruby != jaconv.kata2hira(text):
            escaped_ruby = html.escape(ruby)
            parts.append(f"<ruby>{escaped_text}<rt>{escaped_ruby}</rt></ruby>")
        else:
            parts.append(escaped_text)
    return "".join(parts)


def tokens_to_markup(tokens: list[Token]) -> Optional[str]:
    """Convert a list of tokens to XHTML ruby markup string. Returns None if no annotations."""
    if not any(token.reading is not None for token in tokens):
        return None

    parts = []
    for token in tokens:
        surface = token.surface
        reading = token.reading
        if reading is not None:
            segments = split_okurigana(surface, reading)
            parts.append(segments_to_xml(segments))
        else:
            parts.append(html.escape(surface))
    return "".join(parts)


def _replace_text_with_markup(elem, attr: str, markup: str):
    """
    Replace elem.text or elem.tail with parsed XHTML markup.
    Parses the ruby-annotated string into lxml nodes and splices them
    into the tree without disturbing surrounding structure.
    """
    wrapper_xml = f'<span xmlns="{XHTML_NS}">{markup}</span>'
    try:
        wrapper = etree.fromstring(wrapper_xml.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        log.warning("Failed to parse generated markup: %s — skipping", e)
        return

    if attr == "text":
        elem.text = wrapper.text or ""
        # Insert parsed ruby nodes before existing children
        for i, child in enumerate(wrapper):
            elem.insert(i, child)
    elif attr == "tail":
        parent = elem.getparent()
        if parent is None:
            return
        idx = list(parent).index(elem)
        elem.tail = wrapper.text or ""
        for i, child in enumerate(wrapper):
            parent.insert(idx + 1 + i, child)


def should_skip_element(elem) -> bool:
    """Determine whether an XML element should be ignored during ruby injection."""
    tag = elem.tag if isinstance(elem.tag, str) else ""
    return tag in SKIP_TAGS


def get_block_text(elem) -> str:
    """Find the nearest block-level container and extract all its text."""
    block_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "body", "html"}
    current = elem
    while current is not None:
        tag = current.tag.lower() if isinstance(current.tag, str) else ""
        if tag.split("}")[-1] in block_tags:
            break
        current = current.getparent()

    if current is None:
        current = elem

    text = "".join(current.itertext())
    return re.sub(r"\s+", " ", text).strip()


def _process_node_text(
    target_elem,
    attr: str,
    text: str,
    context_node,
    tokenizer: Tokenizer,
    tasks: list,
    candidates: list,
):
    """
    Tokenize a specific text string within an element and queue it for deferred injection.

    Populates the shared `tasks` list with the tokenized data and the `candidates` list
    with any tokens requiring external LLM verification, including their contextual snippets.
    """
    if not text or not text.strip() or not has_kanji(text):
        return
    tokens = tokenizer.tokenize(text)
    tasks.append((target_elem, attr, tokens))

    full_context = get_block_text(context_node)
    search_start_idx = 0
    for token in tokens:
        if token.to_verify and token.reading is not None:
            surface = token.surface
            idx = full_context.find(surface, search_start_idx)
            if idx == -1:  # Fallback if word not found after offset
                idx = full_context.find(surface)
            if idx != -1:
                start = max(0, idx - 30)
                end = min(len(full_context), idx + len(surface) + 30)
                context_str = full_context[start:end]
                search_start_idx = idx + len(surface)
            else:
                context_str = full_context[:60]

            candidates.append(
                {
                    "word": surface,
                    "reading": token.reading,
                    "context": context_str,
                    "token_ref": token,
                }
            )


def prepare_ruby_injection(
    elem, tokenizer: Tokenizer, inside_ruby: bool = False
) -> tuple[list[tuple[etree._Element, str, list[Token]]], list[dict]]:
    """
    Traverse element tree to tokenize text and collect tasks and candidates.
    Returns:
      - injection_tasks: list of (elem, attr_name, tokens)
      - candidates: list of dicts {"word", "reading", "context", "token_ref"}
    """
    tasks = []
    candidates = []

    tag = elem.tag if isinstance(elem.tag, str) else ""
    is_ruby = inside_ruby or tag in (f"{{{XHTML_NS}}}ruby", "ruby")

    for child in list(elem):
        if not should_skip_element(child):
            child_tasks, child_cands = prepare_ruby_injection(
                child, tokenizer, inside_ruby=is_ruby
            )
            tasks.extend(child_tasks)
            candidates.extend(child_cands)

    if is_ruby or should_skip_element(elem):
        return tasks, candidates

    if elem.text:
        _process_node_text(elem, "text", elem.text, elem, tokenizer, tasks, candidates)
    for child in list(elem):
        if child.tail:
            _process_node_text(
                child, "tail", child.tail, elem, tokenizer, tasks, candidates
            )

    return tasks, candidates


def apply_ruby_injections(tasks: list[tuple[etree._Element, str, list[Token]]]):
    """Iterate over prepared injection tasks and update the XML tree."""
    for elem, attr, tokens in tasks:
        markup = tokens_to_markup(tokens)
        if markup is not None:
            _replace_text_with_markup(elem, attr, markup)


def _fix_self_closing(m: re.Match[str]) -> str:
    """Expand self-closing tags for non-void XHTML elements."""
    tag = m.group(1).split()[0]
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if tag.lower() in VOID_ELEMENTS:
        return m.group(0)
    return m.group(0)[:-2] + "></" + tag + ">"


def serialize_xhtml(tree, original_bytes: bytes) -> str:
    """Serialize lxml tree to XHTML, preserving preamble and line endings."""
    orig_str = original_bytes.decode("utf-8", errors="replace")

    xml_decl = ""
    if orig_str.startswith("<?xml"):
        match_end = orig_str.find("?>")
        if match_end != -1:
            xml_decl = orig_str[: match_end + 2]

    doctype = ""
    dt_match = re.search(r"<!DOCTYPE[^>]*>", orig_str[:500], re.IGNORECASE)
    if dt_match:
        doctype = dt_match.group(0)

    output = cast(
        str,
        etree.tostring(
            tree, xml_declaration=False, encoding="unicode", pretty_print=False
        ),
    )
    output = re.sub(r"<([^>]+)/>", _fix_self_closing, output)
    output = re.sub(
        r"<((?:br|hr|img|meta|link|input|col|area|base|embed|param|source|track|wbr)(?:\s[^>]*)?)\s*/>",
        r"<\1 />",
        output,
    )

    if xml_decl:
        output = xml_decl + "\n" + output
    if doctype and doctype not in output:
        if xml_decl:
            output = output.replace(
                xml_decl + "\n", xml_decl + "\n" + doctype + "\n", 1
            )
        else:
            output = doctype + "\n" + output

    if b"\r\n" in original_bytes:
        output = output.replace("\r", "")
        output = output.replace("\n", "\r\n")

    return output
