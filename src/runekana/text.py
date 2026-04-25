import re
import logging
from typing import Optional

import jaconv

log = logging.getLogger(__name__)

# regex
kanji_pattern = r"[\u4e00-\u9fff\u3400-\u4dbf々〇]"
# IMPORTANT: The capture group () is required so re.split keeps the kanji chunks in the output array
kanji_regex = re.compile(rf"({kanji_pattern}+)")
kanji_regex_once = re.compile(kanji_pattern)


# text helper fucntion
def has_kanji(text: str) -> bool:
    """Check if the string contains any Kanji characters."""
    return bool(kanji_regex_once.search(text))


def chunk_by_kanji(text: str) -> list[tuple[str, bool]]:
    """Split text into chunks, tagging each as kanji (True) or not (False)."""
    return [
        (part, bool(kanji_regex.fullmatch(part)))
        for part in kanji_regex.split(text)
        if part
    ]


def split_okurigana(surface: str, reading: str) -> list[tuple[str, Optional[str]]]:
    """
    Split a word into (text, ruby_or_None) segments.

    食べる + たべる → [("食", "た"), ("べる", None)]
    聞き取る + ききとる → [("聞", "き"), ("き", None), ("取", "と"), ("る", None)]
    """
    if not reading or not has_kanji(surface):
        return [(surface, None)]

    chunks = chunk_by_kanji(surface)
    chunks = chunks[::-1]
    remaining = jaconv.kata2hira(reading)

    segments: list[tuple[str, Optional[str]]] = []
    for idx, (chunk, is_kanji) in enumerate(chunks):
        if not is_kanji:
            # Non-kanji: strip from back of remaining reading, emit bare
            if not remaining.endswith(chunk):
                log.debug("Kana mismatch: remaining=%r chunk=%r", remaining, chunk)
                return [(surface, None)]

            remaining = remaining[: -len(chunk)]
            segments.insert(0, (chunk, None))
            continue

        # Kanji chunk: find where the previous non-kanji anchor starts in remaining
        anchor = None
        if idx + 1 < len(chunks):
            anchor = jaconv.kata2hira(chunks[idx + 1][0])

        if anchor is not None:
            # Skip at least len(chunk) chars (1 reading char per kanji minimum)
            pos = remaining.rfind(anchor)
            if pos < 0:
                log.debug(
                    "Okurigana sync failed: surface=%r reading=%r "
                    "anchor=%r remaining=%r",
                    surface,
                    reading,
                    anchor,
                    remaining,
                )
                return [(surface, None)]
            anchor_end = pos + len(anchor)
            kanji_reading = remaining[anchor_end:]
            remaining = remaining[:anchor_end]
        else:
            # Fist chunk, take everything left
            kanji_reading = remaining
            remaining = ""

        segments.insert(0, (chunk, kanji_reading or None))

    # Validate: nothing left over
    if remaining:
        log.debug(
            "Okurigana reading not fully consumed: surface=%r reading=%r "
            "leftover=%r",
            surface,
            reading,
            remaining,
        )
        return [(surface, None)]

    return segments
