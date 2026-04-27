from __future__ import annotations
import csv
import logging
from typing import Optional

import jaconv
from pydantic import BaseModel
import os
import json
import zipfile
import glob
import sqlite3
import re

from sudachipy import Dictionary, SplitMode
from runekana.text import has_kanji

log = logging.getLogger("runekana.text")


class YomitanDB:
    """Manages the local SQLite database for Yomitan frequency dictionaries."""

    def __init__(self):
        state_home = os.environ.get(
            "XDG_STATE_HOME", os.path.expanduser("~/.local/state")
        )
        db_dir = os.path.join(state_home, "runekana")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, "frequency.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS frequency (word TEXT PRIMARY KEY, rank INTEGER)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rank ON frequency(rank)")
            conn.commit()

    @classmethod
    def _extract_rank(cls, meta) -> int:
        """Recursively extract integer rank from chaotic Yomitan metadata schemas."""
        if isinstance(meta, (int, float)):
            return int(meta)
        if isinstance(meta, str):
            m = re.search(r"\d+", meta)
            return int(m.group(0)) if m else -1

        if isinstance(meta, dict):
            val = meta.get("value")
            if val is not None:
                return cls._extract_rank(val)

            freq = meta.get("frequency")
            if freq is not None:
                return cls._extract_rank(freq)

            for v in meta.values():
                r = cls._extract_rank(v)
                if r != -1:
                    return r

        return -1

    @classmethod
    def _parse_bank(cls, data: list) -> list[tuple[str, int]]:
        """Parse a single term_meta_bank array."""
        results = []
        for entry in data:
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            word = entry[0]
            meta = entry[2]
            rank = cls._extract_rank(meta)
            if rank != -1:
                results.append((word, rank))
        return results

    def import_dict(self, path: str):
        """Import Yomitan frequency dict from a ZIP file or directory to SQLite."""
        all_entries = []

        if os.path.isdir(path):
            bank_files = glob.glob(os.path.join(path, "term_meta_bank_*.json"))
            for fpath in bank_files:
                try:
                    with open(fpath, "rb") as f:
                        all_entries.extend(self._parse_bank(json.load(f)))
                except Exception as e:
                    log.warning("Failed to parse %s: %s", fpath, e)

        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist():
                    if not (
                        name.startswith("term_meta_bank_") and name.endswith(".json")
                    ):
                        continue
                    try:
                        with zf.open(name) as f:
                            all_entries.extend(self._parse_bank(json.load(f)))
                    except Exception as e:
                        log.warning("Failed to parse %s from zip: %s", name, e)
        else:
            log.error("Provided path is neither a directory nor a ZIP file: %s", path)
            return

        if not all_entries:
            log.warning("No frequency data found or parsing failed.")
            return

        # Deduplicate and keep minimum rank per word
        min_ranks = {}
        for w, r in all_entries:
            if w not in min_ranks or r < min_ranks[w]:
                min_ranks[w] = r
        unique_entries = list(min_ranks.items())

        with sqlite3.connect(self.db_path) as conn:
            log.info(
                "Clearing old frequency data and inserting %d unique entries...",
                len(unique_entries),
            )
            conn.execute("DELETE FROM frequency")
            conn.executemany(
                "INSERT INTO frequency (word, rank) VALUES (?, ?)", unique_entries
            )
            conn.commit()
        log.info("Successfully imported frequency data to %s", self.db_path)

    def get_top_n(self, n: int) -> set[str]:
        """Retrieve top-N most frequent words from local SQLite cache."""
        if not os.path.exists(self.db_path):
            log.warning(
                "Frequency database not found. Skipping words will be disabled. Use --freq-dict to import a Yomitan dict."
            )
            return set()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT word FROM frequency ORDER BY rank ASC LIMIT ?", (n,)
            )
            words = {row[0] for row in cursor}

        log.info("Loaded skip set: top %d frequent words from cache", len(words))
        return words


def load_local_dict(path: Optional[str]) -> dict[str, str]:
    """Load local dictionary. Format: 漢字<TAB>よみがな per line."""
    if not path or not os.path.exists(path):
        return {}
    d = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            # Filter comments and empty lines
            filtered = (line for line in f if line.strip() and not line.startswith("#"))
            reader = csv.reader(filtered, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    d[row[0]] = row[1]
    except Exception as e:
        log.warning("Failed to load local dict %s: %s", path, e)
    log.info("Loaded %d entries from local dictionary %s", len(d), path)
    return d


def save_local_dict(path: str, local_dict: dict[str, str]):
    """Save local dictionary sorted by key."""
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for w, r in sorted(local_dict.items()):
                writer.writerow([w, r])
    except OSError as e:
        log.warning("Could not save local dict %s: %s", path, e)


class Token(BaseModel):
    surface: str
    reading: Optional[str] = None
    to_verify: bool = False


class Tokenizer:
    """Sudachi-based tokenizer with support for compound word modes."""

    def __init__(
        self,
        skip_words: set[str],
        local_dict: dict[str, str],
    ) -> None:
        self.dict = Dictionary(dict="full")
        self.tokenizer = self.dict.create()
        self.skip_words = skip_words
        self.local_dict = local_dict

    def is_ambiguous(self, surface: str) -> bool:
        """
        Check if a surface form has multiple possible readings in the dictionary,
        or if it's completely out-of-vocabulary (OOV).
        """
        entries = self.dict.lookup(surface)
        # Unique readings based on Hiragana representation
        unique_readings = {jaconv.kata2hira(e.reading_form()) for e in entries}
        return len(unique_readings) != 1

    def tokenize(self, text: str) -> list[Token]:
        """
        Tokenize text and return a list of Token objects.
        Uses Sudachi's normalized reading form (SplitMode.C).
        """
        results = []

        for m in self.tokenizer.tokenize(text, SplitMode.C):
            surface = m.surface()
            reading = None
            to_verify = False

            if not has_kanji(surface):
                pass
            elif surface in self.skip_words or (m.dictionary_form() in self.skip_words):
                pass
            elif m.part_of_speech()[1] == "数詞" and not self.is_ambiguous(surface):
                pass
            elif surface in self.local_dict:
                reading = self.local_dict[surface]
                to_verify = self.is_ambiguous(surface)
            else:
                reading = jaconv.kata2hira(m.reading_form())
                to_verify = self.is_ambiguous(surface)

            results.append(Token(surface=surface, reading=reading, to_verify=to_verify))

        return results
