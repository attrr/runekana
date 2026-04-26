#!/usr/bin/env python3
import sys
import logging
import sqlite3
import rich
from rich import box
from rich.console import Console
from rich.table import Table
from src.runekana.tokenizer import Tokenizer, YomitanDB
from src.runekana.text import split_okurigana, has_kanji
from sudachipy import Dictionary, SplitMode

# Set up simple logging to stderr
logging.basicConfig(level=logging.WARNING)
console = Console()


def debug_text(text: str):
    # 1. Initialize core components
    db = YomitanDB()
    # We use a dummy skip set to show ranks even for frequent words
    tokenizer_obj = Tokenizer(skip_words=set(), local_dict={})

    sudachi_dict = Dictionary(dict="full")
    sudachi_tokenizer = sudachi_dict.create()

    table = Table(title=f"Analysis: [bold cyan]{text}[/bold cyan]", box=box.SIMPLE_HEAD)
    table.add_column("Surface", style="magenta", no_wrap=True)
    table.add_column("Reading", style="green")
    table.add_column("Rank", style="yellow", justify="right")
    table.add_column("POS", style="dim")
    table.add_column("OKU Split", style="bold white")

    # 2. Process tokens
    for m in sudachi_tokenizer.tokenize(text, SplitMode.C):
        surface = m.surface()
        reading_raw = m.reading_form()
        # simple kata2hira conversion
        reading = "".join(
            [
                chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch
                for ch in reading_raw
            ]
        )
        lemma = m.dictionary_form()
        pos = "/".join(m.part_of_speech()[:2])  # Keep POS concise

        # Check Rank in DB
        rank = -1
        try:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    "SELECT rank FROM frequency WHERE word = ? OR word = ?",
                    (surface, lemma),
                )
                row = cursor.fetchone()
                if row:
                    rank = row[0]
        except Exception:
            pass

        rank_str = str(rank) if rank != -1 else "[dim]N/A[/dim]"

        # Okurigana Split
        split_res = "[dim]N/A[/dim]"
        if has_kanji(surface):
            segments = split_okurigana(surface, reading)
            parts = []
            for s, r in segments:
                if r:
                    parts.append(f"[bold yellow]{s}[/bold yellow]([green]{r}[/green])")
                else:
                    parts.append(s)
            split_res = "".join(parts)
            if len(segments) == 1 and segments[0][1] == reading:
                split_res = f"{split_res}[bold red][GROUP][/bold red]"

        table.add_row(surface, reading, rank_str, pos, split_res)

    console.print(table)
    console.print(
        "[dim]Note: [bold red][GROUP][/bold red] indicates fallback to whole-word (Group Ruby) annotation.[/dim]"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print(
            '[red]Usage:[/red] python3 debug_text.py "あなたの日本語テキスト"'
        )
        sys.exit(1)

    debug_text(sys.argv[1])
