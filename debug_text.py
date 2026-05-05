#!/usr/bin/env python3
import sys
import logging
import sqlite3
import jaconv
from rich import box
from rich.console import Console
from rich.table import Table
from src.runekana.tokenizer import YomitanDB, Tokenizer
from src.runekana.text import split_okurigana, has_kanji

# Set up simple logging to stderr
logging.basicConfig(level=logging.WARNING)
console = Console()


def debug_text(text: str):
    # 1. Initialize core components
    db = YomitanDB()
    skip_words = db.get_top_n(1500)

    tokenizer = Tokenizer(skip_words, {})

    table = Table(title=f"Analysis: [bold cyan]{text}[/bold cyan]", box=box.SIMPLE_HEAD)
    table.add_column("Surface", style="magenta", no_wrap=True)
    table.add_column("Reading", style="green")
    table.add_column("Rank", style="yellow", justify="right")
    table.add_column("POS", style="dim")
    table.add_column("Action", style="bold")
    table.add_column("OKU Split", style="bold white")

    # 2. Process tokens
    for t in tokenizer.tokenize(text):
        m = t.morpheme
        surface = t.surface
        lemma = m.dictionary_form()
        pos = "/".join(m.part_of_speech()[:2])  # Keep POS concise

        # Check Rank in DB using MIN(rank)
        rank = -1
        try:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    "SELECT MIN(rank) FROM frequency WHERE word = ? OR word = ?",
                    (surface, lemma),
                )
                row = cursor.fetchone()
                if row and row[0] is not None:
                    rank = row[0]
        except Exception:
            pass

        rank_str = str(rank) if rank != -1 else "[dim]N/A[/dim]"

        # Determine action (skipped, to verify, etc)
        action = "[dim]N/A[/dim]"
        reading_to_display = t.reading
        if t.reading is None and has_kanji(surface):
            action = "[bold red]SKIPPED[/bold red]"
            reading_to_display = jaconv.kata2hira(m.reading_form())
        elif not has_kanji(surface):
            action = "[dim]KANA[/dim]"
        elif t.to_verify:
            action = "[bold yellow]VERIFY[/bold yellow]"
        else:
            action = "[bold green]ANNOTATE[/bold green]"

        reading_str = reading_to_display if reading_to_display else "[dim]N/A[/dim]"
        if t.reading is None and has_kanji(surface):
            reading_str = f"[dim]{reading_str}[/dim]"

        # Okurigana Split
        split_res = "[dim]N/A[/dim]"
        if has_kanji(surface) and reading_to_display:
            segments = split_okurigana(surface, reading_to_display)
            parts = []
            for s, r in segments:
                if r:
                    parts.append(f"[bold yellow]{s}[/bold yellow]([green]{r}[/green])")
                else:
                    parts.append(s)
            split_res = "".join(parts)
            if len(segments) == 1 and segments[0][1] == reading_to_display:
                split_res = f"{split_res}[bold red][GROUP][/bold red]"

        table.add_row(surface, reading_str, rank_str, pos, action, split_res)

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
