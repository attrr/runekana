import os
import json
import pytest
import tempfile
from pathlib import Path
from runekana.tokenizer import (
    YomitanDB,
    load_local_dict,
    save_local_dict,
    Tokenizer,
)
from runekana.text import has_kanji


@pytest.fixture
def temp_xdg_home(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("XDG_STATE_HOME", tmpdir)
        yield tmpdir


def test_yomitandb_rank_extraction():
    db = YomitanDB()
    # Test chaotic Yomitan metadata schemas
    assert db._extract_rank(10) == 10
    assert db._extract_rank("rank 50") == 50
    assert db._extract_rank({"value": 100}) == 100
    assert db._extract_rank({"frequency": {"value": 200}}) == 200
    assert db._extract_rank({"other": "foo", "nested": 300}) == 300
    assert db._extract_rank("no rank") == -1


def test_yomitandb_import_and_get(temp_xdg_home):
    db = YomitanDB()

    # Create a dummy Yomitan bank file
    bank_data = [
        ["食べる", "v5b", {"value": 10}],
        ["寝る", "v1", 20],
        ["走る", "v5r", "rank 5"],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        bank_path = Path(tmpdir) / "term_meta_bank_1.json"
        with open(bank_path, "w", encoding="utf-8") as f:
            json.dump(bank_data, f)

        db.import_dict(tmpdir)

        # Test retrieval
        top_1 = db.get_top_n(1)
        assert top_1 == {"走る"}

        top_2 = db.get_top_n(2)
        assert top_2 == {"走る", "食べる"}

        top_10 = db.get_top_n(10)
        assert len(top_10) == 3


def test_local_dict_io():
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".tsv", encoding="utf-8"
    ) as tmp:
        tmp.write("# Comment\n")
        tmp.write("漢字\tかんじ\n")
        tmp.write("食べる\tたべる\n")
        tmp_path = tmp.name

    try:
        d = load_local_dict(tmp_path)
        assert d == {"漢字": "かんじ", "食べる": "たべる"}

        d["新しい"] = "あたらしい"
        save_local_dict(tmp_path, d)

        d2 = load_local_dict(tmp_path)
        assert d2 == {"漢字": "かんじ", "食べる": "たべる", "新しい": "あたらしい"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_tokenizer_basic():
    # Requires sudachidict-full to be installed
    tok = Tokenizer(skip_words=set(), local_dict={})
    results = tok.tokenize("私は漢字が大好きです。")

    surfaces = [t.surface for t in results]
    assert "私" in surfaces
    assert "漢字" in surfaces

    # Check if 漢字 has a reading
    kanji_token = next(t for t in results if t.surface == "漢字")
    assert kanji_token.reading == "かんじ"


def test_tokenizer_skip_words():
    tok = Tokenizer(skip_words={"私", "大好き"}, local_dict={})
    results = tok.tokenize("私は大好きです。")

    watashi = next(t for t in results if t.surface == "私")
    assert watashi.reading is None  # Skipped

    daisuki = next(t for t in results if t.surface == "大好き")
    assert daisuki.reading is None  # Skipped


def test_tokenizer_local_dict():
    # Force a specific reading via local_dict
    tok = Tokenizer(skip_words=set(), local_dict={"身体": "からだ"})
    results = tok.tokenize("身体を動かす。")

    karada = next(t for t in results if t.surface == "身体")
    assert karada.reading == "からだ"


def test_tokenizer_ambiguity():
    tok = Tokenizer(skip_words=set(), local_dict={})

    # '身体' is usually ambiguous in Sudachi (shintai, karada)
    assert tok.is_ambiguous("身体") is True

    # '私' (watashi, watakushi) might be ambiguous depending on dict
    # But 'あ' should not be ambiguous
    assert tok.is_ambiguous("あ") is False


def test_tokenizer_to_verify_flag():
    tok = Tokenizer(skip_words=set(), local_dict={})
    results = tok.tokenize("身体")

    karada = next(t for t in results if t.surface == "身体")
    # Should be true because '身体' has multiple readings
    assert karada.to_verify is True

    results2 = tok.tokenize("食べる")
    taberu = next(t for t in results2 if t.surface == "食べる")
    # '食べる' is typically not ambiguous in reading
    assert taberu.to_verify is False


def test_tokenizer_compound_words():
    """
    Test how the tokenizer handles compound words that Sudachi (SplitMode.C)
    keeps as a single token, such as '打ち合わせ'.
    """
    tok = Tokenizer(skip_words=set(), local_dict={})
    results = tok.tokenize("今日の打ち合わせは午後からです。")

    # Verify '打ち合わせ' is a single token in SplitMode.C
    token = next(t for t in results if "打ち合わせ" in t.surface)
    assert token.surface == "打ち合わせ"
    assert token.reading == "うちあわせ"

    # Test another compound word
    results2 = tok.tokenize("引っ越し作業")
    hikkoshi = next(t for t in results2 if "引っ越し" in t.surface)
    assert hikkoshi.surface == "引っ越し"
    assert hikkoshi.reading == "ひっこし"

    # Check ambiguity for these compound words
    assert tok.is_ambiguous("打ち合わせ") is False
    assert tok.is_ambiguous("引っ越し") is False


def test_tokenizer_compound_skip_prevention():
    """
    Ensure kanji tokens in a compound are not skipped even if part of skip_words.
    Example: '日本' in '全日本'.
    """
    tok = Tokenizer(skip_words={"日本"}, local_dict={})

    results = tok.tokenize("全日本")

    for token in results:
        if has_kanji(token.surface):
            assert token.reading is not None


def test_tokenizer_numeric_handling():
    """Test handling of numerals: unambiguous numerals should be skipped."""
    tok = Tokenizer(skip_words=set(), local_dict={})

    # '100円'
    results = tok.tokenize("100円")
    num_token = next(t for t in results if "100" in t.surface)
    assert num_token.reading is None

    # '三千円'
    results2 = tok.tokenize("三千円")
    sen_token = next(t for t in results2 if "三千" in t.surface)
    if not tok.is_ambiguous(sen_token.surface):
        assert sen_token.reading is None


def test_tokenizer_complex_ambiguity():
    """Test complex ambiguous words to ensure to_verify flag is correct."""
    tok = Tokenizer(skip_words=set(), local_dict={})

    # '行方' is ambiguous
    assert tok.is_ambiguous("行方") is True

    # '明日' is ambiguous
    assert tok.is_ambiguous("明日") is True

    # OOV words should be treated as ambiguous
    assert tok.is_ambiguous("𠮷野家") is True


def test_tokenizer_mixed_sentence():
    """Test a mixed sentence with kana, kanji, and symbols."""
    tok = Tokenizer(skip_words={"です", "ます"}, local_dict={})
    text = "最新の技術を駆使して、高品質なサービスを提供します。"
    results = tok.tokenize(text)

    surfaces = [t.surface for t in results]
    assert "最新" in surfaces
    assert "技術" in surfaces
    assert "提供" in surfaces

    teikyo = next(t for t in results if t.surface == "提供")
    assert teikyo.reading == "ていきょう"
