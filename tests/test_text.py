import pytest
from runekana.text import split_okurigana, has_kanji, chunk_by_kanji, normalize_kana


@pytest.mark.parametrize(
    "surface, reading, expected",
    [
        # Basic cases
        ("食べる", "たべる", [("食", "た"), ("べる", None)]),
        (
            "聞き取る",
            "ききとる",
            [("聞", "き"), ("き", None), ("取", "と"), ("る", None)],
        ),
        # Complex okurigana
        (
            "打ち合わせる",
            "うちあわせる",
            [("打", "う"), ("ち", None), ("合", "あ"), ("わせる", None)],
        ),
        (
            "引っ越す",
            "ひっこす",
            [("引", "ひ"), ("っ", None), ("越", "こ"), ("す", None)],
        ),
        ("見受けられる", "みうけられる", [("見受", "みう"), ("けられる", None)]),
        # Kanji only
        ("漢字", "かんじ", [("漢字", "かんじ")]),
        ("東京", "とうきょう", [("東京", "とうきょう")]),
        # No kanji
        ("あいうえお", "あいうえお", [("あいうえお", None)]),
        ("カタカナ", "かたかな", [("カタカナ", None)]),
        # Katakana in surface
        ("ページ目", "ぺーじめ", [("ページ", None), ("目", "め")]),
        (
            "サボる",
            "さぼる",
            [("サボる", None)],
        ),  # No kanji, so returns (surface, None)
        # Special characters
        ("佐々木", "ささき", [("佐々木", "ささき")]),
        ("一〇〇", "ひゃく", [("一〇〇", "ひゃく")]),
        # Prefixes/Suffixes
        ("お酒", "おさけ", [("お", None), ("酒", "さけ")]),
        ("読み方", "よみかた", [("読", "よ"), ("み", None), ("方", "かた")]),
        # Mismatches (fallback cases)
        ("食べる", "およぐ", [("食べる", "およぐ")]),  # Reading mismatch
        ("食べる", "たべます", [("食べる", "たべます")]),  # Leftover reading
        ("食べる", "た", [("食べる", "た")]),  # Incomplete reading
        # Empty inputs
        ("", "あ", [("", None)]),
        ("漢", "", [("漢", None)]),
    ],
)
def test_split_okurigana(surface, reading, expected):
    assert split_okurigana(surface, reading) == expected


def test_split_okurigana_normalized_match():
    # Katakana in reading should match Hiragana in surface if normalized
    # But split_okurigana normalizes reading to Hiragana via jaconv.kata2hira
    # So surface 'べる' (hiragana) should match reading 'ベル' (katakana)
    assert split_okurigana("食べる", "タベル") == [("食", "た"), ("べる", None)]

    # Surface 'ページ' (katakana) should match reading 'ぺーじ' (hiragana)
    assert split_okurigana("ページ目", "ぺーじめ") == [("ページ", None), ("目", "め")]


def test_split_okurigana_multiple_kanji_chunks():
    # 割り当てる (war-i-a-te-ru)
    # surface: 割(K) り(N) 当(K) てる(N)
    # reading: わりあてる
    assert split_okurigana("割り当てる", "わりあてる") == [
        ("割", "わ"),
        ("り", None),
        ("当", "あ"),
        ("てる", None),
    ]


def test_has_kanji():
    assert has_kanji("漢字") is True
    assert has_kanji("漢") is True
    assert has_kanji("々") is True
    assert has_kanji("〇") is True
    assert has_kanji("あいうえお") is False
    assert has_kanji("ABC") is False
    assert has_kanji("") is False


def test_chunk_by_kanji():
    assert chunk_by_kanji("食べる") == [("食", True), ("べる", False)]
    assert chunk_by_kanji("聞き取る") == [
        ("聞", True),
        ("き", False),
        ("取", True),
        ("る", False),
    ]
    assert chunk_by_kanji("漢字") == [("漢字", True)]
    assert chunk_by_kanji("あいうえお") == [("あいうえお", False)]
    assert chunk_by_kanji("お酒") == [("お", False), ("酒", True)]
    assert chunk_by_kanji("") == []


def test_normalize_kana():
    assert normalize_kana("カタカナ") == "かたかな"
    assert normalize_kana("ひらがな") == "ひらがな"
    assert normalize_kana("ページ") == "ぺーじ"
    assert normalize_kana("") == ""
