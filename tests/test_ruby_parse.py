import pytest
from lxml import etree
from runekana.document.nodes import Ruby


def parse_ruby(html_str):
    # Ruby._parse expects an etree._Element
    # We use a parser that doesn't strip blank text to be safe,
    # though Ruby._parse uses .strip() on base and annotation.
    parser = etree.XMLParser(remove_blank_text=False)
    # We might need a namespace if the code expects it,
    # but get_tag_from_element uses QName(elem).localname which is robust.
    ruby_elem = etree.fromstring(html_str, parser)
    return Ruby._parse(ruby_elem)


@pytest.mark.parametrize(
    "html, expected",
    [
        # Happy path: only <rb> <rt>, or text/tail with rt
        ("<ruby>漢字<rt>かんじ</rt></ruby>", [("漢字", "かんじ")]),
        ("<ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby>", [("漢", "かん"), ("字", "じ")]),
        ("<ruby><rb>漢字</rb><rt>かんじ</rt></ruby>", [("漢字", "かんじ")]),
        (
            "<ruby><rb>漢</rb><rt>かん</rt><rb>字</rb><rt>じ</rt></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        # With rp
        ("<ruby>漢字<rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby>", [("漢字", "かんじ")]),
        (
            "<ruby>漢<rp>(</rp><rt>かん</rt><rp>)</rp>字<rp>(</rp><rt>じ</rt><rp>)</rp></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        (
            "<ruby><rb>漢</rb><rp>(</rp><rt>かん</rt><rp>)</rp><rb>字</rb><rp>(</rp><rt>じ</rt><rp>)</rp></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        (
            "<ruby>漢<rp>(</rp><rt>かん</rt><rp>)</rp>字<rt>じ</rt></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        # With junk tags like span
        ("<ruby><span>漢</span>字<rt>かんじ</rt></ruby>", [("漢字", "かんじ")]),
        ("<ruby>漢<span>字</span><rt>かんじ</rt></ruby>", [("漢字", "かんじ")]),
        (
            "<ruby><span>漢</span><span>字</span><rt>かんじ</rt></ruby>",
            [("漢字", "かんじ")],
        ),
        (
            "<ruby><span>漢</span><rp>(</rp><rt>かん</rt><rp>)</rp><span>字</span><rp>(</rp><rt>じ</rt><rp>)</rp></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        (
            "<ruby><span class='foo'>漢字</span><rt>かんじ</rt></ruby>",
            [("漢字", "かんじ")],
        ),
        (
            "<ruby><span>漢</span><rt>かん</rt><span>字</span><rt>じ</rt></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        (
            "<ruby><b>漢</b><rp>(</rp><rt>かん</rt><rp>)</rp><i>字</i><rt>じ</rt></ruby>",
            [("漢", "かん"), ("字", "じ")],
        ),
        (
            "<ruby><span>漢</span>字<rt>かん</rt>字<span>漢</span><rt>じ</rt></ruby>",
            [("漢字", "かん"), ("字漢", "じ")],
        ),
        # Edge cases
        ("<ruby>漢字</ruby>", []),
        ("<ruby><rt>none</rt>漢字<rt>かんじ</rt></ruby>", [("漢字", "かんじ")]),
    ],
)
def test_ruby_parse_variants(html, expected):
    assert parse_ruby(html) == expected


def test_ruby_parse_with_namespace():
    # Test if it works with namespaces as well, since it's likely to be used in XHTML
    xhtml_ns = "http://www.w3.org/1999/xhtml"
    html = f'<ruby xmlns="{xhtml_ns}">漢字<rt>かんじ</rt></ruby>'
    assert parse_ruby(html) == [("漢字", "かんじ")]


def test_ruby_parse_complex_junk():
    # More complex junk tags
    html = "<ruby><div><b>漢</b></div><i>字</i><rt>かんじ</rt></ruby>"
    assert parse_ruby(html) == [("漢字", "かんじ")]
