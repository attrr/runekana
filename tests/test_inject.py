import pytest
from typing import Any
from lxml import etree
from unittest.mock import MagicMock
from runekana.document import Paragraph
from runekana.document import TextNode, Ruby
from runekana.document import TokenizedText, Yomi
from runekana.tokenizer import Token

XHTML_NS = "http://www.w3.org/1999/xhtml"


# Mock Morpheme for Token
class MockMorpheme:
    def __init__(
        self, surface, begin=0, end=None, dict_form=None, pos=None, reading=None
    ):
        self._surface = surface
        self._begin = begin
        self._end = end if end is not None else begin + len(surface)
        self._dict_form = dict_form or surface
        self._pos = pos or ["", "", "", "", "", ""]
        self._reading = reading or surface

    def surface(self):
        return self._surface

    def begin(self):
        return self._begin

    def end(self):
        return self._end

    def dictionary_form(self):
        return self._dict_form

    def part_of_speech(self):
        return self._pos

    def reading_form(self):
        return self._reading


# Mock Tokenizer
class MockTokenizer:
    def __init__(self, token_data=None):
        self.token_data = token_data or {}
        self.local_dict = {}

    def tokenize(self, text):
        if text in self.token_data:
            return self.token_data[text]
        # Default: return one token for the whole text
        m: Any = MockMorpheme(text)
        return [Token(morpheme=m)]


@pytest.fixture
def xhtml_p():
    def _make(inner_html):
        html = f'<p xmlns="{XHTML_NS}">{inner_html}</p>'
        return etree.fromstring(html.encode("utf-8"))

    return _make


def test_paragraph_parse_basic(xhtml_p):
    p_elem = xhtml_p("こんにちは<span>世界</span>")
    p = Paragraph(p_elem)

    assert len(p.segments) == 2
    assert isinstance(p.segments[0], TextNode)
    assert p.segments[0].text == "こんにちは"
    assert p.segments[1].text == "世界"


def test_paragraph_parse_with_ruby(xhtml_p):
    p_elem = xhtml_p("前<ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby>後")
    p = Paragraph(p_elem)

    assert len(p.segments) == 3
    assert isinstance(p.segments[0], TextNode)
    assert isinstance(p.segments[1], Ruby)
    assert isinstance(p.segments[2], TextNode)
    assert p.segments[1].text == "漢字"
    assert p.segments[1].pairs == [("漢", "かん"), ("字", "じ")]


def test_tokenized_text_inject_basic(xhtml_p):
    p_elem = xhtml_p("漢字")
    node = TextNode(p_elem, "text", paragraph=MagicMock())

    # Simulate tokenized result for "漢字" -> "かんじ"
    yomi = Yomi(base="漢字", reading="かんじ", begin=0, end=2)
    tokenized = TokenizedText(node, [yomi])

    tokenized.inject()

    # Verify DOM
    ruby = p_elem.find(f"{{{XHTML_NS}}}ruby")
    assert ruby is not None
    assert ruby.text == "漢字"
    assert ruby[0].text == "かんじ"
    assert p_elem.text is None


def test_tokenized_text_inject_skip_words(xhtml_p):
    p_elem = xhtml_p("私は漢字")
    node = TextNode(p_elem, "text", paragraph=MagicMock())

    # "私" is skipped (reading=None), "漢字" is annotated
    yomis = [
        Yomi(base="私", reading=None, begin=0, end=1),
        Yomi(base="は", reading=None, begin=1, end=2),
        Yomi(base="漢字", reading="かんじ", begin=2, end=4),
    ]
    tokenized = TokenizedText(node, yomis)

    tokenized.inject()

    # Verify DOM: "私は" should be plain text, followed by <ruby>
    assert p_elem.text == "私は"
    ruby = p_elem.find(f"{{{XHTML_NS}}}ruby")
    assert ruby is not None
    assert ruby.text == "漢字"


def test_tokenized_text_inject_mixed_existing_ruby(xhtml_p):
    # Testing that injecting into a TextNode next to a Ruby doesn't break things
    p_elem = xhtml_p("<ruby>既成<rt>きせい</rt></ruby>のテキスト")
    # The TextNode is the tail of the ruby
    ruby_elem = p_elem[0]
    node = TextNode(ruby_elem, "tail", paragraph=MagicMock())
    node.text = "のテキスト"

    yomis = [
        Yomi(base="の", reading=None, begin=0, end=1),
        Yomi(base="テキスト", reading=None, begin=1, end=5),
    ]
    tokenized = TokenizedText(node, yomis)
    tokenized.inject()

    # Verify DOM structure remains intact
    assert len(p_elem) == 1
    assert p_elem[0].tag == f"{{{XHTML_NS}}}ruby"
    assert p_elem[0].tail == "のテキスト"


def test_text_node_tokenize_with_context():
    # Integrated test for TextNode.tokenize using MockTokenizer
    p_elem = etree.fromstring(f'<p xmlns="{XHTML_NS}">麓町</p>')
    p = Paragraph(p_elem)
    node = p.segments[0]
    assert isinstance(node, TextNode)

    # Mock tokenizer to return one token for "麓町"
    m: Any = MockMorpheme("麓町", reading="ふもとまち")
    tok = MagicMock()
    tok.tokenize.return_value = [Token(morpheme=m, reading="ふもとまち")]

    tokenized = node.tokenize(tok)

    assert len(tokenized.annotations) == 1
    assert tokenized.annotations[0].base == "麓町"
    assert tokenized.annotations[0].reading == "ふもとまち"


def test_tokenized_text_inject_with_okurigana(xhtml_p):
    p_elem = xhtml_p("食べる")
    node = TextNode(p_elem, "text", paragraph=MagicMock())

    # "食べる" -> "食（た）べる"
    yomi = Yomi(base="食べる", reading="たべる", begin=0, end=3)
    tokenized = TokenizedText(node, [yomi])

    tokenized.inject()

    ruby = p_elem.find(f"{{{XHTML_NS}}}ruby")
    assert ruby is not None
    assert ruby.text == "食"
    assert ruby[0].text == "た"
    assert ruby.tail == "べる"
