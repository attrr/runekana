import pytest
from lxml import etree
from unittest.mock import MagicMock
from runekana.inject import DomTraverser, InjectionTask, XHTML_NS
from runekana.tokenizer import Token, Tokenizer


# Mock Tokenizer to avoid dependency on Sudachi
class MockTokenizer(Tokenizer):
    def __init__(self, token_map):
        self.token_map = token_map
        self.local_dict = {}

    def tokenize(self, text):
        return self.token_map.get(text, [Token(surface=text)])


@pytest.fixture
def xhtml_doc_mock():
    doc = MagicMock()
    doc.root = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><body><p>汉字</p></body></html>'
    )
    doc.get_block_text.return_value = "汉字"
    return doc


def test_dom_traverser_basic():
    # Setup: "汉字" should be tokenized into one token with reading
    token = Token(surface="汉字", reading="かんじ")
    tokenizer = MockTokenizer({"汉字": [token]})
    traverser = DomTraverser(tokenizer)

    doc = MagicMock()
    doc.root = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><body><p>汉字</p></body></html>'
    )
    doc.get_block_text.return_value = "汉字"

    tasks, candidates = traverser.traverse(doc)

    assert len(tasks) == 1
    assert tasks[0].tokens[0].surface == "汉字"
    assert tasks[0].attr == "text"


def test_dom_traverser_skip_ruby():
    # Setup: "汉字" inside a ruby tag should be skipped
    tokenizer = MockTokenizer({})
    traverser = DomTraverser(tokenizer)

    doc = MagicMock()
    doc.root = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><body><p><ruby>汉字<rt>かんじ</rt></ruby></p></body></html>'
    )

    tasks, candidates = traverser.traverse(doc)
    assert len(tasks) == 0


def test_dom_traverser_tail_text():
    # Setup: text after an element (tail)
    token = Token(surface="汉字", reading="かんじ")
    tokenizer = MockTokenizer({"汉字": [token]})
    traverser = DomTraverser(tokenizer)

    doc = MagicMock()
    doc.root = etree.fromstring(
        f'<html xmlns="{XHTML_NS}"><body><p><span>prefix</span>汉字</p></body></html>'
    )
    doc.get_block_text.return_value = "prefix汉字"

    tasks, candidates = traverser.traverse(doc)

    # The task should be on the span's tail
    assert len(tasks) == 1
    assert tasks[0].attr == "tail"
    tag = tasks[0].elem.tag
    # Type narrowing for LSP
    assert isinstance(tag, (str, bytes))
    assert etree.QName(tag).localname == "span"


def test_injection_task_apply_text():
    # Setup a simple p tag with "汉字"
    root = etree.fromstring(f'<p xmlns="{XHTML_NS}">汉字</p>')
    token = Token(surface="汉字", reading="かんじ")
    task = InjectionTask(elem=root, attr="text", tokens=[token])

    task.apply()

    # Check if ruby is injected
    ruby = root.find(f".//{{{XHTML_NS}}}ruby")
    assert ruby is not None
    assert ruby.text == "汉字"
    assert len(ruby) > 0
    rt = ruby[0]
    assert rt.text == "かんじ"
    assert root.text is None


def test_injection_task_apply_mixed():
    # Setup: "A汉字B" -> "A<ruby>汉字<rt>かんじ</rt></ruby>B"
    root = etree.fromstring(f'<p xmlns="{XHTML_NS}">A汉字B</p>')
    tokens = [
        Token(surface="A"),
        Token(surface="汉字", reading="かんじ"),
        Token(surface="B"),
    ]
    task = InjectionTask(elem=root, attr="text", tokens=tokens)

    task.apply()

    assert root.text == "A"
    ruby = root.find(f"{{{XHTML_NS}}}ruby")
    assert ruby is not None
    assert ruby.text == "汉字"
    assert ruby.tail == "B"


def test_injection_task_with_okurigana():
    # Setup: "食べる" -> "食（た）べる"
    # split_okurigana is called internally
    root = etree.fromstring(f'<p xmlns="{XHTML_NS}">食べる</p>')
    token = Token(surface="食べる", reading="たべる")
    task = InjectionTask(elem=root, attr="text", tokens=[token])

    task.apply()

    ruby = root.find(f"{{{XHTML_NS}}}ruby")
    assert ruby is not None
    assert ruby.text == "食"
    assert len(ruby) > 0
    rt = ruby[0]
    assert rt.text == "た"
    assert ruby.tail == "べる"
