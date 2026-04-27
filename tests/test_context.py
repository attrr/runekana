import pytest
import os
import tempfile
from runekana.document import XhtmlDocument
from runekana.inject import DomTraverser
from runekana.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer(skip_words=set(), local_dict={})


def create_temp_xhtml(content, namespaces=None):
    """
    Helper to create a temporary XHTML file.
    Default namespace is http://www.w3.org/1999/xhtml.
    """
    fd, path = tempfile.mkstemp(suffix=".xhtml")
    header = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'

    if namespaces is None:
        root_tag = '<html xmlns="http://www.w3.org/1999/xhtml">'
    else:
        ns_attrs = " ".join(
            [
                f'xmlns:{prefix}="{uri}"' if prefix else f'xmlns="{uri}"'
                for prefix, uri in namespaces.items()
            ]
        )
        root_tag = f"<html {ns_attrs}>"

    xhtml = f"""{header}
{root_tag}
<head><title>Test</title></head>
<body>
{content}
</body>
</html>"""
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(xhtml)
    return path


@pytest.fixture
def temp_xhtml():
    paths = []

    def _create(content, namespaces=None):
        path = create_temp_xhtml(content, namespaces)
        paths.append(path)
        return path

    yield _create
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# --- XhtmlDocument.get_block_text Tests ---


def test_get_block_text_nested_blocks(temp_xhtml):
    """
    Verify the "closest" principle: in <div><p>target</p></div>,
    selecting target should return <p>'s content, not <div>'s.
    """
    content = """
    <div id="outer">
        <p id="inner">これは<span>ターゲット</span>テキストです</p>
        <p>無関係なテキスト</p>
    </div>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)
    span = doc.tree.xpath("//*[local-name()='span' and text()='ターゲット']")[0]
    result = doc.get_block_text(span)
    assert result == "これはターゲットテキストです"
    assert "無関係なテキスト" not in result


def test_get_block_text_whitespace_normalization(temp_xhtml):
    """
    Verify that newlines, tabs, and multiple spaces are normalized to a single space.
    """
    content = """
    <p>
        一行目
        二行目\tタブあり    と複数のスペース
    </p>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)
    p = doc.tree.xpath("//*[local-name()='p']")[0]
    result = doc.get_block_text(p)
    assert result == "一行目 二行目 タブあり と複数のスペース"


def test_get_block_text_various_block_tags(temp_xhtml):
    """
    Test extraction across different block-level tags: li, h1, body.
    """
    content = """
    <ul>
        <li>リスト項目 1</li>
    </ul>
    <h1>タイトル内容</h1>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)

    li = doc.tree.xpath("//*[local-name()='li']")[0]
    assert doc.get_block_text(li) == "リスト項目 1"

    h1 = doc.tree.xpath("//*[local-name()='h1']")[0]
    assert doc.get_block_text(h1) == "タイトル内容"

    body = doc.tree.xpath("//*[local-name()='body']")[0]
    result = doc.get_block_text(body)
    assert "リスト項目 1" in result
    assert "タイトル内容" in result


def test_get_block_text_deep_nesting(temp_xhtml):
    """
    Verify that the method can traverse up multiple levels of inline tags.
    """
    content = """
    <p>
        <span><i><b><ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby></b></i></span>
    </p>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)
    rt = doc.tree.xpath("//*[local-name()='rt']")[0]
    result = doc.get_block_text(rt)
    # itertext() joins text nodes in document order
    assert result == "漢かん字じ"


def test_get_block_text_edge_positions(temp_xhtml):
    """
    Test when the target element itself is a block or a direct child of html/body.
    """
    content = """
    <p id="target">私はブロック要素です</p>
    <span>私はbodyの直接の子です</span>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)

    p = doc.tree.xpath("//*[@id='target']")[0]
    assert doc.get_block_text(p) == "私はブロック要素です"

    span = doc.tree.xpath(
        "//*[local-name()='span' and text()='私はbodyの直接の子です']"
    )[0]
    result = doc.get_block_text(span)
    assert "私はブロック要素です" in result
    assert "私はbodyの直接の子です" in result


def test_get_block_text_namespace_robustness(temp_xhtml):
    """
    Verify that the method correctly handles namespaced tags using QName.
    """
    fd, path = tempfile.mkstemp(suffix=".xhtml")
    xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html:html xmlns:html="http://www.w3.org/1999/xhtml">
<html:body>
<html:p>名前空間テスト</html:p>
</html:body>
</html:html>"""
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(xhtml)

    try:
        doc = XhtmlDocument(path)
        p = doc.tree.xpath("//*[local-name()='p']")[0]
        assert doc.get_block_text(p) == "名前空間テスト"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_get_block_text_empty_or_whitespace(temp_xhtml):
    """
    Test behavior with empty blocks, or blocks containing only comments/whitespace.
    """
    content = """
    <p><!-- コメント --></p>
    <div>   </div>
    <p><?php echo "PI"; ?></p>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)

    p1 = doc.tree.xpath("//*[local-name()='p']")[0]
    assert doc.get_block_text(p1) == ""

    div = doc.tree.xpath("//*[local-name()='div']")[0]
    assert doc.get_block_text(div) == ""

    p3 = doc.tree.xpath("//*[local-name()='p']")[1]
    assert doc.get_block_text(p3) == ""


# --- DomTraverser Integration Tests ---


def test_full_integration_with_messy_html(tokenizer, temp_xhtml):
    # Use compact HTML to avoid regex whitespace merging
    prefix = "あ" * 50
    suffix = "い" * 50
    content = f"<p>{prefix[:20]}<span>{prefix[20:40]}<i>{prefix[40:]}</i></span><b>身体</b><span>{suffix[:20]}</span>{suffix[20:]}</p>"
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)
    traverser = DomTraverser(tokenizer)
    tasks, candidates = traverser.traverse(doc)

    cand = next(c for c in candidates if c["word"] == "身体")
    ctx = cand["context"]

    expected_before = "あ" * 30
    expected_after = "い" * 30
    assert ctx == expected_before + "身体" + expected_after
    assert len(ctx) == 62


def test_context_across_blocks_fails_by_design(tokenizer, temp_xhtml):
    content = """
    <p>短い開始。</p>
    <p><b>身体</b> はここにあります。</p>
    <p>短い終了。</p>
    """
    path = temp_xhtml(content)
    doc = XhtmlDocument(path)
    traverser = DomTraverser(tokenizer)
    _, candidates = traverser.traverse(doc)

    cand = next(c for c in candidates if c["word"] == "身体")
    # Should only get context from the second <p>
    assert cand["context"] == "身体 はここにあります。"
