import pytest
from lxml import etree
from runekana.document import XhtmlDocument


def normalize_html(html_str):
    """Helper to test normalization logic by calling the static method."""
    parser = etree.XMLParser(remove_blank_text=False)
    # Wrap in a root if not present to allow multiple tags
    wrapped = f"<root>{html_str}</root>"
    tree = etree.fromstring(wrapped, parser)

    # Call the static method directly
    XhtmlDocument._normalize_empty_tags(tree)

    # Return inner HTML
    res = etree.tostring(tree, encoding="unicode", method="xml")
    # Clean up the root wrapper
    return res.replace("<root>", "").replace("</root>", "")


@pytest.mark.parametrize(
    "input_html, expected_html",
    [
        # Void elements (should remain self-closing)
        ("<br/>", "<br/>"),
        ("<hr />", "<hr/>"),
        ('<img src="test.png" />', '<img src="test.png"/>'),
        (
            '<link rel="stylesheet" href="style.css"/>',
            '<link rel="stylesheet" href="style.css"/>',
        ),
        # Non-void elements (should become open/close pairs)
        ("<div/>", "<div></div>"),
        ("<p />", "<p></p>"),
        ('<span class="test"/>', '<span class="test"></span>'),
        ('<a href="https://example.com"/>', '<a href="https://example.com"></a>'),
        # Check spacing and attribute handling
        ('<img  src="test.png"  />', '<img src="test.png"/>'),
        ("<div   />", "<div></div>"),
        ('<p class="a" id="b" />', '<p class="a" id="b"></p>'),
        # Multiple tags in one string
        ("<br/><div/><hr />", "<br/><div></div><hr/>"),
        # Mixed case tags
        ("<BR/>", "<BR/>"),
        ("<DIV/>", "<DIV></DIV>"),
        ('<Img src="a.png"/>', '<Img src="a.png"/>'),
        # Namespaces
        (
            '<xhtml:div xmlns:xhtml="http://www.w3.org/1999/xhtml"/>',
            '<xhtml:div xmlns:xhtml="http://www.w3.org/1999/xhtml"></xhtml:div>',
        ),
        (
            '<xhtml:br xmlns:xhtml="http://www.w3.org/1999/xhtml"/>',
            '<xhtml:br xmlns:xhtml="http://www.w3.org/1999/xhtml"/>',
        ),
    ],
)
def test_normalization_logic(input_html, expected_html):
    assert normalize_html(input_html) == expected_html


@pytest.mark.parametrize(
    "content, expected_header",
    [
        (b'<?xml version="1.0"?><html/>', b'<?xml version="1.0"?>'),
        (b"<!DOCTYPE html><html>", b"<!DOCTYPE html>"),
        (
            b'<?xml version="1.0"?>\n<!DOCTYPE html>\n<html>',
            b'<?xml version="1.0"?>\n<!DOCTYPE html>\n',
        ),
        (b"<!-- comment --><html>", b"<!-- comment -->"),
        (
            b'<?xml version="1.0"?><!-- comment -->\n<html xmlns="..."/>',
            b'<?xml version="1.0"?><!-- comment -->\n',
        ),
        (b"<html/>", b""),
        (b"   \n<html/>", b"   \n"),
    ],
)
def test_extract_header(content, expected_header):
    assert XhtmlDocument._extract_header(content) == expected_header
