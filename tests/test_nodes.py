import pytest
import textwrap
from lxml import etree
from runekana.document.xhtml import Paragraph
from runekana.document.context import Context
from runekana.document.tokens import XHTML_NS


def make_paragraph(inner_html: str) -> Paragraph:
    """Wrap inner HTML in a <p> and parse into a Paragraph."""
    html = textwrap.dedent(
        f"""\
        <html xmlns="{XHTML_NS}">
        <body><p>{inner_html}</p></body>
        </html>
    """
    )
    root = etree.fromstring(html.encode("utf-8"))
    p_elem = root[0][0]  # body -> p
    return Paragraph(p_elem)


@pytest.mark.parametrize(
    "text, start_idx, end_idx, expected_clause, expected_start, expected_end",
    [
        # Basic cases
        ("こんにちは。世界。", 0, 5, "こんにちは。", 0, 5),  # "こんにちは"
        ("こんにちは。世界。", 6, 8, "世界。", 0, 2),  # "世界"
        # No punctuation
        ("こんにちは世界", 2, 4, "こんにちは世界", 2, 4),  # "にち"
        # Spanning punctuation
        ("第一。第二。第三。", 1, 4, "第一。第二。", 1, 4),  # "一。第" (spans "。")
        ("第一。第二。第三。", 0, 8, "第一。第二。第三。", 0, 8),  # spans two "。"
        # Edge cases - beginning and end
        ("あ。い。う", 0, 1, "あ。", 0, 1),
        ("あ。い。う", 4, 5, "う", 0, 1),
        # Japanese punctuation variety
        ("あ！い？う…え、お", 0, 1, "あ！", 0, 1),
        ("あ！い？う…え、お", 2, 3, "い？", 0, 1),
        ("あ！い？う…え、お", 4, 5, "う…", 0, 1),
        ("あ！い？う…え、お", 6, 7, "え、", 0, 1),
        ("あ！い？う…え、お", 8, 9, "お", 0, 1),
        # Brackets (includes trailing, excludes leading as per current implementation)
        ("『はじめ』「つぎ」", 1, 4, "はじめ』", 0, 3),  # "はじめ"
        ("『はじめ』「つぎ」", 0, 1, "『", 0, 1),  # "『"
        ("『はじめ』「つぎ」", 4, 5, "はじめ』", 3, 4),  # "』"
        ("『はじめ』「つぎ」", 6, 8, "つぎ」", 0, 2),  # "つぎ"
        # Multiple consecutive punctuation
        ("あ。。い", 0, 1, "あ。", 0, 1),
        ("あ。。い", 1, 2, "あ。", 1, 2),  # the first "。"
        ("あ。。い", 2, 3, "。", 0, 1),  # the second "。"
        ("あ。。い", 3, 4, "い", 0, 1),
        # Entire string is one clause
        ("あ", 0, 1, "あ", 0, 1),
        ("。", 0, 1, "。", 0, 1),
    ],
)
def test_get_nearest_clause(
    text, start_idx, end_idx, expected_clause, expected_start, expected_end
):
    # Mock a Context to test get_nearest_clause
    # We need a fake segment for Context
    p = make_paragraph(text)
    ctx = Context(
        start_idx=start_idx,
        end_idx=end_idx,
        segment=p[0],
        context=[(text, None)],
        size=len(text),
        offsets=[0],
    )

    clause_ctx = ctx.get_nearest_clause()
    assert str(clause_ctx) == expected_clause
    assert clause_ctx.start_idx == expected_start
    assert clause_ctx.end_idx == expected_end


def test_get_nearest_clause_empty_text():
    text = ""
    p = make_paragraph(text)
    # Paragraph(text) might have no segments if empty
    if not p.segments:
        # If no segments, we can't really make a Context for a segment.
        # But let's assume we want to test the logic.
        return

    ctx = Context(
        start_idx=0,
        end_idx=0,
        segment=p[0],
        context=[(text, None)],
        size=0,
        offsets=[0],
    )
    clause_ctx = ctx.get_nearest_clause()
    assert str(clause_ctx) == ""
    assert clause_ctx.start_idx == 0
    assert clause_ctx.end_idx == 0


def test_get_context_no_limits():
    # Case 1: No limits
    p = make_paragraph("前<ruby>中<rt>なか</rt></ruby>後")
    target = p[1]  # Ruby("中")
    ctx = p.get_context(target)

    assert str(ctx) == "前中後"
    assert ctx.start_idx == 1
    assert ctx.end_idx == 2
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "中"
    assert ctx.segment is target
    assert ctx.context == [("前", None), ("中", "なか"), ("後", None)]

    # Re-checking xhtml.py parse: it uses Ruby.parse which produces pairs.
    # Ruby("中") -> pairs=[("中", "なか")]
    # Paragraph.get_context for Ruby segment:
    # context_tuples.extend(seg.pairs)
    # So it should be [("前", None), ("中", "なか"), ("後", None)]
    assert ctx.context == [("前", None), ("中", "なか"), ("後", None)]


def test_get_context_textnode_truncation_backward():
    # Case 2: TextNode truncation at backward boundary
    p = make_paragraph("前のテキスト<ruby>標的<rt>ひようてき</rt></ruby>")
    target = p[1]
    # "前のテキスト" is 6 chars.
    # backward_max=3 should take "キスト"
    ctx = p.get_context(target, backward_max=3)

    assert str(ctx) == "キスト標的"
    assert ctx.start_idx == 3
    assert ctx.context[0] == ("キスト", None)
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "標的"


def test_get_context_textnode_truncation_forward():
    # Case 3: TextNode truncation at forward boundary
    p = make_paragraph("<ruby>標的<rt>ひようてき</rt></ruby>後のテキスト")
    target = p[0]
    # "後のテキスト" is 6 chars.
    # forward_max=3 should take "後のテ"
    ctx = p.get_context(target, forward_max=3)

    assert str(ctx) == "標的後のテ"
    assert ctx.start_idx == 0
    assert ctx.end_idx == 2
    assert ctx.context[-1] == ("後のテ", None)
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "標的"


def test_get_context_ruby_snap_backward():
    # Case 4: Ruby snap at backward boundary
    p = make_paragraph("<ruby>東京<rt>とうきょう</rt></ruby>のテキスト")
    target = p[1]  # "のテキスト"
    # "東京" is 2 chars.
    # backward_max=1 would cut into "東京", but it should snap to whole Ruby.
    ctx = p.get_context(target, backward_max=1)

    assert str(ctx) == "東京のテキスト"
    assert ctx.start_idx == 2
    assert ctx.context[0] == ("東京", "とうきょう")
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "のテキスト"


def test_get_context_ruby_snap_forward():
    # Case 5: Ruby snap at forward boundary
    p = make_paragraph("のテキスト<ruby>東京<rt>とうきょう</rt></ruby>")
    target = p[0]  # "のテキスト"
    # forward_max=1 would cut into "東京", but should snap.
    ctx = p.get_context(target, forward_max=1)

    assert str(ctx) == "のテキスト東京"
    assert ctx.start_idx == 0
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "のテキスト"


def test_get_context_mixed_segments():
    """Verify middle segments are included whole and boundary rules apply in a mixed paragraph."""
    p = make_paragraph("始<ruby>ル<rt>ruby</rt></ruby>中<ruby>ビ<rt>ruby</rt></ruby>終")
    target = p[2]  # "中"

    # Take whole thing
    ctx = p.get_context(target, backward_max=5, forward_max=5)
    assert str(ctx) == "始ル中ビ終"
    assert ctx.start_idx == 2
    assert ctx.end_idx == 3
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "中"

    # Truncate at boundaries: backward_max=1 snaps to include whole Ruby but excludes "始"
    ctx2 = p.get_context(target, backward_max=1)
    assert str(ctx2) == "ル中ビ終"
    assert ctx2.start_idx == 1
    assert str(ctx2)[ctx2.start_idx : ctx2.end_idx] == "中"


def test_get_context_target_first_last():
    # Case 7: Target is first
    p = make_paragraph("標的だけ")
    target = p[0]
    ctx = p.get_context(target, backward_max=10)
    assert str(ctx) == "標的だけ"
    assert ctx.start_idx == 0
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "標的だけ"

    # Case 8: Target is last
    # Note: make_paragraph wraps in <p>, so <span> becomes a child
    p2 = make_paragraph("だけ<span>標的</span>")
    target2 = p2[1]
    ctx2 = p2.get_context(target2, forward_max=10)
    assert str(ctx2) == "だけ標的"
    assert ctx2.end_idx == len(str(ctx2))
    assert str(ctx2)[ctx2.start_idx : ctx2.end_idx] == "標的"


def test_get_context_target_is_ruby():
    # Case 9
    p = make_paragraph("前<ruby>標的<rt>ターゲット</rt></ruby>後")
    target = p[1]
    ctx = p.get_context(target, backward_max=1, forward_max=1)
    assert str(ctx) == "前標的後"
    assert ctx.start_idx == 1
    assert ctx.end_idx == 3
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "標的"


def test_get_context_single_segment():
    # Case 10
    p = make_paragraph("単一")
    target = p[0]
    ctx = p.get_context(target, backward_max=10, forward_max=10)
    assert str(ctx) == "単一"
    assert ctx.start_idx == 0
    assert ctx.end_idx == 2
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "単一"


def test_get_context_value_error():
    # Case 11
    p1 = make_paragraph("一")
    p2 = make_paragraph("二")
    with pytest.raises(ValueError, match="not in Paragraph"):
        p1.get_context(p2[0])


def test_get_context_tuple_structure():
    # Case 12
    p = make_paragraph("前<ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby>後")
    target = p[2]  # "後"
    ctx = p.get_context(target)
    assert ctx.context == [("前", None), ("漢", "かん"), ("字", "じ"), ("後", None)]


def test_get_context_both_limits():
    """Verify both forward_max and backward_max apply correctly in a single call."""
    p = make_paragraph(
        "遠い前方のテキスト<ruby>標的<rt>ターゲット</rt></ruby>遠い後方のテキスト"
    )
    target = p[1]
    # "遠い前方のテキスト" is 9 chars. "遠い後方のテキスト" is 9 chars.
    # backward_max=4 -> "テキスト"
    # forward_max=4 -> "遠い後方"
    ctx = p.get_context(target, backward_max=4, forward_max=4)
    assert str(ctx) == "テキスト標的遠い後方"
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "標的"


def test_get_context_exact_boundary():
    """Verify behavior when limits align exactly with segment boundaries."""
    p = make_paragraph("前<ruby>中<rt>なか</rt></ruby>後")
    target = p[1]  # "中"

    # backward_max=0 should exactly hit the boundary of "前" and "中"
    ctx_b = p.get_context(target, backward_max=0)
    assert str(ctx_b) == "中後"
    assert ctx_b.start_idx == 0

    # forward_max=0 should exactly hit the boundary of "中" and "後"
    ctx_f = p.get_context(target, forward_max=0)
    assert str(ctx_f) == "前中"
    assert str(ctx_f)[ctx_f.start_idx : ctx_f.end_idx] == "中"


def test_get_context_multi_pair_ruby_preservation():
    """Verify that a multi-pair Ruby is included whole even if truncation boundary lands inside it."""
    p = make_paragraph("前<ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby>後")
    target = p[2]  # "後"

    # Ruby text is "漢字" (2 chars).
    # target "後" starts at index 3 (0:"前", 1:"漢", 2:"字").
    # backward_max=1 should land inside the Ruby (covers "字"), but must include whole "漢字".
    ctx = p.get_context(target, backward_max=1)
    assert str(ctx) == "漢字後"
    assert ctx.context[0:2] == [("漢", "かん"), ("字", "じ")]
    assert str(ctx)[ctx.start_idx : ctx.end_idx] == "後"
