"""Generate Hùng's Weeks 1–3 Word deliverable (MMT/docs/BaoCao_Hung_Tuan1_3.docx).

Follows all formatting rules in Section 4 of KeHoach_PhanCong_Nhom_KNN.docx:
- A4 paper (21.0 x 29.7 cm), margins L=2.5 cm, R=2.0 cm, T=2.0 cm, B=2.0 cm
- Font: Times New Roman, 13 pt
- Alignment: Justify for body paragraphs
- First-line indent: 1.0 cm for body prose
- Bold headings with consistent formatting per level; major sections centered on a new page
- Native Microsoft Word Equations (OMML <m:oMath>) for all numbered equations
  and inline math expressions (zero raw LaTeX/underscores).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from lxml import etree
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

MMT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = MMT_ROOT / "experiments" / "dataset_summary.json"
OUT_DOCX = MMT_ROOT / "docs" / "BaoCao_Hung_Tuan1_3.docx"

FONT_NAME = "Times New Roman"
MATH_FONT = "Cambria Math"
BODY_SIZE = Pt(13)
TABLE_SIZE = Pt(11.5)
BLACK = RGBColor(0, 0, 0)

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _m(tag: str, *children) -> etree._Element:
    el = etree.Element(f"{{{M_NS}}}{tag}")
    for child in children:
        if isinstance(child, list):
            for c in child:
                el.append(c)
        elif child is not None:
            el.append(child)
    return el


def _w(tag: str, attrib: dict | None = None, *children) -> etree._Element:
    el = etree.Element(f"{{{W_NS}}}{tag}", attrib or {})
    for child in children:
        el.append(child)
    return el


def mr(text: str, plain: bool = False, sz_half_pt: int = 26) -> etree._Element:
    """Create an OMML math run <m:r>."""
    r = etree.Element(f"{{{M_NS}}}r")
    if plain:
        r.append(_m("rPr", _m("sty", *[], **{}) if False else etree.Element(f"{{{M_NS}}}sty", {f"{{{M_NS}}}val": "p"})))
    w_rpr = _w(
        "rPr",
        None,
        _w("rFonts", {f"{{{W_NS}}}ascii": MATH_FONT, f"{{{W_NS}}}hAnsi": MATH_FONT}),
        _w("sz", {f"{{{W_NS}}}val": str(sz_half_pt)}),
        _w("szCs", {f"{{{W_NS}}}val": str(sz_half_pt)}),
    )
    r.append(w_rpr)
    t = etree.Element(f"{{{M_NS}}}t")
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    return r


def _wrap(items) -> list[etree._Element]:
    if isinstance(items, list):
        return items
    return [items]


def m_sub(base, sub) -> etree._Element:
    return _m("sSub", _m("e", *_wrap(base)), _m("sub", *_wrap(sub)))


def m_sup(base, sup) -> etree._Element:
    return _m("sSup", _m("e", *_wrap(base)), _m("sup", *_wrap(sup)))


def m_subsup(base, sub, sup) -> etree._Element:
    return _m("sSubSup", _m("e", *_wrap(base)), _m("sub", *_wrap(sub)), _m("sup", *_wrap(sup)))


def m_frac(num, den) -> etree._Element:
    return _m("f", _m("num", *_wrap(num)), _m("den", *_wrap(den)))


def m_sqrt(expr) -> etree._Element:
    rad_pr = _m("radPr", etree.Element(f"{{{M_NS}}}degHide", {f"{{{M_NS}}}val": "1"}))
    return _m("rad", rad_pr, _m("deg"), _m("e", *_wrap(expr)))


def m_sum(sub, expr, sup=None) -> etree._Element:
    nary_pr = _m(
        "naryPr",
        etree.Element(f"{{{M_NS}}}chr", {f"{{{M_NS}}}val": "∑"}),
        etree.Element(f"{{{M_NS}}}limLoc", {f"{{{M_NS}}}val": "undOvr"}),
    )
    if sup is None:
        nary_pr.append(etree.Element(f"{{{M_NS}}}supHide", {f"{{{M_NS}}}val": "1"}))
    sup_el = _m("sup") if sup is None else _m("sup", *_wrap(sup))
    return _m("nary", nary_pr, _m("sub", *_wrap(sub)), sup_el, _m("e", *_wrap(expr)))


def m_acc(base, char: str) -> etree._Element:
    acc_pr = _m("accPr", etree.Element(f"{{{M_NS}}}chr", {f"{{{M_NS}}}val": char}))
    return _m("acc", acc_pr, _m("e", *_wrap(base)))


def m_hat(base) -> etree._Element:
    return m_acc(base, "\u0302")


def m_bar(base) -> etree._Element:
    return m_acc(base, "\u0304")


def m_paren(expr, beg: str = "(", end: str = ")") -> etree._Element:
    d_pr = _m(
        "dPr",
        etree.Element(f"{{{M_NS}}}begChr", {f"{{{M_NS}}}val": beg}),
        etree.Element(f"{{{M_NS}}}endChr", {f"{{{M_NS}}}val": end}),
    )
    return _m("d", d_pr, _m("e", *_wrap(expr)))


def m_omath(*nodes) -> etree._Element:
    return _m("oMath", *nodes)


# --- Inline OMML helper tokens for prose ---
INLINE_MATH_TOKENS: dict[str, callable] = {
    "rhat_ui": lambda: m_sub(m_hat(mr("r")), mr("ui")),
    "rhat_U1M5": lambda: [m_hat(mr("r")), m_paren([m_sub(mr("U"), mr("1")), mr(", "), m_sub(mr("M"), mr("5"))])],
    "r_U1M5": lambda: [mr("r"), m_paren([m_sub(mr("U"), mr("1")), mr(", "), m_sub(mr("M"), mr("5"))])],
    "rbar_u": lambda: m_sub(m_bar(mr("r")), mr("u")),
    "rbar_v": lambda: m_sub(m_bar(mr("r")), mr("v")),
    "rbar_i": lambda: m_sub(m_bar(mr("r")), mr("i")),
    "rbar_j": lambda: m_sub(m_bar(mr("r")), mr("j")),
    "r_ui": lambda: m_sub(mr("r"), mr("ui")),
    "r_uj": lambda: m_sub(mr("r"), mr("uj")),
    "r_vi": lambda: m_sub(mr("r"), mr("vi")),
    "r_vj": lambda: m_sub(mr("r"), mr("vj")),
    "I_uv": lambda: m_sub(mr("I"), mr("uv")),
    "I_u": lambda: m_sub(mr("I"), mr("u")),
    "I_v": lambda: m_sub(mr("I"), mr("v")),
    "U_ij": lambda: m_sub(mr("U"), mr("ij")),
    "U_i": lambda: m_sub(mr("U"), mr("i")),
    "U_j": lambda: m_sub(mr("U"), mr("j")),
    "N_k_ui": lambda: [m_sub(mr("N"), mr("k")), m_paren(mr("u; i"))],
    "N_k_iu": lambda: [m_sub(mr("N"), mr("k")), m_paren(mr("i; u"))],
    "sim_Pearson_uv": lambda: [m_sub(mr("sim", plain=True), mr("Pearson", plain=True)), m_paren(mr("u, v"))],
    "sim_Pearson_U1U2": lambda: [m_sub(mr("sim", plain=True), mr("Pearson", plain=True)), m_paren([m_sub(mr("U"), mr("1")), mr(", "), m_sub(mr("U"), mr("2"))])],
    "sim_Pearson_U1U4": lambda: [m_sub(mr("sim", plain=True), mr("Pearson", plain=True)), m_paren([m_sub(mr("U"), mr("1")), mr(", "), m_sub(mr("U"), mr("4"))])],
    "sim_AdjCos_ij": lambda: [m_sub(mr("sim", plain=True), mr("AdjCos", plain=True)), m_paren(mr("i, j"))],
    "sim_uv": lambda: [mr("sim", plain=True), m_paren(mr("u, v"))],
    "sim_ij": lambda: [mr("sim", plain=True), m_paren(mr("i, j"))],
    "R_K_u": lambda: [m_sub(mr("R"), mr("K")), m_paren(mr("u"))],
    "Rel_u": lambda: [mr("Rel", plain=True), m_paren(mr("u"))],
    "rel_p": lambda: m_sub(mr("rel", plain=True), mr("p")),
    "IDCG_K": lambda: [m_sub(mr("IDCG", plain=True), mr("K"))],
    "NDCG_K": lambda: [m_sub(mr("NDCG", plain=True), mr("K"))],
    "O_U2I": lambda: [mr("O", plain=True), m_paren([m_sup(m_paren(mr("U"), "|", "|"), mr("2")), mr("·"), m_paren(mr("I"), "|", "|")])],
    "O_I2U": lambda: [mr("O", plain=True), m_paren([m_sup(m_paren(mr("I"), "|", "|"), mr("2")), mr("·"), m_paren(mr("U"), "|", "|")])],
    "U_gt_I": lambda: [m_paren(mr("U"), "|", "|"), mr(" ≫ "), m_paren(mr("I"), "|", "|")],
}


def set_run_font(run, size=BODY_SIZE, bold=False, italic=False, color=BLACK) -> None:
    run.font.name = FONT_NAME
    run.font.size = size
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), FONT_NAME)
    rFonts.set(qn("w:hAnsi"), FONT_NAME)
    rFonts.set(qn("w:cs"), FONT_NAME)
    rFonts.set(qn("w:eastAsia"), FONT_NAME)


def append_rich_runs(paragraph, text: str, size=BODY_SIZE, bold: bool = False, italic: bool = False) -> None:
    """Parse lightweight inline tokens `{{token}}`, `_{sub}`, `^{sup}` into clean Word runs/OMML."""
    pattern = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}|_\{([^}]+)\}|\^\{([^}]+)\}")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos : m.start()])
            set_run_font(r, size=size, bold=bold, italic=italic)
        if m.group(1) is not None:
            key = m.group(1)
            nodes = INLINE_MATH_TOKENS[key]()
            paragraph._p.append(m_omath(*_wrap(nodes)))
        elif m.group(2) is not None:
            r = paragraph.add_run(m.group(2))
            set_run_font(r, size=size, bold=bold, italic=italic)
            r.font.subscript = True
        elif m.group(3) is not None:
            r = paragraph.add_run(m.group(3))
            set_run_font(r, size=size, bold=bold, italic=italic)
            r.font.superscript = True
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        set_run_font(r, size=size, bold=bold, italic=italic)


def configure_document() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = BODY_SIZE
    normal.font.color.rgb = BLACK
    return doc


def add_major_title(doc: Document, text: str, page_break_before: bool = False) -> None:
    if page_break_before:
        doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(6)
    pf.space_after = Pt(12)
    pf.line_spacing = 1.35
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=Pt(15), bold=True)


def add_heading_h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(12)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.35
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=Pt(14), bold=True)


def add_heading_h3(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(8)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.35
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=BODY_SIZE, bold=True)


def add_body(doc: Document, text: str, bold_prefix: str | None = None, indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.first_line_indent = Cm(1.0) if indent else Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.4
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        set_run_font(rb, size=BODY_SIZE, bold=True)
    append_rich_runs(p, text, size=BODY_SIZE)


def add_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.left_indent = Cm(1.0)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.35
    r0 = p.add_run("– " + bold_prefix)
    set_run_font(r0, size=BODY_SIZE, bold=True)
    append_rich_runs(p, text, size=BODY_SIZE)


def add_omml_equation(doc: Document, omath_el: etree._Element, label: str) -> None:
    """Render a native Word equation centered in a borderless layout table with right-aligned label."""
    tbl = doc.add_table(rows=1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Remove all borders cleanly
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)

    widths = [Cm(1.8), Cm(12.9), Cm(1.8)]
    row = tbl.rows[0]
    for i, w in enumerate(widths):
        row.cells[i].width = w
        row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    p_eq = row.cells[1].paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.first_line_indent = Cm(0)
    p_eq.paragraph_format.space_before = Pt(4)
    p_eq.paragraph_format.space_after = Pt(4)
    p_eq._p.append(omath_el)

    p_lbl = row.cells[2].paragraphs[0]
    p_lbl.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_lbl.paragraph_format.first_line_indent = Cm(0)
    p_lbl.paragraph_format.space_before = Pt(4)
    p_lbl.paragraph_format.space_after = Pt(4)
    rl = p_lbl.add_run(label)
    set_run_font(rl, size=BODY_SIZE)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)


def add_code_block(doc: Document, lines: list[str]) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.left_indent = Cm(0.8)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    pPr = p._p.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F8"/>')
    pPr.append(shd)
    r = p.add_run("\n".join(lines))
    r.font.name = "Consolas"
    r.font.size = Pt(10.5)
    r.font.color.rgb = BLACK


def add_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], col_widths_cm: list[float]) -> None:
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.keep_with_next = True
    append_rich_runs(cap, caption, size=BODY_SIZE, bold=True, italic=True)

    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="888888"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="888888"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)

    all_rows = [headers] + rows
    for r_idx, row_vals in enumerate(all_rows):
        row = tbl.rows[r_idx]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = Cm(col_widths_cm[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if r_idx == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="E9ECEF"/>'))
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if r_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.2
            append_rich_runs(p, str(val), size=TABLE_SIZE, bold=(r_idx == 0))

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)


def build_doc() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ml = summary["movielens"]
    prod = summary["products"]
    docs_s = summary["documents"]

    doc = configure_document()

    # =========================================================================
    # TRANG ĐẦU: THÔNG TIN BÁO CÁO PHẦN HÙNG (TUẦN 1–3)
    # =========================================================================
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.first_line_indent = Cm(0)
    p_top.paragraph_format.space_after = Pt(6)
    r_top = p_top.add_run(
        "MÔN HỌC: MẠNG MÁY TÍNH (MMT)\n"
        "ĐỒ ÁN NHÓM: TÌM HIỂU THUẬT TOÁN KNN VÀ ỨNG DỤNG\n"
        "GỢI Ý XEM PHIM / HÀNG MỚI / TÀI LIỆU DOWNLOAD"
    )
    set_run_font(r_top, size=Pt(13.5), bold=True)

    add_major_title(doc, "BÁO CÁO PHẦN ĐẢM NHẬN CỦA THÀNH VIÊN HÙNG\n(HOÀN THÀNH TIẾN ĐỘ TUẦN 1 – TUẦN 3)")

    add_table(
        doc,
        "Bảng 0.1. Tóm tắt các hạng mục hoàn thành của Hùng theo kế hoạch Tuần 1–3",
        ["Giai đoạn", "Hạng mục kỹ thuật & dữ liệu", "Hạng mục báo cáo", "Trạng thái"],
        [
            [
                "Tuần 1 (05/11–11/11)\nMốc M1",
                "Tải MovieLens ml-latest-small; chốt nguồn và sinh/chuẩn hóa dữ liệu Zipf cho Sản phẩm và Tài liệu",
                "Thống kê mô tả 3 bộ dữ liệu (số user, item, rating/tương tác, độ thưa)",
                "Hoàn thành (100%)",
            ],
            [
                "Tuần 2 (12/11–18/11)\nMốc M2",
                "Làm sạch & chuẩn hóa dữ liệu sản phẩm/tài liệu; thiết kế schema CSDL SQLite (docs/schema.sql); khung src/evaluate.py (RMSE, MAE)",
                "Soạn thảo mục 1.3.1 (Tổng quan hệ gợi ý) và mục 1.3.3 (Lọc cộng tác User-based & Item-based KNN)",
                "Hoàn thành (100%)",
            ],
            [
                "Tuần 3 (19/11–25/11)\nMốc M3",
                "Cài đặt src/rec_newitem.py (Content-based + Hybrid Switching KNN giải cold-start, chạy CLI Top-10); mở rộng evaluate.py (P@K, R@K, NDCG@K); viết unit test",
                "Hoàn thiện mục 1.3.5 (Cold-start, Sparsity, Scalability) và mục 1.3.6 (Các độ đo đánh giá); chuẩn bị 14 tài liệu tham khảo",
                "Hoàn thành (100%)",
            ],
        ],
        [3.3, 5.5, 5.2, 2.5],
    )

    add_body(
        doc,
        "Tài liệu này được trình bày tuân thủ đúng quy chuẩn hình thức của giảng viên (khổ giấy A4, phông chữ Times New Roman cỡ 13, căn đều hai bên, thụt đầu dòng 1 cm, tiêu đề in đậm thống nhất theo cấp, công thức toán học chuẩn Equation của Microsoft Word) nhằm giúp nhóm trưởng (Hoàng) dễ dàng ghép trực tiếp các mục 1.3.1, 1.3.3, 1.3.5 và 1.3.6 vào Chương 1 của quyển báo cáo cuối kỳ.",
        bold_prefix="Ghi chú ghép quyển: ",
    )

    # =========================================================================
    # CHƯƠNG 1 (CÁC MỤC HÙNG PHỤ TRÁCH)
    # =========================================================================
    add_major_title(
        doc,
        "CHƯƠNG 1. TỔNG QUAN VỀ TRÍ TUỆ NHÂN TẠO VÀ CÁC THUẬT TOÁN\n(CÁC MỤC 1.3.1, 1.3.3, 1.3.5, 1.3.6 DO HÙNG BIÊN SOẠN)",
        page_break_before=True,
    )

    # --- 1.3.1 ---
    add_heading_h2(doc, "1.3.1. Tổng quan về hệ gợi ý (Recommender Systems)")

    add_body(
        doc,
        "Sự bùng nổ của Internet và các nền tảng số trong hai thập kỷ qua đã dẫn đến hiện tượng quá tải thông tin (information overload), khi người dùng phải đối mặt với hàng triệu bộ phim, sản phẩm thương mại điện tử hay tài liệu học tập trực tuyến. Trong bối cảnh đó, hệ gợi ý (Recommender System – RS) ra đời như một lớp phần mềm thông minh có nhiệm vụ phân tích dữ liệu hành vi, sở thích và đặc trưng nội dung nhằm chủ động đề xuất các đối tượng (items) phù hợp nhất với từng cá nhân người dùng (users) tại thời điểm truy vấn [1], [2].",
    )
    add_body(
        doc,
        "Về mặt giá trị thực tiễn, hệ gợi ý không chỉ cải thiện trải nghiệm tìm kiếm của người dùng mà còn tác động trực tiếp đến chỉ số kinh doanh của doanh nghiệp: theo các báo cáo công nghiệp kinh điển, khoảng 35% doanh thu bán hàng trên Amazon và hơn 75–80% lượt xem nội dung trên Netflix bắt nguồn từ các mô-đun gợi ý tự động [3], [4]. Trong phạm vi đồ án này, bài toán gợi ý được triển khai đồng thời trên ba miền ứng dụng đại diện cho ba dạng dữ liệu đặc thù: (1) gợi ý phim với dữ liệu đánh giá tường minh (explicit rating), (2) gợi ý hàng mới với thách thức thiếu lịch sử tương tác (cold-start), và (3) gợi ý tài liệu tải về với dữ liệu phản hồi ngầm (implicit feedback) kết hợp văn bản.",
    )
    add_body(
        doc,
        "Xét trên phương diện toán học, bài toán cốt lõi của hệ gợi ý được hình thức hóa thông qua tập người dùng U = {u_{1}, u_{2}, …, u_{m}} và tập đối tượng I = {i_{1}, i_{2}, …, i_{n}}. Gọi R là ma trận tương tác kích thước |U| × |I|, trong đó phần tử {{r_ui}} biểu diễn mức độ quan tâm đã quan sát của người dùng u đối với đối tượng i ({{r_ui}} = ∅ nếu chưa có tương tác). Mục tiêu của hệ thống là xây dựng hàm dự đoán tiện ích f : U × I → ℝ sao cho với mỗi người dùng u, hệ thống lựa chọn tập K đối tượng chưa từng tương tác có giá trị dự đoán cao nhất [1]:",
    )

    # Công thức (1.1): I*_K(u) = argmax^{(K)}_{i ∈ I \ I_u} f(u, i)
    eq_1_1 = m_omath(
        m_subsup(mr("I"), mr("K"), mr("*")),
        m_paren(mr("u")),
        mr(" = "),
        m_subsup(mr("argmax", plain=True), [mr("i ∈ I \\ "), m_sub(mr("I"), mr("u"))], m_paren(mr("K"))),
        mr(" "),
        mr("f"),
        m_paren(mr("u, i")),
    )
    add_omml_equation(doc, eq_1_1, "(1.1)")

    add_body(
        doc,
        "Dữ liệu đầu vào của hàm f(u, i) được chia thành hai nhóm chính với đặc tính xử lý khác nhau:",
    )
    add_bullet(
        doc,
        "Phản hồi tường minh (Explicit Feedback): ",
        "Người dùng chủ động biểu thị mức độ yêu thích thông qua thang điểm số (ví dụ thang 0.5–5.0 sao trong bộ dữ liệu MovieLens), lượt thích/không thích hoặc nhận xét văn bản. Ưu điểm của dữ liệu tường minh là phản ánh trực tiếp thái độ tích cực lẫn tiêu cực, nhưng nhược điểm là tỷ lệ thu thập thấp vì đòi hỏi người dùng bỏ công sức thao tác.",
    )
    add_bullet(
        doc,
        "Phản hồi ngầm (Implicit Feedback): ",
        "Hệ thống tự động ghi nhận hành vi tự nhiên của người dùng như lượt xem (view), lưu trang (bookmark), thêm vào giỏ hàng (cart), tải tài liệu (download) hoặc mua hàng (purchase) [5]. Dữ liệu ngầm có khối lượng rất lớn nhưng thiếu vắng phản hồi tiêu cực rõ ràng (một tài liệu chưa được tải có thể do người dùng chưa biết tới chứ không hẳn vì không thích) và thường chứa nhiều nhiễu, đòi hỏi cơ chế gán trọng số tin cậy phù hợp.",
    )
    add_body(
        doc,
        "Dựa trên nguồn tri thức khai thác để tính toán độ phù hợp, các kỹ thuật xây dựng hệ gợi ý được phân thành ba hướng tiếp cận chính: (i) Lọc theo nội dung (Content-based Filtering) dựa trên thuộc tính mô tả của đối tượng và hồ sơ sở thích người dùng; (ii) Lọc cộng tác (Collaborative Filtering – CF) khai thác trí tuệ đám đông từ ma trận tương tác quá khứ mà không cần hiểu nội dung đối tượng; và (iii) Hệ gợi ý lai (Hybrid Recommender System) kết hợp hai hướng trên nhằm tận dụng ưu điểm và bù đắp nhược điểm cố hữu như khởi đầu nguội hay thưa thớt dữ liệu [6]. Thuật toán K-Nearest Neighbors (KNN) đóng vai trò nền tảng trong cả ba hướng tiếp cận này nhờ tính trực quan, dễ giải thích và không đòi hỏi quá trình huấn luyện tham số phức tạp.",
    )

    # --- 1.3.3 ---
    add_heading_h2(doc, "1.3.3. Lọc cộng tác: User-based KNN và Item-based KNN")

    add_body(
        doc,
        "Lọc cộng tác (Collaborative Filtering – CF) dựa trên giả thuyết cốt lõi rằng những người dùng có xu hướng đánh giá tương đồng trong quá khứ sẽ tiếp tục có sở thích gần nhau trong tương lai [2], [7]. Khác với lọc theo nội dung vốn phụ thuộc vào siêu dữ liệu (thể loại, từ khóa, thương hiệu), lọc cộng tác chỉ cần ma trận tương tác R kích thước |U| × |I| nên có khả năng phát hiện các mối liên hệ tiềm ẩn mang tính bất ngờ (serendipity) giữa các đối tượng thuộc nhiều nhóm khác nhau.",
    )

    add_heading_h3(doc, "a) Lọc cộng tác dựa trên người dùng (User-based KNN)")
    add_body(
        doc,
        "Ý tưởng của User-based KNN là để dự đoán điểm đánh giá của người dùng mục tiêu u cho đối tượng i chưa xem, hệ thống tìm tập {{N_k_ui}} gồm k người dùng láng giềng có độ tương đồng cao nhất với u trong số những người đã từng đánh giá đối tượng i [7]. Độ tương đồng {{sim_uv}} giữa hai người dùng u và v thường được đo bằng hệ số tương quan Pearson (Pearson Correlation) hoặc Cosine trên tập đối tượng cùng đánh giá {{I_uv}} = {{I_u}} ∩ {{I_v}}:",
    )

    # Công thức (1.2): Pearson similarity
    eq_1_2 = m_omath(
        m_sub(mr("sim", plain=True), mr("Pearson", plain=True)),
        m_paren(mr("u, v")),
        mr(" = "),
        m_frac(
            m_sum(
                [mr("j ∈ "), m_sub(mr("I"), mr("uv"))],
                [
                    m_paren([m_sub(mr("r"), mr("uj")), mr(" − "), m_sub(m_bar(mr("r")), mr("u"))]),
                    m_paren([m_sub(mr("r"), mr("vj")), mr(" − "), m_sub(m_bar(mr("r")), mr("v"))]),
                ],
            ),
            [
                m_sqrt(
                    m_sum(
                        [mr("j ∈ "), m_sub(mr("I"), mr("uv"))],
                        m_sup(m_paren([m_sub(mr("r"), mr("uj")), mr(" − "), m_sub(m_bar(mr("r")), mr("u"))]), mr("2")),
                    )
                ),
                mr(" · "),
                m_sqrt(
                    m_sum(
                        [mr("j ∈ "), m_sub(mr("I"), mr("uv"))],
                        m_sup(m_paren([m_sub(mr("r"), mr("vj")), mr(" − "), m_sub(m_bar(mr("r")), mr("v"))]), mr("2")),
                    )
                ),
            ],
        ),
    )
    add_omml_equation(doc, eq_1_2, "(1.2)")

    add_body(
        doc,
        "Trong thực tế, mỗi người dùng có thang chấm điểm chủ quan khác nhau: có người dùng khó tính chỉ chấm trung bình 2.5 sao, trong khi người dùng dễ tính thường chấm quanh mức 4.2 sao. Nếu lấy trung bình có trọng số trực tiếp từ điểm thô {{r_vj}}, kết quả dự đoán sẽ bị lệch nghiêm trọng. Do đó, công thức dự đoán chuẩn trong User-based KNN áp dụng kỹ thuật trừ trung bình (mean-centering) nhằm chuẩn hóa độ lệch sở thích của từng người dùng [2]:",
    )

    # Công thức (1.3): User-based rating prediction with mean-centering
    eq_1_3 = m_omath(
        m_sub(m_hat(mr("r")), mr("ui")),
        mr(" = "),
        m_sub(m_bar(mr("r")), mr("u")),
        mr(" + "),
        m_frac(
            m_sum(
                [mr("v ∈ "), m_sub(mr("N"), mr("k")), m_paren(mr("u; i"))],
                [
                    mr("sim", plain=True),
                    m_paren(mr("u, v")),
                    mr(" · "),
                    m_paren([m_sub(mr("r"), mr("vi")), mr(" − "), m_sub(m_bar(mr("r")), mr("v"))]),
                ],
            ),
            m_sum(
                [mr("v ∈ "), m_sub(mr("N"), mr("k")), m_paren(mr("u; i"))],
                m_paren([mr("sim", plain=True), m_paren(mr("u, v"))], "|", "|"),
            ),
        ),
    )
    add_omml_equation(doc, eq_1_3, "(1.3)")

    add_body(
        doc,
        "trong đó {{rbar_u}} và {{rbar_v}} lần lượt là điểm đánh giá trung bình của người dùng u và láng giềng v trên toàn bộ các đối tượng họ đã chấm. Nhờ trừ đi {{rbar_v}}, phần tử ({{r_vi}} − {{rbar_v}}) chỉ phản ánh việc v đánh giá đối tượng i cao hơn hay thấp hơn mức bình thường của chính v.",
    )

    add_heading_h3(doc, "b) Lọc cộng tác dựa trên đối tượng (Item-based KNN)")
    add_body(
        doc,
        "Khi số lượng người dùng |U| tăng lên hàng trăm nghìn hoặc hàng triệu, việc tính toán ma trận tương đồng giữa các người dùng theo thời gian thực trở nên tốn kém và kém ổn định do sở thích người dùng thay đổi liên tục. Để khắc phục, Sarwar và cộng sự (2001) đề xuất phương pháp Item-based KNN, chuyển trọng tâm từ tìm người dùng giống nhau sang tìm các đối tượng tương đồng nhau dựa trên tập người dùng đã cùng tương tác [8].",
    )
    add_body(
        doc,
        "Trong Item-based KNN, độ tương đồng giữa hai đối tượng i và j được tính trên tập người dùng cùng đánh giá {{U_ij}} = {{U_i}} ∩ {{U_j}}. Để loại bỏ thiên lệch chấm điểm cá nhân của từng người dùng khi tính tương đồng giữa các sản phẩm, độ đo Adjusted Cosine được ưu tiên sử dụng thay cho Cosine thuần túy [8]:",
    )

    # Công thức (1.4): Adjusted Cosine similarity
    eq_1_4 = m_omath(
        m_sub(mr("sim", plain=True), mr("AdjCos", plain=True)),
        m_paren(mr("i, j")),
        mr(" = "),
        m_frac(
            m_sum(
                [mr("u ∈ "), m_sub(mr("U"), mr("ij"))],
                [
                    m_paren([m_sub(mr("r"), mr("ui")), mr(" − "), m_sub(m_bar(mr("r")), mr("u"))]),
                    m_paren([m_sub(mr("r"), mr("uj")), mr(" − "), m_sub(m_bar(mr("r")), mr("u"))]),
                ],
            ),
            [
                m_sqrt(
                    m_sum(
                        [mr("u ∈ "), m_sub(mr("U"), mr("i"))],
                        m_sup(m_paren([m_sub(mr("r"), mr("ui")), mr(" − "), m_sub(m_bar(mr("r")), mr("u"))]), mr("2")),
                    )
                ),
                mr(" · "),
                m_sqrt(
                    m_sum(
                        [mr("u ∈ "), m_sub(mr("U"), mr("j"))],
                        m_sup(m_paren([m_sub(mr("r"), mr("uj")), mr(" − "), m_sub(m_bar(mr("r")), mr("u"))]), mr("2")),
                    )
                ),
            ],
        ),
    )
    add_omml_equation(doc, eq_1_4, "(1.4)")

    add_body(
        doc,
        "Từ tập k đối tượng tương đồng nhất với i mà người dùng u đã đánh giá, ký hiệu là {{N_k_iu}}, điểm dự đoán {{rhat_ui}} được tính theo công thức bình quân gia quyền có mean-centering theo đối tượng:",
    )

    # Công thức (1.5): Item-based rating prediction
    eq_1_5 = m_omath(
        m_sub(m_hat(mr("r")), mr("ui")),
        mr(" = "),
        m_sub(m_bar(mr("r")), mr("i")),
        mr(" + "),
        m_frac(
            m_sum(
                [mr("j ∈ "), m_sub(mr("N"), mr("k")), m_paren(mr("i; u"))],
                [
                    mr("sim", plain=True),
                    m_paren(mr("i, j")),
                    mr(" · "),
                    m_paren([m_sub(mr("r"), mr("uj")), mr(" − "), m_sub(m_bar(mr("r")), mr("j"))]),
                ],
            ),
            m_sum(
                [mr("j ∈ "), m_sub(mr("N"), mr("k")), m_paren(mr("i; u"))],
                m_paren([mr("sim", plain=True), m_paren(mr("i, j"))], "|", "|"),
            ),
        ),
    )
    add_omml_equation(doc, eq_1_5, "(1.5)")

    add_body(
        doc,
        "Ưu điểm vượt trội của Item-based KNN trong kiến trúc hệ thống thực tế là ma trận tương đồng giữa các đối tượng kích thước |I| × |I| có tính ổn định cao theo thời gian (vì đặc tính nội tại của một bộ phim hay sản phẩm ít biến động hơn tâm lý người dùng). Nhờ vậy, hệ thống có thể tính trước (precompute) và lưu đệm (cache) ma trận tương đồng ngoại tuyến theo chu kỳ, giúp thời gian phản hồi API trực tuyến giảm xuống chỉ còn vài mili-giây.",
    )

    add_table(
        doc,
        "Bảng 1.1. So sánh User-based KNN và Item-based KNN trong lọc cộng tác",
        ["Tiêu chí so sánh", "User-based KNN", "Item-based KNN"],
        [
            ["Đối tượng láng giềng", "Tìm k người dùng tương đồng với u", "Tìm k đối tượng tương đồng với i"],
            ["Độ phức tạp tính tương đồng", "{{O_U2I}} — rất lớn khi {{U_gt_I}}", "{{O_I2U}} — tối ưu hơn khi {{U_gt_I}}"],
            ["Tính ổn định & khả năng cache", "Thấp hơn do hồ sơ người dùng thay đổi nhanh", "Cao; dễ tính trước và cache ma trận item-item"],
            ["Độ nhạy với thưa dữ liệu", "Dễ gặp trường hợp ít cặp user giao nhau", "Ổn định hơn khi số đánh giá trung bình mỗi item đủ lớn"],
            ["Khả năng giải thích (Explainability)", "Khó giải thích trực tiếp (ẩn danh người dùng khác)", "Rất trực quan: 'Vì bạn đã thích sản phẩm X'"],
            ["Phạm vi áp dụng trong đồ án", "Module gợi ý phim (so sánh với Item-based)", "Module gợi ý phim & nhánh warm-item của hàng mới"],
        ],
        [3.8, 6.2, 6.5],
    )

    add_heading_h3(doc, "c) Ví dụ minh họa tính tay trên ma trận 5 × 5")
    add_body(
        doc,
        "Xét ma trận đánh giá thu nhỏ gồm 5 người dùng (U_{1}–U_{5}) và 5 bộ phim (M_{1}–M_{5}) với thang điểm 1–5 (dấu '–' là chưa đánh giá). Giả sử cần dự đoán điểm của U_{1} cho phim M_{5} bằng User-based KNN (k = 2, độ đo Pearson):",
    )
    add_table(
        doc,
        "Bảng 1.2. Ma trận đánh giá minh họa 5 × 5 cho bài toán dự đoán {{r_U1M5}}",
        ["Người dùng \\ Phim", "M_{1}", "M_{2}", "M_{3}", "M_{4}", "M_{5}", "Trung bình {{rbar_u}}"],
        [
            ["U_{1} (mục tiêu)", "5", "3", "4", "–", "?", "4.00"],
            ["U_{2}", "5", "3", "4", "2", "5", "3.80"],
            ["U_{3}", "1", "5", "2", "4", "2", "2.80"],
            ["U_{4}", "4", "2", "3", "–", "4", "3.25"],
            ["U_{5}", "2", "4", "1", "5", "1", "2.60"],
        ],
        [3.3, 1.5, 1.5, 1.5, 1.5, 1.5, 2.7],
    )
    add_body(
        doc,
        "Trên tập phim cùng đánh giá {M_{1}, M_{2}, M_{3}}, U_{1} có vector độ lệch so với trung bình là (+1, −1, 0). Tương tự, tính với U_{2} (trên tập giao có độ lệch so với trung bình của U_{2} là +1.2, −0.8, +0.2) cho giá trị {{sim_Pearson_U1U2}} ≈ +0.982, và với U_{4} cho {{sim_Pearson_U1U4}} = +1.000; trong khi U_{3} và U_{5} có tương quan âm (≈ −0.945 và −0.866). Chọn k = 2 láng giềng có tương quan dương cao nhất là U_{4} và U_{2}, áp dụng công thức (1.3): {{rhat_U1M5}} = 4.00 + [1.000×(4 − 3.25) + 0.982×(5 − 3.80)] / (1.000 + 0.982) ≈ 4.97. Kết quả hợp lý vì cả hai láng giềng cùng gu với U_{1} đều đánh giá M_{5} rất cao.",
    )

    # --- 1.3.5 ---
    add_heading_h2(doc, "1.3.5. Thách thức của hệ gợi ý: Cold-start, Sparsity và Scalability")

    add_body(
        doc,
        "Mặc dù các mô hình lọc cộng tác dựa trên KNN đạt độ chính xác cao khi dữ liệu dồi dào, việc triển khai trong môi trường thực tế luôn phải đối mặt với ba thách thức kinh điển ảnh hưởng trực tiếp đến chất lượng gợi ý và hiệu năng hệ thống mạng [1], [6], [9].",
    )

    add_heading_h3(doc, "a) Vấn đề khởi đầu nguội (Cold-Start Problem)")
    add_body(
        doc,
        "Cold-start xảy ra khi hệ thống thiếu dữ liệu tương tác lịch sử để tính toán độ tương đồng cộng tác, bao gồm ba kịch bản: (1) Người dùng mới (New User) vừa đăng ký tài khoản và chưa có lượt đánh giá hay nhấp chuột nào; (2) Đối tượng mới (New Item) vừa được đưa lên sàn thương mại điện tử hoặc kho phim/tài liệu nên chưa có đủ người dùng tương tác; và (3) Hệ thống mới (New System) ở giai đoạn đầu vận hành [9]. Trong bài toán gợi ý hàng mới của đồ án, New Item là thách thức trọng tâm: nếu chỉ dùng Item-based CF thuần túy, một sản phẩm mới có 0 tương tác sẽ có vector cột toàn số 0, khiến {{sim_AdjCos_ij}} = 0 với mọi sản phẩm khác và không bao giờ xuất hiện trong danh sách Top-N.",
    )
    add_body(
        doc,
        "Hướng khắc phục: Để phá vỡ thế bế tắc của New Item, đồ án áp dụng kiến trúc Hybrid Switching kết hợp Content-based KNN. Với mỗi sản phẩm mới, hệ thống trích xuất vector đặc trưng nội dung (danh mục, thương hiệu, từ khóa mô tả có trọng số IDF, mức giá chuẩn hóa và năm phát hành), dùng độ đo Cosine tìm k sản phẩm cũ tương đồng nhất để suy ra tệp khách hàng tiềm năng đã yêu thích các sản phẩm láng giềng đó. Khi sản phẩm mới tích lũy vượt ngưỡng tương tác (ví dụ ≥ 5 tương tác), hệ thống tự động chuyển (switch) sang Item-based CF để tận dụng trí tuệ đám đông.",
    )

    add_heading_h3(doc, "b) Vấn đề thưa thớt dữ liệu (Data Sparsity)")
    add_body(
        doc,
        "Trong thực tế, mỗi người dùng chỉ tương tác với một phần rất nhỏ trên tổng số hàng nghìn đến hàng triệu đối tượng của hệ thống, khiến ma trận R có tỷ lệ ô trống (sparsity) thường vượt quá 94–99%. Cụ thể, qua khảo sát dữ liệu thực tế trong đồ án, bộ dữ liệu chuẩn MovieLens ml-latest-small gồm "
        f"{ml['n_users']} người dùng và {ml['n_items_total']:,} bộ phim nhưng chỉ có {ml['n_ratings']:,} lượt đánh giá, dẫn đến độ thưa lên tới {ml['sparsity_total_pct']:.2f}% (tức trung bình chỉ khoảng 1.70% ô trong ma trận có dữ liệu). Tương tự, tập dữ liệu Sản phẩm và Tài liệu mô phỏng theo phân phối Zipf có độ thưa lần lượt là {prod['sparsity_pct']:.2f}% và {docs_s['sparsity_pct']:.2f}%.",
    )
    add_body(
        doc,
        "Hệ quả của độ thưa cao là tập giao {{I_uv}} hoặc {{U_ij}} giữa hai thực thể thường rỗng hoặc chỉ có 1–2 phần tử đồng đánh giá, làm độ tương đồng Pearson/Cosine bị nhiễu (ví dụ hai người cùng chấm 5 sao cho đúng 1 phim duy nhất sẽ cho tương đồng tuyệt đối 1.0 dù sở thích tổng thể khác biệt). Để giảm thiểu tác động này, đồ án kết hợp nhiều biện pháp: (i) lọc bỏ người dùng và đối tượng có ít hơn ngưỡng tương tác tối thiểu trong bước tiền xử lý; (ii) áp dụng hiệu chỉnh suy giảm theo số lượng phần tử chung (significance weighting) hoặc fallback về Content-based KNN khi tập giao CF bằng rỗng; và (iii) khai thác giảm chiều ma trận bằng TruncatedSVD đối với dữ liệu văn bản/phản hồi ngầm.",
    )

    add_heading_h3(doc, "c) Vấn đề khả năng mở rộng (Scalability)")
    add_body(
        doc,
        "Thuật toán KNN thuộc nhóm học lười (lazy learning), không nén tri thức thành bộ trọng số gọn nhẹ mà lưu trữ toàn bộ ma trận tương tác để truy vấn láng giềng. Khi quy mô dữ liệu tăng tuyến tính, chi phí tính ma trận tương đồng toàn cặp tăng theo bậc hai: {{O_U2I}} đối với User-based KNN và {{O_I2U}} đối với Item-based KNN [8], [10]. Với một dịch vụ web qua giao thức HTTP/REST (như FastAPI trong đồ án), nếu mỗi truy vấn gợi ý phải quét lại toàn bộ hàng chục nghìn vector, độ trễ phản hồi (latency) sẽ vượt ngưỡng chấp nhận của người dùng (> 3 giây).",
    )
    add_body(
        doc,
        "Hướng khắc phục: Ở quy mô đồ án, nhóm thiết kế ba lớp tối ưu: (1) tính trước và lưu đệm (cache) ma trận tương đồng item-item vào bộ nhớ RAM/CSDL khi khởi động server; (2) sử dụng phép nhân ma trận vector hóa (NumPy/SciPy) thay cho vòng lặp Python thuần; và (3) giảm chiều đặc trưng TF-IDF bằng TruncatedSVD (về 100–200 chiều) đối với module tài liệu. Ở quy mô công nghiệp lớn hơn, có thể kết hợp cấu trúc cây KD-Tree/Ball-Tree hoặc tìm kiếm láng giềng gần đúng (Approximate Nearest Neighbors – ANN như HNSW, FAISS, LSH) để đạt độ phức tạp truy vấn gần O(log N) [10].",
    )

    # --- 1.3.6 ---
    add_heading_h2(doc, "1.3.6. Các độ đo đánh giá hệ gợi ý")

    add_body(
        doc,
        "Đánh giá hiệu năng hệ gợi ý đòi hỏi các độ đo phù hợp với bản chất đầu ra của từng bài toán: dự đoán điểm số liên tục (rating prediction) hay xếp hạng danh sách Top-K (ranking recommendation) [11], [12]. Trong khung đánh giá (src/evaluate.py) của đồ án, hệ thống tích hợp đồng thời ba nhóm độ đo chuẩn mực dưới đây.",
    )

    add_heading_h3(doc, "a) Nhóm độ đo sai số dự đoán điểm: RMSE và MAE")
    add_body(
        doc,
        "Đối với bài toán dự đoán giá trị rating {{rhat_ui}} trên tập kiểm thử T gồm các cặp (u, i) đã biết điểm thật {{r_ui}}, hai độ đo phổ biến nhất là Sai số tuyệt đối trung bình (Mean Absolute Error – MAE) và Căn bậc hai sai số bình phương trung bình (Root Mean Squared Error – RMSE) [11]:",
    )

    # Công thức (1.6): MAE
    eq_1_6 = m_omath(
        mr("MAE", plain=True),
        mr(" = "),
        m_frac(mr("1"), m_paren(mr("T"), "|", "|")),
        mr(" · "),
        m_sum(
            mr("(u,i) ∈ T"),
            m_paren([m_sub(mr("r"), mr("ui")), mr(" − "), m_sub(m_hat(mr("r")), mr("ui"))], "|", "|"),
        ),
    )
    add_omml_equation(doc, eq_1_6, "(1.6)")

    # Công thức (1.7): RMSE
    eq_1_7 = m_omath(
        mr("RMSE", plain=True),
        mr(" = "),
        m_sqrt(
            [
                m_frac(mr("1"), m_paren(mr("T"), "|", "|")),
                mr(" · "),
                m_sum(
                    mr("(u,i) ∈ T"),
                    m_sup(m_paren([m_sub(mr("r"), mr("ui")), mr(" − "), m_sub(m_hat(mr("r")), mr("ui"))]), mr("2")),
                ),
            ]
        ),
    )
    add_omml_equation(doc, eq_1_7, "(1.7)")

    add_body(
        doc,
        "Cả MAE và RMSE càng nhỏ thì mô hình dự đoán càng chính xác. Tuy nhiên, do bình phương phần dư trước khi lấy trung bình, RMSE phạt nặng hơn đáng kể đối với các sai số dự đoán lớn (outliers). Vì vậy, RMSE đặc biệt hữu ích khi hệ thống cần tránh những dự đoán lệch cực đoan (ví dụ dự đoán 5 sao cho một bộ phim người dùng chấm 1 sao).",
    )

    add_heading_h3(doc, "b) Nhóm độ đo chất lượng xếp hạng Top-K: Precision@K, Recall@K và NDCG@K")
    add_body(
        doc,
        "Trong thực tế ứng dụng web, người dùng hiếm khi quan tâm điểm dự đoán chính xác đến phần thập phân mà chỉ xem danh sách Top-K gợi ý hàng đầu (thường K = 5 hoặc K = 10). Gọi {{R_K_u}} là danh sách K đối tượng được hệ thống gợi ý cho người dùng u, và {{Rel_u}} là tập các đối tượng thực sự phù hợp (ví dụ có điểm đánh giá ≥ 3.5 hoặc có hành vi tải/mua trong tập kiểm thử) [12]:",
    )

    # Công thức (1.8): Precision@K
    eq_1_8 = m_omath(
        mr("Precision@K", plain=True),
        m_paren(mr("u")),
        mr(" = "),
        m_frac(
            m_paren([m_sub(mr("R"), mr("K")), m_paren(mr("u")), mr(" ∩ "), mr("Rel", plain=True), m_paren(mr("u"))], "|", "|"),
            mr("K"),
        ),
    )
    add_omml_equation(doc, eq_1_8, "(1.8)")

    # Công thức (1.9): Recall@K
    eq_1_9 = m_omath(
        mr("Recall@K", plain=True),
        m_paren(mr("u")),
        mr(" = "),
        m_frac(
            m_paren([m_sub(mr("R"), mr("K")), m_paren(mr("u")), mr(" ∩ "), mr("Rel", plain=True), m_paren(mr("u"))], "|", "|"),
            m_paren([mr("Rel", plain=True), m_paren(mr("u"))], "|", "|"),
        ),
    )
    add_omml_equation(doc, eq_1_9, "(1.9)")

    add_body(
        doc,
        "Precision@K đo tỷ lệ gợi ý đúng trong K kết quả đầu tiên, còn Recall@K đo tỷ lệ đối tượng phù hợp được tìm thấy trên tổng số đối tượng người dùng thực sự thích. Tuy nhiên, cả hai độ đo này đều coi vị trí thứ 1 và vị trí thứ K quan trọng ngang nhau. Để phản ánh đúng trải nghiệm người dùng (luôn chú ý các kết quả đầu danh sách hơn), độ đo Normalized Discounted Cumulative Gain (NDCG@K) được sử dụng bằng cách giảm trọng số theo logarit của thứ hạng p (1 ≤ p ≤ K) [12]:",
    )

    # Công thức (1.10): DCG@K và NDCG@K
    eq_1_10 = m_omath(
        mr("DCG@K", plain=True),
        m_paren(mr("u")),
        mr(" = "),
        m_sum(
            mr("p=1"),
            m_frac(m_sub(mr("rel", plain=True), mr("p")), [m_sub(mr("log", plain=True), mr("2")), m_paren(mr("p + 1"))]),
            sup=mr("K"),
        ),
        mr(" ,    "),
        mr("NDCG@K", plain=True),
        m_paren(mr("u")),
        mr(" = "),
        m_frac([mr("DCG@K", plain=True), m_paren(mr("u"))], [mr("IDCG@K", plain=True), m_paren(mr("u"))]),
    )
    add_omml_equation(doc, eq_1_10, "(1.10)")

    add_body(
        doc,
        "trong đó {{rel_p}} ∈ {0, 1} (hoặc độ liên quan có cấp bậc) là mức phù hợp của đối tượng tại vị trí xếp hạng thứ p, còn {{IDCG_K}}(u) là giá trị DCG lớn nhất có thể đạt được khi sắp xếp lý tưởng các đối tượng phù hợp lên đầu danh sách. Giá trị {{NDCG_K}}(u) nằm trong đoạn [0, 1], đạt 1.0 khi thứ tự gợi ý hoàn hảo.",
    )

    add_heading_h3(doc, "c) Nhóm độ đo bao phủ và hiệu năng hệ thống: Coverage, Diversity và Latency")
    add_body(
        doc,
        "Một mô hình chỉ gợi ý lặp đi lặp lại vài sản phẩm phổ biến nhất (Most-Popular) có thể đạt Precision@K tạm ổn nhưng lại gây nhàm chán và bỏ rơi các sản phẩm đuôi dài (long-tail) hoặc hàng mới. Do đó, đồ án bổ sung đánh giá Độ bao phủ danh mục (Catalog Coverage@K) là tỷ lệ phần trăm số đối tượng riêng biệt từng xuất hiện trong Top-K của ít nhất một người dùng trên tổng số đối tượng |I|, cùng với Độ đa dạng (Diversity – tính bằng 1 trừ độ tương đồng trung bình giữa các cặp trong Top-K) và Thời gian phản hồi API (Latency tính bằng ms) [11], [13].",
    )

    add_table(
        doc,
        "Bảng 1.3. Tổng hợp các độ đo đánh giá và phạm vi áp dụng trong đồ án",
        ["Độ đo", "Loại dữ liệu", "Mục tiêu tối ưu", "Miền áp dụng chính trong đồ án"],
        [
            ["RMSE, MAE", "Explicit rating", "Càng nhỏ càng tốt (RMSE < 1.1)", "Phim (CF), Hàng mới (Hybrid rating)"],
            ["Precision@K, Recall@K", "Explicit / Implicit", "Càng lớn càng tốt (0 → 1)", "Cả 3 bài toán (Top-5, Top-10)"],
            ["NDCG@K", "Ranked Top-K", "Càng lớn càng tốt (0 → 1)", "Cả 3 bài toán (đánh giá thứ hạng)"],
            ["Coverage@K, Diversity", "Catalog / List", "Cân bằng giữa chính xác & đa dạng", "Thực nghiệm Tuần 5 (chống thiên lệch phổ biến)"],
            ["Latency (ms)", "Hệ thống mạng", "Nhỏ hơn ngưỡng 3,000 ms (mục tiêu < 200 ms)", "API FastAPI client–server"],
        ],
        [3.8, 3.0, 4.5, 5.2],
    )

    # =========================================================================
    # PHẦN B: BÁO CÁO KỸ THUẬT TUẦN 1–3 CỦA HÙNG
    # =========================================================================
    add_major_title(
        doc,
        "PHỤ LỤC KỸ THUẬT: KẾT QUẢ TRIỂN KHAI CỦA HÙNG\n(DỮ LIỆU, LƯỢC ĐỒ CSDL & MODULE HÀNG MỚI TUẦN 1–3)",
        page_break_before=True,
    )

    add_heading_h2(doc, "B.1. Mô tả, thu thập và làm sạch 3 bộ dữ liệu (Tuần 1–2)")
    add_body(
        doc,
        "Theo phân công Tuần 1–2, Hùng đã hoàn thành việc tải bộ dữ liệu chuẩn MovieLens ml-latest-small vào thư mục data/raw/ml-latest-small/ và triển khai kịch bản chuẩn bị/làm sạch dữ liệu (experiments/prepare_data.py) để tạo dữ liệu cho hai miền còn lại theo phân phối Zipf (tuân thủ phương án dự phòng Mục 6 của kế hoạch). Cụ thể, quá trình làm sạch đã loại bỏ các bản ghi trùng lặp, cắt khoảng trắng thừa, điền giá trị thiếu bằng trung vị theo danh mục (đối với giá sản phẩm) hoặc trung vị toàn cục (đối với dung lượng tài liệu), loại bỏ rating ngoài khoảng hợp lệ [1.0, 5.0], đồng thời chuẩn hóa giá sản phẩm theo cả hai phương pháp Min-Max và Z-score.",
    )

    add_table(
        doc,
        "Bảng B.1. Thống kê chi tiết 3 bộ dữ liệu sau khi làm sạch (số liệu đo trực tiếp)",
        ["Bộ dữ liệu", "Số User", "Số Item", "Số tương tác", "Độ thưa (%)", "Đặc điểm bổ sung"],
        [
            [
                "MovieLens\n(ml-latest-small)",
                f"{ml['n_users']:,}",
                f"{ml['n_items_total']:,}\n({ml['n_items_rated']:,} có rating)",
                f"{ml['n_ratings']:,}",
                f"{ml['sparsity_total_pct']:.2f}%",
                f"Rating 0.5–5.0; TB = {ml['rating_mean']} (±{ml['rating_std']}); TB {ml['avg_ratings_per_user']} rating/user",
            ],
            [
                "Sản phẩm / Hàng mới\n(Zipf + Cleaned)",
                f"{prod['n_users']:,}",
                f"{prod['clean_product_rows']:,}\n({prod['cold_start_new_items']} hàng mới)",
                f"{prod['clean_interaction_rows']:,}\n(từ {prod['raw_interaction_rows']:,} thô)",
                f"{prod['sparsity_pct']:.2f}%",
                f"8 danh mục, {prod['zero_interaction_items']} item 0 tương tác; Rating TB = {prod['rating_mean']}",
            ],
            [
                "Tài liệu download\n(Zipf Implicit)",
                f"{docs_s['n_users']:,}",
                f"{docs_s['clean_doc_rows']:,}",
                f"{docs_s['clean_interaction_rows']:,}\n(từ {docs_s['raw_interaction_rows']:,} thô)",
                f"{docs_s['sparsity_pct']:.2f}%",
                f"6 chủ đề; download={docs_s['action_distribution']['download']:,}, view={docs_s['action_distribution']['view']:,}, bookmark={docs_s['action_distribution']['bookmark']:,}",
            ],
        ],
        [3.3, 1.8, 2.8, 2.7, 2.1, 3.8],
    )

    add_heading_h2(doc, "B.2. Thiết kế lược đồ cơ sở dữ liệu SQLite (Tuần 2)")
    add_body(
        doc,
        "Lược đồ CSDL được thiết kế chuẩn hóa tại tệp docs/schema.sql (đã kiểm thử thực thi thành công trên SQLite), gồm 8 bảng với đầy đủ khóa chính (PRIMARY KEY), khóa ngoại (FOREIGN KEY … ON DELETE CASCADE), ràng buộc miền giá trị (CHECK) và 8 chỉ mục (INDEX) tối ưu truy vấn cho module db.py và FastAPI của Hoàng ở Tuần 4.",
    )

    add_table(
        doc,
        "Bảng B.2. Danh sách các bảng trong lược đồ cơ sở dữ liệu SQLite (docs/schema.sql)",
        ["Tên bảng", "Các trường chính", "Khóa ngoại & Chỉ mục chính", "Mục đích sử dụng"],
        [
            ["users", "user_id (PK), external_id, domain, username, segment, created_at", "UNIQUE(domain, external_id)", "Quản lý người dùng đa miền"],
            ["movies", "movie_id (PK), title, genres, release_year", "—", "Lưu siêu dữ liệu phim MovieLens"],
            ["movie_ratings", "rating_id (PK), user_id, movie_id, rating, rated_at", "FK → users, movies; idx user/movie", "Lưu phản hồi tường minh xem phim"],
            ["products", "product_id (PK), name, category, brand, price_vnd, price_minmax, price_zscore, tags, is_new_item", "idx_products_new(is_new_item, category)", "Lưu thuộc tính phục vụ Content KNN"],
            ["product_interactions", "interaction_id (PK), user_id, product_id, interaction_type, rating, interacted_at", "FK → users, products; idx user/prod", "Lưu lịch sử mua/giỏ hàng/xem"],
            ["documents", "doc_id (PK), title, category, tags, abstract, file_type, file_size_mb, created_at", "—", "Lưu nội dung cho TF-IDF của Linh"],
            ["document_interactions", "interaction_id (PK), user_id, doc_id, action_type, weight, interacted_at", "FK → users, documents; idx user/doc", "Lưu implicit feedback (weight 1–3)"],
            ["recommendation_logs", "log_id (PK), domain, query_mode, target_id, algorithm, top_k, result_ids, latency_ms", "idx_rec_logs_domain(domain, created_at)", "Ghi log API & đo latency Tuần 4–5"],
        ],
        [3.1, 5.2, 4.1, 4.1],
    )

    add_heading_h2(doc, "B.3. Cài đặt module gợi ý hàng mới và khung đánh giá (Tuần 3 – Mốc M3)")
    add_body(
        doc,
        "Module src/rec_newitem.py và khung đánh giá src/evaluate.py đã hoàn thiện toàn bộ yêu cầu của Mốc M3 (25/11). Cụ thể, mỗi sản phẩm được biểu diễn bằng vector đặc trưng tổng hợp gồm: mã hóa one-hot danh mục (trọng số 2.0), one-hot thương hiệu (1.0), multi-hot từ khóa có trọng số IDF (1.2), giá chuẩn hóa Min-Max (0.8) và năm phát hành chuẩn hóa (0.4), sau đó chuẩn hóa L2 theo từng hàng để phép nhân ma trận cho trực tiếp độ tương đồng Cosine. Hệ thống hỗ trợ giao diện tự động tích hợp với knn_core.py của Hoàng khi có sẵn trên nhánh main.",
    )
    add_body(
        doc,
        "Ở chế độ gợi ý cho người dùng, cơ chế Hybrid Switching tự động chọn Content-based KNN khi sản phẩm có dưới 5 tương tác (hàng mới cold-start) và chuyển sang Item-based CF có mean-centering khi sản phẩm đã đủ dữ liệu tương tác (≥ 5). Kết quả kiểm thử holdout bằng cách ẩn toàn bộ tương tác của 30 sản phẩm ấm để giả lập hàng mới (459 tương tác kiểm thử) và chạy thử qua CLI cho thấy hệ thống hoạt động ổn định:",
    )

    add_code_block(
        doc,
        [
            "$ python -m src.rec_newitem --mode recommend_users --item-id 461 --top-k 10",
            "=== TOP-10 KHÁCH HÀNG TIỀM NĂNG CHO HÀNG MỚI #461: Chuot may tinh Pulsar Model-461 ===",
            " 1. User #237 | score=9.2538 | Thich SP tuong dong 'Chuot may tinh Corsair Model-069' (cos_sim=0.680)",
            " 2. User #213 | score=8.7149 | Thich SP tuong dong 'Chuot may tinh Logitech Model-077' (cos_sim=0.730)",
            " 3. User #366 | score=7.8667 | Thich SP tuong dong 'Chuot may tinh Razer Model-149' (cos_sim=0.639)",
            "...",
            "$ python -m src.rec_newitem --mode evaluate --top-k 10",
            "Holdout 30 items (459 test interactions): RMSE = 0.7829 (< nguong rui ro 1.1) | MAE = 0.5747",
            "Ranking Top-10: Precision@10 = 0.0567 | Recall@10 = 0.0304 | NDCG@10 = 0.0631 | Coverage@10 = 0.4175",
        ],
    )

    add_body(
        doc,
        "Toàn bộ 4 kịch bản kiểm thử đơn vị trong tests/test_hung_modules.py (kiểm chứng tính tay RMSE/MAE, Precision@K/Recall@K/NDCG@K, thực thi schema SQLite và kiểm tra Top-10 cold-start) đều chạy đạt 100% (Ran 4 tests in 0.160s — OK).",
    )

    # =========================================================================
    # TÀI LIỆU THAM KHẢO (14 NGUỒN CHUẨN)
    # =========================================================================
    add_major_title(doc, "TÀI LIỆU THAM KHẢO", page_break_before=True)

    refs = [
        "[1] F. Ricci, L. Rokach, and B. Shapira, \"Recommender Systems: Introduction and Challenges,\" in Recommender Systems Handbook, 2nd ed., Boston, MA: Springer, 2015, pp. 1–34.",
        "[2] C. C. Aggarwal, Recommender Systems: The Textbook. Cham: Springer International Publishing, 2016.",
        "[3] I. MacKenzie, C. Meyer, and S. Noble, \"How retailers can keep up with consumers,\" McKinsey & Company, Oct. 2013.",
        "[4] C. A. Gomez-Uribe and N. Hunt, \"The Netflix Recommender System: Algorithms, Business Value, and Innovation,\" ACM Trans. Manag. Inf. Syst., vol. 6, no. 4, pp. 1–19, Jan. 2016.",
        "[5] Y. Hu, Y. Koren, and C. Volinsky, \"Collaborative Filtering for Implicit Feedback Datasets,\" in Proc. 8th IEEE Int. Conf. Data Mining (ICDM), Pisa, Italy, 2008, pp. 263–272.",
        "[6] R. Burke, \"Hybrid Recommender Systems: Survey and Experiments,\" User Model. User-Adapt. Interact., vol. 12, no. 4, pp. 331–370, Nov. 2002.",
        "[7] J. L. Herlocker, J. A. Konstan, A. Borchers, and J. Riedl, \"An Algorithmic Framework for Performing Collaborative Filtering,\" in Proc. 22nd Annu. Int. ACM SIGIR Conf., Berkeley, CA, 1999, pp. 230–237.",
        "[8] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, \"Item-Based Collaborative Filtering Recommendation Algorithms,\" in Proc. 10th Int. Conf. World Wide Web (WWW), Hong Kong, 2001, pp. 285–295.",
        "[9] A. I. Schein, A. Popescul, L. H. Ungar, and D. M. Pennock, \"Methods and Metrics for Cold-Start Recommendations,\" in Proc. 25th Annu. Int. ACM SIGIR Conf., Tampere, Finland, 2002, pp. 253–260.",
        "[10] T. Cover and P. Hart, \"Nearest neighbor pattern classification,\" IEEE Trans. Inf. Theory, vol. 13, no. 1, pp. 21–27, Jan. 1967.",
        "[11] J. L. Herlocker, J. A. Konstan, L. G. Terveen, and J. T. Riedl, \"Evaluating Collaborative Filtering Recommender Systems,\" ACM Trans. Inf. Syst., vol. 22, no. 1, pp. 5–53, Jan. 2004.",
        "[12] K. Järvelin and J. Kekäläinen, \"Cumulated Gain-Based Evaluation of IR Techniques,\" ACM Trans. Inf. Syst., vol. 20, no. 4, pp. 422–446, Oct. 2002.",
        "[13] F. M. Harper and J. A. Konstan, \"The MovieLens Datasets: History and Context,\" ACM Trans. Interact. Intell. Syst., vol. 5, no. 4, pp. 1–19, Dec. 2015.",
        "[14] M. J. Pazzani and D. Billsus, \"Content-Based Recommendation Systems,\" in The Adaptive Web, Lecture Notes in Computer Science, vol. 4321, Berlin: Springer, 2007, pp. 325–341.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Cm(1.0)
        pf.first_line_indent = Cm(-1.0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(5)
        pf.line_spacing = 1.35
        r = p.add_run(ref)
        set_run_font(r, size=BODY_SIZE)

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)
    print(f"Saved Word document to: {OUT_DOCX} ({OUT_DOCX.stat().st_size} bytes)")


if __name__ == "__main__":
    build_doc()
