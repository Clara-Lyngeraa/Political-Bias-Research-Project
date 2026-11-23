import argparse
import json
import random
import re
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from xml.sax.saxutils import escape
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)

PERSONA_KEYS = [ "age", "gender", "nation", "political_orientation" ]

def extract_answer(record: dict):
    """Return (selection, response_text, json_ok) from a record's `answer`.

    Tries a normal JSON parse first; if the model produced malformed JSON
    (raw newlines, unescaped quotes, ...) falls back to regexes.
    """
    raw = record.get("answer") or ""

    try:
        parsed = json.loads(raw, strict=False)
        selection, text = parsed.get("selection"), parsed.get("response", "")
        # open mode: the model's answer has no "selection"; the classifier
        # stored it as a top-level key, which takes precedence
        if "selection" in record:
            selection = record["selection"]
        return selection, text, True
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    sel = re.search(r'"selection"\s*:\s*"([^"]*)"', raw)
    selection = record["selection"] if "selection" in record else (sel.group(1) if sel else None)

    resp = re.search(r'"response"\s*:\s*(.*)$', raw, re.DOTALL)
    text = resp.group(1) if resp else raw
    text = text.strip()
    text = re.sub(r'"?\s*\}\s*$', "", text)  # trailing   " }
    text = text.lstrip('"').strip()
    text = text.replace("\\'", "'").replace('\\"', '"')
    return selection, text, False


def load_records(path: Path) :
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rec["_line_no"] = line_no
            records.append(rec)
    return records

def register_fonts():
    candidates = [
        # (regular, bold, italic)
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        ("/Library/Fonts/Arial Unicode.ttf", None, None),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "C:/Windows/Fonts/ariali.ttf"),
    ]
    for reg, bold, ital in candidates:
        if Path(reg).exists():
            pdfmetrics.registerFont(TTFont("Body", reg))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold if bold and Path(bold).exists() else reg))
            pdfmetrics.registerFont(TTFont("Body-Italic", ital if ital and Path(ital).exists() else reg))
            return "Body", "Body-Bold", "Body-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"  # built-in fallback

def build_pdf(samples, out_path: Path, seed: int, source_name: str, start: int = 1):
    regular, bold, italic = register_fonts()
    
    s_head = ParagraphStyle("head", fontName=bold, fontSize=11, leading=14)
    s_meta = ParagraphStyle("meta", fontName=regular, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
    s_q = ParagraphStyle("q", fontName=bold, fontSize=10, leading=13, spaceBefore=3)
    s_opt = ParagraphStyle("opt", fontName=regular, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
    s_sel = ParagraphStyle("sel", fontName=bold, fontSize=10, leading=13, spaceBefore=4)
    s_resp = ParagraphStyle("resp", fontName=regular, fontSize=9.5, leading=12.5, alignment=TA_LEFT, spaceBefore=3)
    s_check = ParagraphStyle("chk", fontName=regular, fontSize=9.5, leading=12)
    s_title = ParagraphStyle("title", fontName=bold, fontSize=16, leading=20)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(15 * mm, 8 * mm, f"{source_name}  |  seed {seed}")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
        title="Review samples",
    )

    story = [
        Paragraph("Review samples", s_title),
        Paragraph(f"{len(samples)} random answers from {escape(source_name)} (seed {seed})", s_meta),
        Spacer(1, 6),
    ]

    for i, rec in enumerate(samples, start=start):
        selection, text, json_ok = rec["_parsed"]
        persona = [f"{k}: {rec[k]}" for k in PERSONA_KEYS if rec.get(k)]
        persona_str = " | ".join(persona) if persona else "no persona (base case)"
        flag = "" if json_ok else "   <font color='#B00020'>[malformed JSON]</font>"

        head = Paragraph(
            f"#{i} <font name='{regular}' size='8.5' color='#555555'>"
            f"&nbsp;&nbsp;prompt_index {rec.get('prompt_index')} · line {rec['_line_no']}</font>{flag}",
            s_head,
        )

        meta = Paragraph(escape(persona_str), s_meta)
        question = Paragraph(escape(rec.get("question", "")), s_q)
        options = Paragraph("Options: " + escape(" / ".join(rec.get("options", []))), s_opt)
        sel = Paragraph(f"Extracted selection: <u>{escape(str(selection))}</u>", s_sel)
        resp = Paragraph(escape(text).replace("\n", "<br/>"), s_resp)

        checks = Table(
            [[Paragraph("[&nbsp;&nbsp;&nbsp;] selection matches text &nbsp;&nbsp;&nbsp; "
                        "[&nbsp;&nbsp;&nbsp;] wrong &nbsp;&nbsp;&nbsp; [&nbsp;&nbsp;&nbsp;] unclear", s_check),
              Paragraph("Notes:", s_check)]],
            colWidths=[95 * mm, None],
        )
        checks.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (1, 0), (1, 0), 0.4, colors.HexColor("#999999")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))

        story.append(KeepTogether([
            head, meta, question, options, sel, resp, checks,
            HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#BBBBBB"),
                       spaceBefore=4, spaceAfter=8),
        ]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="path to responses.jsonl")
    parser.add_argument("-n", "--n", type=int, default=200, help="number of samples (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, for reproducibility")
    parser.add_argument("-o", "--output", type=Path, default=Path("review_samples.pdf"))
    parser.add_argument("--parts", type=int, default=1, help="split the sample into this many parts")
    parser.add_argument("--part", type=int, default=1, help="which part to output (1-based)")
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise SystemExit("No valid JSON lines found.")

    n = min(args.n, len(records))
    samples = random.Random(args.seed).sample(records, n)
    for rec in samples:
        rec["_parsed"] = extract_answer(rec)

    size = n // args.parts
    lo = (args.part - 1) * size
    hi = n if args.part == args.parts else lo + size
    part_samples = samples[lo:hi]

    build_pdf(part_samples, args.output, args.seed, args.input.name, start=lo + 1)
    print(f"Wrote {len(part_samples)} samples (#{lo + 1}-#{hi}) to {args.output}")  

if __name__ == "__main__":
    main()
