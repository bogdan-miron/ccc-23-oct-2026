"""Clean text out of a decorated contest PDF (slide-style, 2 columns, mascots, boilerplate).
Usage: ./run.sh tools/pdf2spec.py <level.pdf> [-o notes/<challenge>/level-N/pdf.md] [--pages] [--ocr]
Prints cleaned text in reading order + KEY LINES (metrics, thresholds, formats, filenames).
Never drops story text silently: the story can hide clues (e.g. species descriptions, regime-change hints).
Only repeated boilerplate (footers, 'continue', 'submit', banner) is removed."""
import argparse, re, subprocess, sys, collections, tempfile, os
from pathlib import Path
import pymupdf

BOILER = [r"^#?WeLoveSoftware$", r"^u too\?.*", r"^continue$", r"^submit$", r"^good luck!?$", r"^-?\s*recap\s*-?$", r"^LEVEL\s*\d+$",
          r".*cloudflight\.io/career.*", r"^https?://\S+$"]
BULLETS = {"•", "-", "●", "▪", "◦", "–", "*"}

def page_blocks(page):
    W = page.rect.width
    bl = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    items = [dict(x0=b[0], y0=b[1], x1=b[2], y1=b[3], t=b[4].strip("\n")) for b in bl]
    # attach bullet-only blocks to the nearest text block on the same line
    bullets = [i for i in items if i["t"].strip() in BULLETS]
    for bu in bullets:
        cand = [i for i in items if i is not bu and i["t"].strip() not in BULLETS and abs(i["y0"] - bu["y0"]) < 12 and i["x0"] >= bu["x0"]]
        if cand:
            min(cand, key=lambda i: (abs(i["y0"] - bu["y0"]), i["x0"] - bu["x0"]))["t"] = "- " + min(cand, key=lambda i: (abs(i["y0"] - bu["y0"]), i["x0"] - bu["x0"]))["t"]
        items.remove(bu)
    mid = W / 2
    key = lambda i: (round(i["y0"] / 4), i["x0"])
    fl = [i for i in items if i["x1"] < mid + 15]; fr = [i for i in items if i["x0"] > mid - 15]
    if len(fl) >= 2 and len(fr) >= 2:  # genuine two-column slide
        left = [i for i in items if (i["x0"] + i["x1"]) / 2 < mid]
        right = [i for i in items if (i["x0"] + i["x1"]) / 2 >= mid]
        return sorted(left, key=key), sorted(right, key=key)
    return items, []  # single flow: keep the PDF's own block order (speech bubbles stay in reading order)

def clean_lines(text):
    out, pend = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if not s: continue
        if any(re.match(p, s, re.I) for p in BOILER): continue
        if s in BULLETS: pend = True; continue
        s = re.sub(r"[ \t]+", " ", ln.rstrip())
        if pend and not s.lstrip().startswith("- "): s = "- " + s.lstrip()
        pend = False
        out.append(s)
    return out

def ocr_filter(text):
    keep = []
    for ln in text.splitlines():
        words = re.findall(r"[A-Za-z]{2,}", ln)
        alpha = sum(c.isalpha() or c.isspace() for c in ln) / max(len(ln), 1)
        if len([w for w in words if len(w) >= 4]) >= 4 and alpha > 0.75: keep.append(ln)
    return "\n".join(keep)

def ocr_page(page, dpi=150):
    with tempfile.TemporaryDirectory() as d:
        png = os.path.join(d, "p.png"); page.get_pixmap(dpi=dpi).save(png)
        try:
            r = subprocess.run(["tesseract", png, "-", "--psm", "6"], capture_output=True, text=True)
        except FileNotFoundError:
            print("[pdf2spec] tesseract not installed: OCR skipped (view page PNGs instead)", file=sys.stderr); return ""
        return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("-o"); ap.add_argument("--pages", action="store_true", help="also render pages to PNG next to -o")
    ap.add_argument("--ocr", action="store_true", help="force OCR on every page (slow); auto for pages with almost no text layer")
    a = ap.parse_args()
    doc = pymupdf.open(a.pdf)
    pages = []
    for pno, page in enumerate(doc, 1):
        l, r = page_blocks(page)
        txt = "\n".join(i["t"] for i in l + r)
        nchar = len(txt.strip())
        nimg = len(page.get_images())
        note = ""
        if a.ocr or (nchar < 80 and nimg):
            o = ocr_filter(ocr_page(page)); note = " [OCR used]" if o else ""
            txt = (txt + "\n" + o) if nchar else o
        pages.append((pno, clean_lines(txt), note, nchar, nimg))
    # drop lines repeated on >=3 pages (footers/banners), keep first copy nowhere
    cnt = collections.Counter(s for _, ls, *_ in pages for s in set(ls))
    rep = {s for s, c in cnt.items() if c >= 3 and len(s) < 80}
    res = []
    seen_para = set()
    seen_hdr = set()
    for pno, ls, note, nchar, nimg in pages:
        ls2 = []
        for s_ in ls:
            if s_ in rep: continue
            if re.match(r"^Level \d+:", s_):
                if s_ in seen_hdr: continue
                seen_hdr.add(s_)
            ls2.append(s_)
        if not ls2: continue
        res.append(f"--- page {pno}{note} ---")
        for s in ls2:
            res.append(s)
    body = "\n".join(res)
    key = []
    pat = re.compile(r"(rmse|mae|mse|f1|accuracy|auc|r2|log ?loss|precision|recall|score above|below|above|at least|required|[<>]\s?\d|\d\s?%)", re.I)
    for s in body.splitlines():
        if pat.search(s) and re.search(r"\d", s) and not s.startswith(("---", "Level")) and s.count(",") < 2: key.append(s.strip())
    files = sorted(set(re.findall(r"[\w\-]*level[_\-]?\d[\w\-.]*(?:\.\w+)?", body, re.I)))
    fmt = [s.strip() for s in body.splitlines() if re.search(r"(csv|txt|upload|submission|sample|format|separated|sorted|header|column|\.in\b|\.zip)", s, re.I) and not s.startswith("---")]
    low = [p for p, ls, note, nchar, nimg in pages if nchar < 80 and nimg]
    out = ["# PDF TEXT (cleaned, reading order)", body, "", "# KEY LINES (metrics / thresholds)", *(key or ["(none found)"]),
           "", "# FORMAT / FILE LINES", *(fmt or ["(none found)"]), "", "# FILE NAMES MENTIONED", ", ".join(files) or "(none)"]
    if low: out += ["", f"# WARNING: pages with image-only content: {low}. View them (--pages) or rerun with --ocr."]
    text = "\n".join(out)
    if a.o:
        Path(a.o).parent.mkdir(parents=True, exist_ok=True); Path(a.o).write_text(text, encoding="utf-8")
        if a.pages:
            pd_ = Path(a.o).parent / "pages"; pd_.mkdir(exist_ok=True)
            for i, page in enumerate(doc, 1): (pd_ / f"p{i}.png").write_bytes(page.get_pixmap(dpi=70).tobytes("png"))  # write_bytes overwrites; pixmap.save() deletes first and fails where deletes are blocked
    print(text)

if __name__ == "__main__":
    main()
