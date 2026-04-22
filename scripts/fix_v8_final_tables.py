"""Correct table numbering in UAV_v8_final.docx.

The initial build assigned Table 3 to both the new parameter summary and the
scenario overview. This patch restores the intended sequence:
  Table 2 = Simulation parameter summary (§3.7)
  Table 3 = Figure-to-scenario traceability (§4.1)
  Table 4 = Scenario overview (§4.2)
  Table 5 = Mission scorecard (§4.4)
  Table 6 = Statistical validation (§4.8)

Operates on paragraph text: for each paragraph that contains one of the target
strings, the paragraph is rebuilt with a single run carrying the corrected
text (preserving only paragraph properties).
"""
from __future__ import annotations
import re
import shutil
import zipfile
from pathlib import Path

SRC = "UAV_v8_final.docx"
DST = "UAV_v8_final.docx"  # in-place patch; caller should back up if needed

PARAGRAPH_FIXES: list[tuple[str, str]] = [
    ("Table 3. Simulation parameter summary", "Table 2. Simulation parameter summary"),
    ("Table 3. Scenario overview",            "Table 4. Scenario overview"),
    ("In Table 3, Service time",              "In Table 4, Service time"),
    ("Table 4 disaggregates by scenario",     "Table 5 disaggregates by scenario"),
    ("Table 4. Mission scorecard",            "Table 5. Mission scorecard"),
    ("Table 5. Statistical validation",       "Table 6. Statistical validation"),
]


def paragraph_joined_text(p_xml: str) -> str:
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p_xml))


def rebuild_paragraph(p_xml: str, new_text: str) -> str:
    # Keep <w:pPr> if present; drop all runs; add one run with new_text.
    ppr_match = re.search(r"<w:pPr>.*?</w:pPr>", p_xml, re.S)
    ppr = ppr_match.group(0) if ppr_match else ""
    opening = re.match(r"<w:p\b[^>]*>", p_xml).group(0)
    closing = "</w:p>"
    run = (
        f'<w:r><w:t xml:space="preserve">{new_text}</w:t></w:r>'
    )
    return f"{opening}{ppr}{run}{closing}"


def patch_document(xml: str) -> tuple[str, list[tuple[int, str, str]]]:
    paras = list(re.finditer(r"<w:p\b[^>]*>.*?</w:p>", xml, re.S))
    applied: list[tuple[int, str, str]] = []
    # Walk back-to-front so earlier offsets remain valid
    new_xml = xml
    for idx, m in reversed(list(enumerate(paras))):
        p_xml = m.group(0)
        text = paragraph_joined_text(p_xml)
        if not text.strip():
            continue
        changed = text
        for bad, good in PARAGRAPH_FIXES:
            if bad in changed:
                changed = changed.replace(bad, good)
        if changed != text:
            applied.append((idx, text[:80], changed[:80]))
            new_p = rebuild_paragraph(p_xml, changed)
            new_xml = new_xml[: m.start()] + new_p + new_xml[m.end():]
    return new_xml, list(reversed(applied))


def main() -> None:
    if not Path(SRC).exists():
        raise SystemExit(f"missing source: {SRC}")
    # Back up original
    backup = Path(SRC).with_suffix(".pre_tablefix.docx")
    if not backup.exists():
        shutil.copy2(SRC, backup)
        print(f"[info] backup -> {backup}")
    with zipfile.ZipFile(SRC) as z_in:
        names = z_in.namelist()
        members: dict[str, bytes] = {n: z_in.read(n) for n in names}
    xml = members["word/document.xml"].decode("utf-8")
    new_xml, applied = patch_document(xml)
    if not applied:
        print("[warn] no paragraphs matched any fix; aborting without write")
        return
    members["word/document.xml"] = new_xml.encode("utf-8")
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z_out:
        for n in names:
            z_out.writestr(n, members[n])
    print(f"[done] wrote {DST}; applied {len(applied)} paragraph fix(es):")
    for idx, before, after in applied:
        print(f"  p{idx:4d}  '{before}...' -> '{after}...'")


if __name__ == "__main__":
    main()
