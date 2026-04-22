"""Decode Figure-number-to-image-slot mapping from a docx file."""
import json, re, sys, zipfile
from pathlib import Path

def decode(docx_path: str) -> dict[str, str]:
    z = zipfile.ZipFile(docx_path)
    doc = z.read("word/document.xml").decode("utf-8")
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8")

    # rel id -> image filename
    rid_to_img: dict[str, str] = {}
    for m in re.finditer(r'Id="([^"]+)"[^>]*Target="media/([^"]+)"', rels):
        rid_to_img[m.group(1)] = m.group(2)

    # walk paragraphs; remember most recently seen embedded image blip,
    # attach it to the next "Figure N" caption paragraph
    paras = re.findall(r'<w:p\b[^>]*>.*?</w:p>', doc, re.S)
    fig_to_slot: dict[str, str] = {}
    pending_blip: str | None = None
    for p in paras:
        for m in re.finditer(r'r:embed="([^"]+)"', p):
            pending_blip = rid_to_img.get(m.group(1), pending_blip)
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
        fig_m = re.match(r"\s*Figure\s+(\d+)\b", text)
        if fig_m and pending_blip:
            fig_to_slot[f"Figure {fig_m.group(1)}"] = pending_blip
            pending_blip = None
    return fig_to_slot


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/v8_final/figure_slot_map.json"
    mapping = decode(src)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(json.dumps(mapping, indent=2, ensure_ascii=False))
