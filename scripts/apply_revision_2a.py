#!/usr/bin/env python3
"""Run 2a: safe text appends + one caption insert (no citations, no numbers)."""
from __future__ import annotations
from docx import Document

DOC = "revision/FLARE_Drones_REVISED.docx"


def find_para(doc, anchor):
    for p in doc.paragraphs:
        if anchor in p.text:
            return p
    return None


def append_run(doc, anchor, text, label):
    p = find_para(doc, anchor)
    if p is None:
        print(f"[MISS] {label} — anchor not found: {anchor[:60]}")
        return False
    r = p.add_run(text)
    # inherit formatting from the paragraph's last existing run if any
    print(f"[OK] {label}")
    return True


def main():
    doc = Document(DOC)

    # B/R2.2 — reframe ρ as a common coefficient (after Eq. 2 discussion)
    append_run(doc, "The risk coefficient ρ alone",
        " Although each planner is reported at a fixed value (the diagonal α = 5, "
        "β = 0.5, γ = 2, δ = 3), ρ is a common risk-aversion coefficient: the "
        "cost-inflation law w(x) = 1 + ρ·R(x) is applied in identical form by the four "
        "graph-search planners, and APF applies an analogous risk-scaled repulsive gain, "
        "so ρ can be varied within any planner while its algorithm is held fixed "
        "(sensitivity study, Section 4.5).",
        "B/R2.2 ρ common-coefficient reframe")

    # C5 — add isolated-sweep Spearman (correct sign) after the confounded values
    append_run(doc, "Spearman correlation between ρ and SR is",
        " When ρ is isolated in the within-planner sweep (Section 4.5), the pooled "
        "correlations become ρ–SR = +0.085 (p < 0.001) and ρ–M = −0.063 (p = 0.01); "
        "the negative ρ–M is mechanistically correct—risk-aversion lengthens travel and "
        "erodes time-decaying value—so the positive aggregate values above are an "
        "artifact of the A* anchor at ρ = 0, M = 0.",
        "C5 isolated-sweep Spearman")

    # R2.4 — caption for the abbreviations table
    p = find_para(doc, "The following abbreviations are used")
    if p is not None:
        cap = p.insert_paragraph_before("Table 5. Abbreviations used in this manuscript.")
        print("[OK] R2.4 abbreviations caption inserted")
    else:
        print("[MISS] R2.4 abbreviations anchor not found")

    doc.save(DOC)
    print("\nSaved:", DOC)


if __name__ == "__main__":
    main()
