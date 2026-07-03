#!/usr/bin/env python3
"""Run 2e: abstract + highlights pass — keep ranking inversion as the confident
headline, frame §4.8 sensitivity as subordinate/supporting, nuance the
C1-challenged 'principled mission-selection parameter' highlight."""
from __future__ import annotations
from docx import Document

DOC = "revision/FLARE_Drones_REVISED.docx"


def replace_in_paragraph(p, old, new):
    full = "".join(r.text for r in p.runs)
    if old not in full:
        return False
    idx = full.index(old); end = idx + len(old); pos = 0; started = False
    for r in p.runs:
        rlen = len(r.text); rstart, rend = pos, pos + rlen; pos = rend
        if rend <= idx or rstart >= end:
            continue
        pre = r.text[:idx - rstart] if rstart < idx else ""
        suf = r.text[end - rstart:] if rend > end else ""
        r.text = (pre + new + suf) if not started else (pre + suf)
        started = True
    return True


def edit(doc, old, new, label):
    for p in doc.paragraphs:
        if replace_in_paragraph(p, old, new):
            print(f"[OK] {label}"); return True
    print(f"[MISS] {label}"); return False


def main():
    doc = Document(DOC)

    # Abstract — subordinate sensitivity clause (frames §4.8 as CONFIRMING the inversion)
    edit(doc,
        "via a mission score function. Empirical results confirm",
        "via a mission score function. A within-planner sensitivity analysis spanning "
        "eight planners and six risk-handling paradigms confirms that the inversion is "
        "driven by the planners' differential response to the risk coefficient rather "
        "than by their fixed configurations. Empirical results confirm",
        "Abstract: subordinate sensitivity clause")

    # Highlight [8] — nuance 'single ρ governs' → paradigm-dependent trade-off
    edit(doc,
        "A single risk coefficient ρ governs the survival–timeliness trade-off: within the adaptive planner set, higher ρ is associated with higher survival probability at the cost of time-induced mission degradation.",
        "The risk coefficient ρ mediates a survival–timeliness trade-off: higher ρ raises "
        "navigation success but erodes time-decaying mission value, with a strength that "
        "depends on the planning paradigm.",
        "Highlight [8]: paradigm-dependent trade-off")

    # Highlight [11] — reframe the C1-challenged 'principled mission-selection parameter'
    edit(doc,
        "Risk coefficient serves as a principled mission-selection parameter, enabling deployment-specific planner configuration.",
        "The risk coefficient is a meaningful parameter for mission-aware planner selection: "
        "a within-planner sensitivity analysis shows its effect on mission value is strong for "
        "some risk-handling paradigms and negligible for others, informing deployment-specific "
        "configuration.",
        "Highlight [11]: C1-challenged claim reframed")

    doc.save(DOC)
    print("\nSaved:", DOC)


if __name__ == "__main__":
    main()
