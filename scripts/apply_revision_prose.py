#!/usr/bin/env python3
"""Run 1 of DOCX revision: safe prose edits via cross-run text replacement.

Operates on revision/FLARE_Drones_REVISED.docx (a copy; original untouched).
Each edit is verified (prints FOUND/MISSING). Does NOT touch numbers, tables,
figures, or references — those are separate staged steps.
"""

from __future__ import annotations
import sys
from docx import Document

DOC = "revision/FLARE_Drones_REVISED.docx"


def replace_in_paragraph(p, old, new):
    full = "".join(r.text for r in p.runs)
    if old not in full:
        return False
    idx = full.index(old)
    end = idx + len(old)
    pos = 0
    started = False
    for r in p.runs:
        rlen = len(r.text)
        rstart, rend = pos, pos + rlen
        pos = rend
        if rend <= idx or rstart >= end:
            continue
        pre = r.text[:idx - rstart] if rstart < idx else ""
        suf = r.text[end - rstart:] if rend > end else ""
        if not started:
            r.text = pre + new + suf
            started = True
        else:
            r.text = pre + suf
    return True


def edit(doc, old, new, label, tables=True):
    """Find old across all paragraphs (and optionally table cells) and replace."""
    for p in doc.paragraphs:
        if replace_in_paragraph(p, old, new):
            print(f"[OK] {label}")
            return True
    if tables:
        for t in doc.tables:
            for row in t.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        if replace_in_paragraph(p, old, new):
                            print(f"[OK] {label} (table cell)")
                            return True
    print(f"[MISS] {label} — old text not found:\n       {old[:90]}")
    return False


def main():
    doc = Document(DOC)

    # C3 — novelty (§1)
    edit(doc,
         "According to our knowledge, no existing works evaluate the mission-level consequences of risk sensitivity—an omission that the present study addresses.",
         "Prior work in UAV disaster response, emergency medical logistics and search-and-rescue does evaluate mission-oriented outcomes; to our knowledge, however, these studies do not couple a risk-parameterised planner configuration to a strictly time-decreasing mission-value function within a deterministic, reproducible wildfire benchmark. FLARE's specific contribution is to show, in this controlled setting, that navigation-success and mission-effectiveness rankings can diverge.",
         "C3 novelty softened")

    # C4 — Abstract grounding
    edit(doc,
         "grounded in real-world wildfire scenarios inspired by recent Greek wildfire events",
         "inspired by recent Greek wildfire events; all hazard dynamics (fire spread, structural collapse, moving obstacles, dynamic no-fly zones) are modeled benchmark abstractions rather than incident-specific reconstructions",
         "C4 abstract grounding")

    # C4 — Table 2 caption
    edit(doc,
         "Three OSM-based scenarios grounded in real Greek wildfires.",
         "Three OSM-based scenarios inspired by recent Greek wildfires; hazard dynamics are modeled abstractions.",
         "C4 Table 2 caption")

    # C4 — 'Real incident' -> 'Inspiring incident' (table label)
    edit(doc, "Real incident", "Inspiring incident", "C4 Table 2 row label")

    # C4 — maritime/road obstacle operational meaning (§3.2)
    edit(doc,
         "blocking UAV movement within a Manhattan (L1, taxicab) buffer of configurable radius.",
         "blocking UAV movement within a Manhattan (L1, taxicab) buffer of configurable radius. For an airborne UAV these model low-altitude corridor stand-off constraints—exclusion buffers over moving emergency vehicles and vessels under which low-flying UAVs must not operate.",
         "C4 maritime/road stand-off rationale")

    # C5 + R2.5 — §5.3(e) rephrase + remove over-broad 'no prior benchmark'
    edit(doc,
         "The ranking inversion was not predictable a priori, as no prior benchmark incorporates time-coupled mission scoring, and the magnitude (|ΔRank|=3) required empirical measurement.",
         "Because conventional, navigation-centric benchmarks do not score the time-dependent value of mission completion, the ranking inversion is not visible under standard evaluation protocols; FLARE surfaces it by coupling risk-parameterised planning to a time-decaying mission value.",
         "R2.5 §5.3(e) rephrase")

    # C5 — soften 'persists across all three' (§5.3)
    edit(doc,
         "the inversion persists across all three geographically distinct scenarios",
         "the inversion is a dominant, aggregate phenomenon rather than a universal per-scenario law",
         "C5 §5.3 soften 'all three'")

    doc.save(DOC)
    print("\nSaved:", DOC)


if __name__ == "__main__":
    main()
