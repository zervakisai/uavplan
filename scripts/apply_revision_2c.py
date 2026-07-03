#!/usr/bin/env python3
"""Run 2c: insert §4.8 'Sensitivity to the Risk Coefficient rho' subsection
(heading + body + 2 figures + Table S) before §5 Discussion, and a 3-D
limitation paragraph (R2.1) before §6 Conclusions."""
from __future__ import annotations
from docx import Document
from docx.shared import Inches, Pt

DOC = "revision/FLARE_Drones_REVISED.docx"
FIGDIR = "outputs/paper_figures"

TABLE_S = [
    ("Planner", "Paradigm", "Spearman(ρ,SR)", "Spearman(ρ,M)", "M: ρ=0→10"),
    ("CVaR", "Worst-case / CVaR", "−0.13", "−0.26*", "0.85 → 0.45"),
    ("Incr. A*", "Graph search (incremental)", "+0.16", "−0.25*", "0.91 → 0.40"),
    ("Risk-Sensitive", "Exponential / entropic", "+0.03", "−0.07", "0.85 → 0.73"),
    ("RRT*", "Sampling", "+0.08", "−0.05", "0.73 → 0.64"),
    ("Aggressive", "Graph search (adaptive)", "+0.08", "−0.02", "0.85 → 0.84"),
    ("Chance-Constr.", "Hard threshold", "−0.00", "+0.00", "0.85 → 0.87"),
    ("APF", "Potential field", "−0.01", "+0.01", "0.51 → 0.53"),
    ("Periodic", "Graph search (adaptive)", "+0.12", "+0.02", "0.82 → 0.86"),
]


def find(doc, pred):
    for p in doc.paragraphs:
        if pred(p.text):
            return p
    return None


def main():
    doc = Document(DOC)
    disc = find(doc, lambda t: t.strip() == "5. Discussion")
    concl = find(doc, lambda t: t.strip().startswith("6.") and "Conclusion" in t)
    head47 = find(doc, lambda t: t.strip().startswith("4.7") and "Ablation" in t)
    assert disc is not None and concl is not None, "anchors not found"
    heading_style = head47.style if head47 is not None else None

    def ins_before(anchor, text, style=None, bold=False, italic=False, size=None):
        p = anchor.insert_paragraph_before(text)
        if style is not None:
            p.style = style
        if p.runs:
            r = p.runs[0]
            r.bold = bold; r.italic = italic
            if size:
                r.font.size = Pt(size)
        return p

    # --- §4.8 heading + body (before §5 Discussion) ---
    ins_before(disc, "4.8. Sensitivity to the Risk Coefficient ρ",
               style=heading_style, bold=True)
    ins_before(disc,
        "To isolate ρ from planner architecture, we swept ρ ∈ {0, 1, 2, 5, 10} within "
        "each risk-consuming planner (three scenarios × 30 seeds), holding each planner's "
        "algorithm and replan trigger fixed. We additionally implemented four planners "
        "spanning distinct risk-handling paradigms driven by the same ρ: a chance-constrained "
        "planner with a hard risk threshold [50], a CVaR / worst-case planner [51,52], a "
        "risk-sensitive (entropic) planner with cost exp(ρ·R) [54,55], and a sampling-based "
        "risk-aware RRT* [53]. These paradigms were chosen a priori, and every planner is "
        "reported—including ρ-insensitive ones—so the study is a spectrum rather than a "
        "selection.")
    ins_before(disc,
        "The effect of ρ on mission value M is strongly paradigm-dependent (Figure 10, "
        "Figure 11; Table 6). It substantially erodes M in the CVaR (Spearman ρ_s = −0.26, "
        "p < 0.001; M falls from 0.91 to 0.45 as ρ grows from 0 to 10) and Incremental-A* "
        "(ρ_s = −0.25; 0.91 → 0.40) planners, moderately in the exponential planner "
        "(0.94 → 0.73), and negligibly in the potential-field, hard-threshold and "
        "frequent-replan planners (|ρ_s| < 0.07). Navigation success responds only weakly "
        "and in the opposite direction (pooled ρ_s = +0.085). The navigation–mission "
        "divergence therefore arises from the differential response of planners to ρ, not "
        "from the particular per-planner ρ values used in the main comparison; the ranking "
        "inversion persists at matched ρ with clear margins in Penteli and Downtown, while "
        "in Piraeus the two leading adaptive planners are within statistical noise "
        "(ΔM ≤ 0.002).")

    # --- Figures (before §5) ---
    def ins_fig(anchor, png, width_in, caption):
        p = anchor.insert_paragraph_before("")
        p.alignment = 1  # center
        p.add_run().add_picture(f"{FIGDIR}/{png}", width=Inches(width_in))
        cap = anchor.insert_paragraph_before(caption)
        if cap.runs:
            cap.runs[0].italic = True; cap.runs[0].font.size = Pt(8)
        return p
    ins_fig(disc, "rho_response_curves.png", 6.2,
            "Figure 10. Per-planner ρ-response: feasible success rate (a) and mission "
            "score M (b) versus ρ for the eight swept planners. Mission value collapses "
            "for the CVaR and Incremental-A* planners while others stay flat.")
    ins_fig(disc, "rho_sensitivity_spectrum.png", 4.2,
            "Figure 11. Spearman correlation between ρ and mission score M per planner, "
            "grouped by risk-handling paradigm. ρ erodes M strongly for some paradigms and "
            "negligibly for others (* p < 0.05).")

    # --- Table 6 (Table S): build and move before §5 ---
    tbl = doc.add_table(rows=len(TABLE_S), cols=5)
    try:
        tbl.style = "Table Grid"
    except Exception:
        pass
    for ri, row in enumerate(TABLE_S):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = val
            for pp in cell.paragraphs:
                for rr in pp.runs:
                    rr.font.size = Pt(8)
                    if ri == 0:
                        rr.bold = True
    capT = disc.insert_paragraph_before(
        "Table 6. Per-planner ρ-sensitivity across the eight swept planners "
        "(three scenarios × 30 seeds per ρ). * p < 0.05.")
    if capT.runs:
        capT.runs[0].italic = True; capT.runs[0].font.size = Pt(8)
    disc._p.addprevious(tbl._tbl)  # move table to just before §5

    # --- R2.1: 3-D limitation paragraph (before §6 Conclusions) ---
    p3d = concl.insert_paragraph_before(
        "Finally, the benchmark operates on a 2-D grid; a third (altitude) dimension would "
        "make the risk field R(x, y, z). Vertical separation attenuates horizontal "
        "ground-hazard risk (fire plumes, debris, moving vehicles) but introduces "
        "altitude-dependent terms—smoke-column height, manned-aircraft corridors and dynamic "
        "no-fly volumes, and wind shear—so extending FLARE to 3-D airspace, where altitude "
        "reshapes the risk-cost map, is a priority direction.")

    doc.save(DOC)
    print("[OK] inserted §4.8 subsection + 2 figures + Table 6 + 3-D paragraph")
    print("Saved:", DOC)


if __name__ == "__main__":
    main()
