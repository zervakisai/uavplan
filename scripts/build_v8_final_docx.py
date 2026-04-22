"""Build UAV_v8_final.docx from UAV_v8_clean.docx.

Applies:
  WS1 -- regex text fixes (section renumbering, word-level artifacts, Figure 18 -> A1).
  WS2 -- new Table 2 (simulation parameter summary) in new subsection 3.7.
  WS3 -- new Table 3 (figure-to-scenario-to-contract traceability) in 4.1.
  WS4 -- expanded 4.3 Experimental Setup (five paragraphs).
  WS5 -- new 4.6 Event Correlation Analysis.
  WS6 -- "Source:" caption suffixes on all figure captions.
  WS7 -- new 4.5.1 Per-Planner and Per-Scenario Mechanism.
  Figure 8 -- new media, new relationship, inline drawing + caption near end of 4.

Acceptance criteria are validated in validate().
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path("/Users/konstantinos/Dev/planning/uavbench")
SRC = ROOT / "UAV_v8_clean.docx"
DST = ROOT / "UAV_v8_final.docx"
PNG_DIR = ROOT / "outputs/v8_final/png_300dpi"
SLOT_MAP_PATH = ROOT / "outputs/v8_final/figure_slot_map.json"

# ---------------------------------------------------------------------------
# WS1 -- regex fixes
# ---------------------------------------------------------------------------
TEXT_FIXES = [
    (r"(?<![0-9])21\.\s*Background and Related Work", "2. Background and Related Work"),
    (r"(?<![0-9])32\.\s*Materials and Methods", "3. Materials and Methods"),
    (r"(?<![0-9])43\.\s*Simulation and Results", "4. Simulation and Results"),
    (r"(?<![0-9])43\.1\.", "4.1."),
    (r"(?<![0-9])43\.2\.", "4.2."),
    (r"(?<![0-9])43\.3\.", "4.3."),
    (r"(?<![0-9])43\.4\.", "4.4."),
    (r"(?<![0-9])32\.6\.", "3.6."),
    (r"3\.6\.\s*Qualitative Evidence", "4.7. Qualitative Evidence"),
    (r"3\.7\.\s*Statistical Validation", "4.8. Statistical Validation"),
    (r"3\.8\.\s*Hazard Ablation", "4.9. Hazard Ablation"),
    (r"(?<![0-9.])4\.\s*Discussion", "5. Discussion"),
    (r"(?<![0-9.])4\.1\.\s*Shapley Attribution", "5.1. Shapley Attribution"),
    (r"(?<![0-9.])4\.2\.\s*Risk Coefficient", "5.2. Risk Coefficient"),
    (r"(?<![0-9.])4\.3\.\s*Limitations", "5.3. Limitations"),
    (r"(?<![0-9.])5\.\s*Conclusions", "6. Conclusions"),
    (r"Figure\s*18\.", "Figure A1."),
    (r"Figure\s*18\s+", "Figure A1 "),
    (r"wildfire eventsscenarios", "wildfire scenarios"),
    (r"studyresearch", "study"),
    (r"selecting cases", "representative scenarios"),
]

# ---------------------------------------------------------------------------
# WS2 -- parameter table
# ---------------------------------------------------------------------------
PARAM_HEADERS = ["Symbol", "Name", "Value", "Unit", "Protocol", "First use"]
PARAM_ROWS = [
    ["c_t", "Traffic severity weight", "0.6", "—", "Design", "Eq. (1)"],
    ["c_b", "Debris severity weight", "0.7", "—", "Design", "Eq. (1)"],
    ["r_f", "Fire decay radius", "3", "cells", "Calibrated (CC-2)", "§3.3"],
    ["r_t", "Traffic decay radius", "2", "cells", "Calibrated (CC-2)", "§3.3"],
    ["r_b", "Debris decay radius", "2", "cells", "Calibrated (CC-2)", "§3.3"],
    ["kappa", "Fire-coupling constant", "5.0", "—", "Design", "Eq. (3)"],
    ["lambda_0", "Survival base decay", "1/600", "1/step", "Design", "Eq. (3)"],
    ["tau_c", "Collapse threshold", "80", "steps", "Calibrated (CC-2)", "§3.2"],
    ["p_d", "Debris spawn probability", "0.6", "—", "Calibrated (CC-2)", "§3.2"],
    ["T_max", "Mission horizon", "900", "steps", "Design", "§3.5"],
    ["|A|", "Action set size", "5", "—", "Design (4 cardinal + idle)", "§3.1"],
    ["rho_A*", "A* risk coefficient", "0", "—", "Design", "§3.4"],
    ["rho_Periodic", "Periodic risk coefficient", "5.0", "—", "Design", "§3.4"],
    ["rho_Aggr", "Aggressive risk coefficient", "0.5", "—", "Design", "§3.4"],
    ["rho_IncA*", "Incremental A* risk coefficient", "2.0", "—", "Design", "§3.4"],
    ["rho_APF", "APF risk coefficient", "3.0", "—", "Design", "§3.4"],
    ["N_seeds", "Seed budget per scenario", "150", "—", "Protocol (§4.3)", "§4.3"],
]

PARAM_HEADING = "3.7. Simulation Parameter Summary"
PARAM_LEADIN = (
    "Table 2 consolidates the simulation parameters calibrated or chosen for the present "
    "study. The Protocol column distinguishes values set by design, values calibrated under "
    "the CC-2 feasibility protocol, and values sourced from protocol constants in §4.3."
)
PARAM_CAPTION = "Table 2. Simulation parameter summary. See §3.1–§3.5 and §4.3 for derivations."

# ---------------------------------------------------------------------------
# WS3 -- traceability table (becomes Table 3 after renumbering)
# ---------------------------------------------------------------------------
TRACE_HEADERS = ["Fig", "Generator", "Scenario", "Seed", "Step", "Contract", "Method ref."]
TRACE_ROWS = [
    ["A1", "scripts/gen_scenario_overview.py", "all three", "—", "—", "—", "§3"],
    ["1", "scripts/gen_wind_fire_figure.py", "Penteli-Evia", "42", "150", "FD-5b, WD-1", "§3.2, Eq. (1)"],
    ["2", "scripts/gen_collapse_cascade_figure.py", "Penteli-Evia", "42", "0–250", "Collapse", "§3.2"],
    ["3", "scripts/gen_traffic_dynamics_figure.py", "Piraeus-Rhodes", "42", "30/80/150", "FC-1, EC-1", "§3.2"],
    ["4", "scripts/gen_risk_perception_figure.py", "Penteli-Evia", "42", "150", "MP-1", "§3.3, Eq. (2)"],
    ["5", "scripts/gen_mission_type_figures.py", "all three", "—", "—", "MC-1..4", "§3.5, Eq. (3)–(6)"],
    ["6", "scripts/analyze_paper_results.py", "all three", "0–149", "—", "—", "Table 1"],
    ["7", "scripts/gen_mission_type_figures.py", "all three", "—", "—", "PC-2, PC-3", "§3.4"],
    ["8", "scripts/gen_mission_type_figures.py", "all three", "—", "—", "MC-1..4", "§3.5"],
]
TRACE_LEADIN = (
    "Table 3 provides figure-level traceability for all paper figures. Each row specifies the "
    "generator script, the scenario YAML, the seed, the simulation step(s), the contract family "
    "exercised, and the methodology subsection or equation illustrated."
)
TRACE_CAPTION = (
    "Table 3. Figure-to-scenario-to-contract traceability. All scripts are version-controlled "
    "under /scripts/; contracts are enumerated in docs/CONTRACTS.md."
)

# ---------------------------------------------------------------------------
# WS4 -- expanded §4.3 (paragraphs after the heading)
# ---------------------------------------------------------------------------
SETUP_PARAS = [
    "The simulation environment is instantiated on a 500x500 grid (N=499) with a cell resolution of "
    "approximately 3 m, extracted from OpenStreetMap tiles of three Greek areas. The Gymnasium-compatible "
    "environment exposes |A|=5 discrete actions (four cardinal moves plus idle) and advances one simulation "
    "step per environment transition. Each episode runs for up to T_max=900 steps or until a terminal "
    "condition is reached.",
    "Stochastic components are driven by a single numpy.random.default_rng(seed) instance per episode, "
    "from which six deterministic child streams are spawned via spawn(6) in a fixed order: fire, traffic, "
    "restriction zones, wind, collapse, and triage. This hierarchy guarantees bit-identical replay "
    "(contract DC-1) under the same (scenario_id, planner_id, seed) triple.",
    "Three scenario configurations drive the experiments: osm_penteli_pharma_delivery_medium.yaml, "
    "osm_piraeus_urban_rescue_medium.yaml, and osm_downtown_fire_surveillance_medium.yaml. Each is "
    "evaluated across 150 seeds (0-149) and the five planners, yielding 450 independent episodes per "
    "planner-scenario pair and 2250 across the full planner suite. Feasible episodes enter the "
    "navigation success rate computation; infeasible episodes are excluded per contract CC-4.",
    "Contract coverage is enforced by 189 automated tests verifying 36 contracts organised into 15 "
    "families: Determinism (DC), Fairness (FC), Events and Decisions (EC), Guardrail (GC), Event "
    "Semantics (EV), Visual Truth (VC), Mission Story (MC), Planner (PC), Fire Dynamics (FD), "
    "Calibration (CC), Sanity Check (SC), Mask Parity (MP), Wind Determinism (WD), Triage (TR), "
    "and Collapse.",
    "Per-action rejection taxonomy: BUILDING, NO_FLY, TRAFFIC_CLOSURE, FIRE, FIRE_BUFFER, SMOKE, "
    "TRAFFIC_BUFFER, DYNAMIC_NFZ, DEBRIS, OUT_OF_BOUNDS. Observed termination reasons: success, "
    "fire_caught, debris_caught, vehicle_collision, goal_stall, infeasible. All event records are "
    "written to outputs/paper_results/all_episodes.csv.",
]

# ---------------------------------------------------------------------------
# WS5 -- Event Correlation (new §4.6)
# ---------------------------------------------------------------------------
EC_HEADING = "4.6. Event Correlation Analysis"
EC_PARAS = [
    "This subsection correlates hazard events with UAV decision outcomes using the 450-episode log. "
    "Of these, 98 episodes terminate with FIRE_CAUGHT and 1 with DEBRIS_CAUGHT, establishing an "
    "on-agent hazard-exposure rate of 22.0%. Additional adverse terminations include 90 vehicle "
    "collisions and 58 goal-stall terminations (the UAV cannot make progress toward its target for a "
    "sustained window).",
    "Fire terminations and the risk coefficient. FIRE_CAUGHT is concentrated in the three mid-rho "
    "planners: Aggressive (30/150), APF (27/150), and Periodic Replan (27/150); Incremental A* adds "
    "14; A* contributes 0 because it terminates earlier via vehicle collisions on the static corridor. "
    "The absence of an A* fire-exposure signal is itself a mechanism finding: when rho=0 the planner "
    "is so committed to its precomputed corridor that it collides with corridor traffic before fire "
    "arrives.",
    "Structural collapse and replanning cascades. Structural collapse fires at tau_c=80 steps of "
    "continuous fire exposure with p_d=0.6. Debris first appears in Penteli at approximately t=85. "
    "Replan counts by planner are consistent with their algorithmic profile: A* never replans (rho=0, "
    "static), while Incremental A* shows the largest replan cascades at the first debris event because "
    "its edge-weight update invalidates large sections of the partial graph. Periodic Replan exhibits "
    "fixed-cadence replans that are, by design, insensitive to collapse timing.",
    "Rejected moves and their mechanism. Over the 450 episodes, 4,071 action rejections are logged. "
    "FIRE_BUFFER dominates (1,653 rejections, 40.6%), followed by FIRE (1,265, 31.1%), TRAFFIC_BUFFER "
    "(769, 18.9%), TRAFFIC_CLOSURE (370, 9.1%), and SMOKE (14, 0.3%). The dominance of fire-layer "
    "rejections confirms that the risk coefficient rho--which inflates costs within the fire-buffer--is "
    "the single most consequential planner parameter: planners with low rho enter the buffer and get "
    "rejected; planners with high rho detour and lose time.",
    "The coupling with mission score. Across 150 Penteli seeds, Aggressive reaches its POI at a mean "
    "executed-step count of 414.7 (successful episodes, N=10) while Periodic reaches it at 522.3 "
    "(N=12). Substituting into the quadratic decay E(t) = 1 - (t / T_max)^2 with T_max = 900 yields "
    "E(414.7) = 0.788 and E(522.3) = 0.663. The 12.5-point gap in mission value is therefore not a "
    "planner-quality difference--both succeed--but a direct consequence of the rho-parameterised "
    "detour cost, which is exactly the ranking-inversion mechanism.",
]

# ---------------------------------------------------------------------------
# WS6 -- caption suffixes
# ---------------------------------------------------------------------------
CAPTION_SUFFIXES = {
    "Figure 1": " Source: scripts/gen_wind_fire_figure.py. See §3.2, Eq. (1).",
    "Figure 2": " Source: scripts/gen_collapse_cascade_figure.py. See §3.2.",
    "Figure 3": " Source: scripts/gen_traffic_dynamics_figure.py. See §3.2.",
    "Figure 4": " Source: scripts/gen_risk_perception_figure.py. See §3.3, Eq. (2).",
    "Figure 5": " Source: scripts/gen_mission_type_figures.py. See §3.5, Eq. (3)-(6).",
    "Figure 6": " Source: scripts/analyze_paper_results.py. See Table 1, §4.5.",
    "Figure 7": " Source: scripts/gen_mission_type_figures.py. See §3.4, Table 3.",
}

# ---------------------------------------------------------------------------
# WS7 -- deeper analysis
# ---------------------------------------------------------------------------
WS7_HEADING = "4.5.1. Per-Planner and Per-Scenario Mechanism"
WS7_PARAS = [
    "Per-planner mechanism. A* fails because the corridor interdiction protocol (FC-1) places fire "
    "ignitions on the A* reference path; with rho=0, A* ignores cost inflation and commits to the "
    "original route until the blocking mask forces a FIRE_BUFFER or BUILDING rejection, by which time "
    "a vehicle collision or fire_caught termination is likely. Aggressive Replan leads on mission "
    "score because rho=0.5 produces only mild cost inflation (w = 1 + 0.5 R), keeping the planner on "
    "near-optimal paths that may traverse within the fire buffer; the resulting short delivery time "
    "(Penteli mean t=414.7) preserves a high mission value (E(t)~0.79). Periodic Replan leads on "
    "success rate but loses on mission score because rho=5.0 inflates edge costs so strongly "
    "(w = 1 + 5 R) that the planner detours around every risk layer, adding roughly 108 steps on the "
    "Penteli scenario (t=522.3 vs 414.7 for Aggressive) and collapsing E(t) to ~0.66.",
    "Per-scenario signatures. Penteli is fire-dominated: wind-driven spread from the two ignition "
    "points creates a moving front that intercepts the A* corridor within 60-90 steps, and this is "
    "reflected in the 98 fire_caught terminations concentrated on Aggressive / APF / Periodic. "
    "Piraeus couples fire with structural collapse and the port's patrol vehicles; the mean delivery "
    "times of Aggressive (475.7) and Periodic (479.8) are nearly identical here because both planners "
    "are forced to detour around debris, compressing the rho advantage. Downtown is dominated by the "
    "NFZ corridors reserved for manned air traffic and extreme building density (0.50); planner "
    "differences compress further because the physical feasibility envelope is so tight that route "
    "options are limited regardless of rho.",
    "Link to Shapley attribution. The per-planner phi vectors reported in §5.1 confirm the "
    "mechanism: fire is the largest negative contributor to A* and APF mission scores, while for "
    "Aggressive the dominant negative contribution in Piraeus is traffic and collapse. The risk "
    "coefficient rho is therefore correctly characterised as a mission-selection parameter: choose "
    "low rho when fire dominates and mission-value time-sensitivity is high; choose high rho when "
    "other hazards dominate and survival is paramount.",
]

# ---------------------------------------------------------------------------
# Figure 8 caption
# ---------------------------------------------------------------------------
FIG8_CAPTION = (
    "Figure 8. FLARE mission portfolio — three scenarios, three decay models, five planners. "
    "Rows depict pharma delivery (Penteli), urban search and rescue (Piraeus), and fire surveillance "
    "(Downtown). Columns show the t=0 setup, a mid-execution schematic (t=T/2), and the mean mission "
    "score per planner aggregated over 150 seeds per scenario. Source: scripts/gen_mission_type_figures.py. "
    "See §3.5, Table 3."
)

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------
WP_RE = re.compile(r"<w:p\b[^>]*>.*?</w:p>", re.S)
WT_RE = re.compile(r"<w:t[^>]*>([^<]*)</w:t>", re.S)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def paragraph(text: str, style: str | None = None, bold: bool = False, italic: bool = False) -> str:
    pPr = ""
    if style:
        pPr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
    rPr = ""
    if bold or italic:
        rPr_inner = ""
        if bold:
            rPr_inner += "<w:b/>"
        if italic:
            rPr_inner += "<w:i/>"
        rPr = f"<w:rPr>{rPr_inner}</w:rPr>"
    return (
        f"<w:p>{pPr}"
        f"<w:r>{rPr}<w:t xml:space=\"preserve\">{_xml_escape(text)}</w:t></w:r>"
        f"</w:p>"
    )


def heading(text: str, level: int) -> str:
    return (
        f"<w:p><w:pPr><w:pStyle w:val=\"heading{level}\"/></w:pPr>"
        f"<w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{_xml_escape(text)}</w:t></w:r>"
        f"</w:p>"
    )


def _tcell(text: str, bold: bool = False) -> str:
    rPr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return (
        "<w:tc>"
        "<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tcBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "</w:tcBorders></w:tcPr>"
        f"<w:p><w:pPr><w:spacing w:before=\"20\" w:after=\"20\"/></w:pPr>"
        f"<w:r>{rPr}<w:t xml:space=\"preserve\">{_xml_escape(text)}</w:t></w:r></w:p>"
        "</w:tc>"
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    n_cols = len(headers)
    grid = "<w:tblGrid>" + "".join("<w:gridCol w:w=\"1200\"/>" for _ in range(n_cols)) + "</w:tblGrid>"
    tbl_pr = (
        "<w:tblPr>"
        "<w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"5000\" w:type=\"pct\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"000000\"/>"
        "</w:tblBorders>"
        "<w:tblLook w:val=\"04A0\"/>"
        "</w:tblPr>"
    )
    header_row = "<w:tr>" + "".join(_tcell(h, bold=True) for h in headers) + "</w:tr>"
    body_rows = "".join(
        "<w:tr>" + "".join(_tcell(str(c)) for c in row) + "</w:tr>" for row in rows
    )
    return f"<w:tbl>{tbl_pr}{grid}{header_row}{body_rows}</w:tbl>"


def paragraphs_xml(texts: list[str]) -> str:
    return "".join(paragraph(t) for t in texts)


# ---------------------------------------------------------------------------
# Paragraph span walker + insertion helpers
# ---------------------------------------------------------------------------
def walk_paragraphs(xml: str) -> list[tuple[int, int, str]]:
    """Return list of (start, end, text) for each <w:p> in order."""
    out: list[tuple[int, int, str]] = []
    for m in WP_RE.finditer(xml):
        text = "".join(WT_RE.findall(m.group(0)))
        out.append((m.start(), m.end(), text))
    return out


def _describe_first_paras(xml: str, n: int = 40) -> str:
    spans = walk_paragraphs(xml)
    lines = []
    for i, (_s, _e, t) in enumerate(spans[:n]):
        lines.append(f"  [{i:3d}] {t[:120]!r}")
    return "\n".join(lines)


def insert_after_paragraph(xml: str, match_regex: str, new_fragment: str) -> str:
    """Find first paragraph whose text matches match_regex; insert new_fragment right after."""
    pat = re.compile(match_regex)
    for start, end, text in walk_paragraphs(xml):
        if pat.search(text):
            return xml[:end] + new_fragment + xml[end:]
    raise SystemExit(
        f"ERROR: marker not found: {match_regex!r}\n"
        f"First 40 paragraph texts:\n{_describe_first_paras(xml)}"
    )


def replace_paragraph_matching(xml: str, match_regex: str, new_fragment: str) -> str:
    """Replace the entire paragraph whose text matches match_regex with new_fragment."""
    pat = re.compile(match_regex)
    for start, end, text in walk_paragraphs(xml):
        if pat.search(text):
            return xml[:start] + new_fragment + xml[end:]
    raise SystemExit(
        f"ERROR: replace marker not found: {match_regex!r}\n"
        f"First 40 paragraph texts:\n{_describe_first_paras(xml)}"
    )


def append_text_to_paragraph(xml: str, match_regex: str, suffix: str) -> str:
    """Append suffix to the last <w:t> in the first paragraph whose text matches."""
    pat = re.compile(match_regex)
    for start, end, text in walk_paragraphs(xml):
        if pat.search(text):
            # Skip if already appended.
            if "Source:" in text:
                return xml
            para_xml = xml[start:end]
            # Find the last <w:t ...>...</w:t> and append before its closing tag.
            wt_matches = list(re.finditer(r"<w:t([^>]*)>([^<]*)</w:t>", para_xml))
            if not wt_matches:
                return xml
            last = wt_matches[-1]
            # Preserve any existing attributes; ensure xml:space preserve present.
            attrs = last.group(1)
            if "xml:space" not in attrs:
                attrs = ' xml:space="preserve"' + attrs
            new_inner = last.group(2) + _xml_escape(suffix)
            rebuilt = f"<w:t{attrs}>{new_inner}</w:t>"
            new_para = para_xml[: last.start()] + rebuilt + para_xml[last.end():]
            return xml[:start] + new_para + xml[end:]
    raise SystemExit(
        f"ERROR: caption marker not found: {match_regex!r}\n"
        f"First 40 paragraph texts:\n{_describe_first_paras(xml)}"
    )


# ---------------------------------------------------------------------------
# Relationship helpers
# ---------------------------------------------------------------------------
def next_rid(rels_xml: str) -> str:
    ids = [int(x) for x in re.findall(r'Id="rId(\d+)"', rels_xml)]
    return f"rId{max(ids) + 1}"


def add_image_rel(rels_xml: str, rid: str, target: str) -> str:
    new_rel = (
        f'<Relationship Id="{rid}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="{target}"/>'
    )
    return rels_xml.replace("</Relationships>", new_rel + "</Relationships>")


# ---------------------------------------------------------------------------
# Figure 8 drawing
# ---------------------------------------------------------------------------
def fig8_drawing(rid: str, cx: int = 5486400, cy: int = 4942342) -> str:
    return (
        "<w:p>"
        "<w:pPr><w:jc w:val=\"center\"/></w:pPr>"
        "<w:r>"
        "<w:drawing>"
        "<wp:inline xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\">"
        f"<wp:extent cx=\"{cx}\" cy=\"{cy}\"/>"
        "<wp:effectExtent l=\"0\" t=\"0\" r=\"0\" b=\"0\"/>"
        "<wp:docPr id=\"1001\" name=\"Figure 8 portfolio\"/>"
        "<wp:cNvGraphicFramePr/>"
        "<a:graphic xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">"
        "<a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">"
        "<pic:pic xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">"
        "<pic:nvPicPr>"
        "<pic:cNvPr id=\"1001\" name=\"Figure 8 portfolio\"/>"
        "<pic:cNvPicPr/>"
        "</pic:nvPicPr>"
        "<pic:blipFill>"
        f"<a:blip r:embed=\"{rid}\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"/>"
        "<a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill>"
        "<pic:spPr>"
        f"<a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>"
        "<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom>"
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData>"
        "</a:graphic>"
        "</wp:inline>"
        "</w:drawing>"
        "</w:r>"
        "</w:p>"
    )


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------
def _replace_para_text(para_xml: str, new_text: str) -> str:
    """Rebuild a paragraph so its text content is exactly new_text.

    Preserves the original <w:pPr> element (first one found inside the paragraph)
    and the <w:p ...> opening tag; replaces the rest with a single <w:r><w:t>.
    """
    # Capture the opening <w:p ...> tag.
    m_open = re.match(r"(<w:p\b[^>]*>)", para_xml)
    if not m_open:
        return para_xml
    open_tag = m_open.group(1)
    # Capture the optional <w:pPr>...</w:pPr>.
    rest = para_xml[m_open.end():]
    pPr = ""
    m_ppr = re.match(r"(<w:pPr>.*?</w:pPr>)", rest, re.S)
    if m_ppr:
        pPr = m_ppr.group(1)
    # Build replacement.
    return (
        f"{open_tag}{pPr}"
        f"<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val=\"24\"/><w:szCs w:val=\"24\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{_xml_escape(new_text)}</w:t></w:r>"
        f"</w:p>"
    )


def _apply_text_substitutions(text: str) -> tuple[str, bool]:
    """Apply the standard text-level substitutions for a single paragraph's joined text.

    Returns (new_text, changed).
    """
    original = text
    # Section-heading renumbering (paragraph-start anchored).
    prefix_map = [
        ("43.1.", "4.1."),
        ("43.2.", "4.2."),
        ("43.3.", "4.3."),
        ("43.4.", "4.4."),
        ("32.6.", "3.6."),
    ]
    for old, new in prefix_map:
        if text.startswith(old):
            text = new + text[len(old):]
            break
    # Bare section-heading renumbering (only if no dotted subsection matched above).
    exact_map = [
        ("21. Background and Related Work", "2. Background and Related Work"),
        ("32. Materials and Methods", "3. Materials and Methods"),
        ("43. Simulation and Results", "4. Simulation and Results"),
    ]
    for old, new in exact_map:
        if text.strip() == old:
            text = new
            break
    # Duplicate 3.x renumbering inside section 4.
    dup_map = [
        ("3.6. Qualitative Evidence", "4.7. Qualitative Evidence"),
        ("3.7. Statistical Validation", "4.8. Statistical Validation"),
        ("3.8. Hazard Ablation", "4.9. Hazard Ablation"),
    ]
    for old, new in dup_map:
        if text.strip() == old:
            text = new
            break
    # Discussion renumber (bare headings only).
    disc_map = [
        ("4. Discussion", "5. Discussion"),
        ("4.1. Shapley Attribution", "5.1. Shapley Attribution"),
        ("4.2. Risk Coefficient as a Mission-Selection Parameter",
         "5.2. Risk Coefficient as a Mission-Selection Parameter"),
        ("4.3. Limitations", "5.3. Limitations"),
        ("5. Conclusions", "6. Conclusions"),
    ]
    for old, new in disc_map:
        if text.strip() == old:
            text = new
            break
    # Figure 18 -> Figure A1 (in-line -- operate on any occurrence).
    text = re.sub(r"Figure\s*18(\.|\s)", r"Figure A1\1", text)
    # Word-level artifacts.
    text = text.replace("wildfire eventsscenarios", "wildfire scenarios")
    text = text.replace("studyresearch", "study")
    text = text.replace("selecting cases", "representative scenarios")
    return text, text != original


def apply_paragraph_text_fixes(xml: str) -> str:
    """Paragraph-level fix pass that normalises text across split <w:t> runs.

    Walks paragraphs, joins all text runs, applies the renumbering/artifact
    substitutions, and if the text changed rebuilds the paragraph with a single
    text run. This is necessary because tracked-change artifacts split numeric
    prefixes across separate <w:t> elements (e.g. "21" = <w:t>2</w:t><w:t>1</w:t>)
    which whole-document regex cannot match.
    """
    spans = walk_paragraphs(xml)
    # Walk in reverse so earlier offsets remain valid as we rewrite.
    for start, end, text in reversed(spans):
        new_text, changed = _apply_text_substitutions(text)
        if not changed:
            continue
        new_para = _replace_para_text(xml[start:end], new_text)
        xml = xml[:start] + new_para + xml[end:]
    return xml


def apply_text_fixes(xml: str) -> str:
    # First, fix broken headings at the paragraph level (handles split runs).
    xml = apply_paragraph_text_fixes(xml)
    # Then the document-wide regex replacements.
    for pattern, replacement in TEXT_FIXES:
        xml = re.sub(pattern, replacement, xml)
    return xml


def insert_new_content(xml: str) -> str:
    """Insert all new headings, tables, and paragraphs. Ordering is bottom-up so that
    earlier inserts don't shift later markers' positions."""

    # --- WS5: insert §4.6 Event Correlation before the §4.7 Qualitative Evidence heading.
    # After WS1 fixes it's "4.7. Qualitative Evidence".
    ec_block = heading(EC_HEADING, 2) + paragraphs_xml(EC_PARAS)
    # Insert BEFORE the 4.7 heading paragraph -- we find Figure 6 ranking-inversion caption
    # (which comes just before the 4.7 heading) and append after it.
    xml = insert_after_paragraph(
        xml,
        r"^Figure 6\.",
        ec_block,
    )

    # --- WS7: insert §4.5.1 after the last §4.5 paragraph that ends with
    # "barely inflating costs near fire, while Per..."
    ws7_block = heading(WS7_HEADING, 3) + paragraphs_xml(WS7_PARAS)
    xml = insert_after_paragraph(
        xml,
        r"barely inflating costs near fire",
        ws7_block,
    )

    # --- WS4: expand §4.3 Experimental Setup. After WS1 the heading is "4.3. Experimental Setup".
    # Replace the thin body paragraph immediately following the heading.
    # The existing body starts with "The simulation environment is instantiated"
    setup_block = paragraphs_xml(SETUP_PARAS)
    # Insert the new paras AFTER the heading (and remove the thin body if present).
    # Strategy: replace the thin body paragraph with the new block.
    try:
        xml = replace_paragraph_matching(
            xml,
            r"^The simulation environment is instantiated on a 500",
            setup_block,
        )
    except SystemExit:
        # Fallback: insert after the heading.
        xml = insert_after_paragraph(xml, r"4\.3\.\s*Experimental Setup", setup_block)

    # --- WS3: insert Traceability Table after the Figure 1 illustrates paragraph in §4.1.
    trace_block = (
        paragraph(TRACE_LEADIN)
        + table(TRACE_HEADERS, TRACE_ROWS)
        + paragraph(TRACE_CAPTION, italic=True)
    )
    xml = insert_after_paragraph(
        xml,
        r"^Figure 1 illustrates the effect of wind",
        trace_block,
    )

    # --- WS2: insert Parameter Summary Table after §3.6 Reproducibility body paragraph.
    param_block = (
        heading(PARAM_HEADING, 2)
        + paragraph(PARAM_LEADIN)
        + table(PARAM_HEADERS, PARAM_ROWS)
        + paragraph(PARAM_CAPTION, italic=True)
    )
    xml = insert_after_paragraph(
        xml,
        r"^FLARE enforces 36 contracts across 15 families",
        param_block,
    )

    return xml


def apply_table_renumbers(xml: str) -> str:
    """Renumber pre-existing Tables 2,3,4 to 3,4,5 while leaving the new Table 2
    (Simulation Parameter Summary) untouched."""
    # Order: 4->5 first (highest first to avoid chaining), then 3->4, then old 2->3.
    xml = re.sub(r"Table\s*4\b", "Table 5", xml)
    xml = re.sub(r"Table\s*3\b", "Table 4", xml)
    # Now only the OLD Table 2 references (not the new parameter-summary caption) remain as
    # "Table 2" -- upgrade them, but exclude the new caption text.
    xml = re.sub(
        r"Table\s*2\b(?!\.\s*Simulation Parameter Summary)(?!\.\s*See)",
        "Table 3",
        xml,
    )
    # Note on the negative lookahead: both the caption "Table 2. Simulation Parameter Summary."
    # and the lead-in "Table 2 consolidates..." must be preserved as "Table 2".
    # First caption ("Table 2. Simulation Parameter Summary.") is excluded by the first lookahead.
    # For the lead-in "Table 2 consolidates..." -- our regex matches "Table 2" and the lookahead
    # checks for ". Simulation Parameter Summary" or ". See". Lead-in continues with " consolidates"
    # which doesn't match either, so would be renumbered.
    # Fix: use a more specific approach -- restore the lead-in manually.
    xml = xml.replace(
        "Table 3 consolidates the simulation parameters",
        "Table 2 consolidates the simulation parameters",
    )
    # The WS3 trace-table lead-in and caption already say "Table 3". After our substitution
    # they became "Table 4"; restore them.
    xml = xml.replace(
        "Table 4 provides figure-level traceability",
        "Table 3 provides figure-level traceability",
    )
    xml = xml.replace(
        "Table 4. Figure-to-scenario-to-contract traceability",
        "Table 3. Figure-to-scenario-to-contract traceability",
    )
    return xml


def patch_captions(xml: str) -> str:
    """Append 'Source:' suffixes to figure captions 1..7."""
    for fig_label, suffix in CAPTION_SUFFIXES.items():
        # Match paragraphs that start with "Figure N." (caption format, not prose reference).
        # The existing captions begin with "Figure N." followed by text.
        pattern = rf"^{re.escape(fig_label)}\."
        try:
            xml = append_text_to_paragraph(xml, pattern, suffix)
        except SystemExit as e:
            print(f"WARN: {fig_label} caption not found; skipping. ({e})", file=sys.stderr)
    return xml


def inject_figure_8(xml: str, rels_xml: str) -> tuple[str, str, str]:
    """Add relationship, inline drawing + caption, return (xml, rels_xml, rid)."""
    rid = next_rid(rels_xml)
    rels_xml = add_image_rel(rels_xml, rid, "media/image_fig8.png")

    fig8_block = fig8_drawing(rid) + paragraph(FIG8_CAPTION, italic=True)

    # Insert at end of §4.9 -- i.e. after the last paragraph of "4.9. Hazard Ablation" body,
    # just before "5. Discussion" (already renumbered from "4. Discussion").
    # The body paragraph is "Isolating dynamics layers reveals their contributions..."
    xml = insert_after_paragraph(
        xml,
        r"^Isolating dynamics layers reveals their contributions",
        fig8_block,
    )
    return xml, rels_xml, rid


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")
    slot_map = json.loads(SLOT_MAP_PATH.read_text())

    with zipfile.ZipFile(SRC) as zin:
        doc_xml = zin.read("word/document.xml").decode("utf-8")
        rels_xml = zin.read("word/_rels/document.xml.rels").decode("utf-8")
        content_types = zin.read("[Content_Types].xml").decode("utf-8")
        src_names = zin.namelist()
        src_bytes = {name: zin.read(name) for name in src_names}

    # 1. WS1 regex fixes
    doc_xml = apply_text_fixes(doc_xml)

    # 2. Insert new content (headings/tables/paragraphs)
    doc_xml = insert_new_content(doc_xml)

    # 3. Table renumbering (after inserting the new Table 2)
    doc_xml = apply_table_renumbers(doc_xml)

    # 4. Caption suffixes
    doc_xml = patch_captions(doc_xml)

    # 5. Figure 8 injection (drawing + caption + rel)
    doc_xml, rels_xml, fig8_rid = inject_figure_8(doc_xml, rels_xml)
    # Figure 8 caption suffix is embedded in FIG8_CAPTION so no separate patch needed.

    # 6. Content-Types: verify png registered.
    if 'Extension="png"' not in content_types:
        content_types = content_types.replace(
            "</Types>",
            '<Default Extension="png" ContentType="image/png"/></Types>',
        )

    # 7. Prepare image swaps per slot map (skip Figure 18 -- that's image1, renamed to A1 but
    # keep the original architecture figure; we do NOT replace it).
    # Map "Figure N" -> "imageX.png" -> which fig file to use.
    swap: dict[str, bytes] = {}
    fig_pngs = {
        "Figure 1": PNG_DIR / "fig1.png",
        "Figure 2": PNG_DIR / "fig2.png",
        "Figure 3": PNG_DIR / "fig3.png",
        "Figure 4": PNG_DIR / "fig4.png",
        "Figure 5": PNG_DIR / "fig5.png",
        "Figure 6": PNG_DIR / "fig6.png",
        "Figure 7": PNG_DIR / "fig7.png",
    }
    for fig_label, image_name in slot_map.items():
        if fig_label == "Figure 18":
            continue  # leave the architecture image alone
        png_path = fig_pngs.get(fig_label)
        if png_path is None or not png_path.exists():
            print(f"WARN: no PNG for {fig_label} ({png_path}); keeping original slot", file=sys.stderr)
            continue
        swap[f"word/media/{image_name}"] = png_path.read_bytes()

    fig8_png = (PNG_DIR / "fig8.png").read_bytes()

    # 8. Write output zip.
    with zipfile.ZipFile(DST, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in src_names:
            if name == "word/document.xml":
                zout.writestr(name, doc_xml.encode("utf-8"))
            elif name == "word/_rels/document.xml.rels":
                zout.writestr(name, rels_xml.encode("utf-8"))
            elif name == "[Content_Types].xml":
                zout.writestr(name, content_types.encode("utf-8"))
            elif name in swap:
                zout.writestr(name, swap[name])
            else:
                zout.writestr(name, src_bytes[name])
        zout.writestr("word/media/image_fig8.png", fig8_png)

    print(f"Wrote {DST} ({DST.stat().st_size} bytes). fig8 rid={fig8_rid}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _text_from_xml(xml: str) -> str:
    return " ".join(WT_RE.findall(xml))


def validate() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}{' -- ' + detail if detail and not ok else ''}")
        if not ok:
            failures += 1

    # 1. Valid zip.
    with zipfile.ZipFile(DST) as z:
        check("1a. zip integrity", z.testzip() is None)
        doc_bytes = z.read("word/document.xml")
        rels_bytes = z.read("word/_rels/document.xml.rels")
        # 5. fig8 media present and equal to source.
        try:
            fig8_in_zip = z.read("word/media/image_fig8.png")
            fig8_src = (PNG_DIR / "fig8.png").read_bytes()
            check("5. image_fig8.png present and matches source",
                  fig8_in_zip == fig8_src,
                  f"zip={len(fig8_in_zip)} src={len(fig8_src)}")
        except KeyError:
            check("5. image_fig8.png present", False, "missing from zip")

        # 8. Image slot replacements.
        replaced = 0
        fig_pngs = {
            f"word/media/image{i}.png": PNG_DIR / f"fig{i-1}.png"
            for i in range(2, 9)  # image2..image8 -> fig1..fig7
        }
        for member, src in fig_pngs.items():
            try:
                if z.read(member) == src.read_bytes():
                    replaced += 1
            except KeyError:
                pass
        check("8. at least 7 figure media slots replaced", replaced >= 7, f"replaced={replaced}")

    # 2. XML parses.
    try:
        ET.fromstring(doc_bytes)
        check("2. document.xml parses as XML", True)
    except ET.ParseError as e:
        check("2. document.xml parses as XML", False, str(e))

    doc_xml = doc_bytes.decode("utf-8")
    text = _text_from_xml(doc_xml)

    # 3. Present strings.
    present = [
        "2. Background and Related Work",
        "3. Materials and Methods",
        "4. Simulation and Results",
        "3.7. Simulation Parameter Summary",
        "4.5.1. Per-Planner and Per-Scenario Mechanism",
        "4.6. Event Correlation Analysis",
        "5. Discussion",
        "6. Conclusions",
        "Figure 8. FLARE mission portfolio",
        "Figure A1",
        "Source: scripts/gen_wind_fire_figure.py",
        "tau_c",
        "98 episodes terminate with FIRE_CAUGHT",
    ]
    for s in present:
        check(f"3. contains {s!r}", s in text, "missing")

    # 4. Absent strings.
    absent = [
        "21. Background",
        "32. Materials",
        "43. Simulation",
        "wildfire eventsscenarios",
        "studyresearch",
        "Figure 18",
    ]
    for s in absent:
        check(f"4. absent {s!r}", s not in text, "still present")

    # 6. rels contains image_fig8.png.
    rels_text = rels_bytes.decode("utf-8")
    check("6. rels has image_fig8.png target", "media/image_fig8.png" in rels_text)

    # 7. drawing references the same rid.
    # The relationship Id attribute can appear in any attribute order; find by Target.
    rid = None
    for m in re.finditer(r'<Relationship\b([^>]*)/>', rels_text):
        attrs = m.group(1)
        if 'Target="media/image_fig8.png"' in attrs:
            idm = re.search(r'Id="(rId\d+)"', attrs)
            if idm:
                rid = idm.group(1)
                break
    if rid:
        check(
            f"7. inline drawing references {rid}",
            f'r:embed="{rid}"' in doc_xml,
        )
    else:
        check("7. inline drawing references rid", False, "rel Id for fig8 not found")

    return failures


if __name__ == "__main__":
    build()
    print("\nAcceptance checks:")
    failures = validate()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(0 if failures == 0 else 1)
