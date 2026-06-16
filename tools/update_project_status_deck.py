"""Update selected text boxes in the project status PowerPoint deck.

The script edits only slide XML text content and writes a new PPTX file.
It intentionally keeps the original deck, slide count, layouts, and media.
"""

from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def shape_texts(root: ET.Element) -> list[ET.Element]:
    return root.findall(".//p:sp", NS)


def template_props(tx_body: ET.Element) -> tuple[ET.Element | None, ET.Element | None]:
    paragraph = tx_body.find("a:p", NS)
    if paragraph is None:
        return None, None
    p_pr = paragraph.find("a:pPr", NS)
    r_pr = paragraph.find(".//a:rPr", NS)
    return p_pr, r_pr


def replace_shape_text(root: ET.Element, shape_index: int, paragraphs: list[str]) -> None:
    shapes = shape_texts(root)
    try:
        shape = shapes[shape_index - 1]
    except IndexError as exc:
        raise ValueError(f"slide has no shape #{shape_index}") from exc

    tx_body = shape.find("p:txBody", NS)
    if tx_body is None:
        raise ValueError(f"shape #{shape_index} has no text body")

    p_pr_template, r_pr_template = template_props(tx_body)

    for child in list(tx_body):
        if child.tag == qn("a", "p"):
            tx_body.remove(child)

    for text in paragraphs:
        paragraph = ET.Element(qn("a", "p"))
        if p_pr_template is not None:
            paragraph.append(copy.deepcopy(p_pr_template))

        run = ET.SubElement(paragraph, qn("a", "r"))
        if r_pr_template is not None:
            run.append(copy.deepcopy(r_pr_template))
        text_el = ET.SubElement(run, qn("a", "t"))
        text_el.text = text
        tx_body.append(paragraph)


UPDATES: dict[int, dict[int, list[str]]] = {
    6: {
        5: [
            "Indexed MP4 files: 19",
            "Exact duplicate files excluded: 4",
            "Representative videos processed: 15/15",
            "Frame rate processed: full 100 fps",
            "MediaPipe frames extracted: 450,805",
            "DeepFace auxiliary samples: completed",
            "Candidate micro-events: 3,614",
            "Candidate sniff-like windows: 114",
        ],
        6: [
            "• File governance is implemented: videos are indexed, hashed, deduplicated, and originals stay unchanged.",
            "• Full-video automatic extraction is complete across all representative videos.",
            "• Automatic events and sniff windows are candidate markers, not final FACS labels.",
            "• Condition labels and validated sniff onsets are still required for sweat/control inference.",
            "• No participant frames, face images, or video stills are used in this status deck.",
        ],
    },
    8: {
        5: [
            "• Runnable Python package and CLI manage indexing, extraction, analysis, and reporting stages.",
            "• Video indexing, SHA-256 duplicate detection, and representative-video selection are implemented.",
            "• Full 100 fps MediaPipe-based AU proxy extraction completed for 15 representative videos.",
            "• DeepFace runs as a sampled auxiliary emotion layer, not as the micro-expression measure.",
            "• GitHub repository was updated with commit accd4f4 and numeric results are available.",
        ],
        6: [
            "• Fill condition=sweat/control for all included trials before hypothesis testing.",
            "• Confirm candidate sniff windows and complete missing sniff/baseline timings.",
            "• Complete 21 missing timing cells or mark trials as excluded with reasons.",
            "• Manually review 2025-12-28_14-03-58 and P2L42107 because of QC flags.",
            "• Final statistics should run only after valid condition labels and QC decisions.",
        ],
    },
    9: {
        4: ["1", "Data governance", "indexed, hashed, deduplicated"],
        5: ["2", "Input system", "workbook partly filled"],
        6: ["3", "Feature engine", "15/15 videos processed at 100 fps"],
        7: ["4", "Analysis layer", "blocked until condition coding is valid"],
        13: [
            "• condition remains blank for all 34 included trials, so no sweat/control test is reported yet.",
            "• 23 active trials have baseline/sniff timing; 21 timing cells are still missing.",
            "• Four trials are marked exclude=yes and should retain documented exclusion reasons.",
            "• Candidate sniff-like windows require manual confirmation before trial-level analysis.",
        ],
        14: [
            "• video_index/dedupe outputs document file-level governance.",
            "• frame_features_100fps, DeepFace samples, quality_full, candidate events, and candidate sniffs now exist.",
            "• statistical_report.md and statistical_summary.xlsx summarize the full automatic run.",
            "• After annotation completion: trial_features, statistics_summary, mixed_model_ready, and QC summaries.",
        ],
    },
    10: {
        5: [
            "• The presentation now tells a status story for supervisors: what is done, what is missing, and why conclusions wait.",
            "• The theoretical frame links sweat chemosignals, implicit processing, and facial micro-expressions.",
            "• The method remains event-based and privacy-preserving: no participant frames or contact sheets are shown.",
            "• The current update separates completed automatic extraction from not-yet-final trial-level inference.",
        ],
        6: [
            "• Python package, CLI, tests, README, templates, workbook, and analysis bundle were created.",
            "• Nineteen MP4 files were scanned; four exact duplicates were excluded from representative counts.",
            "• Fifteen representative videos were processed at 100 fps, producing 450,805 frame rows.",
            "• Outputs include 3,614 candidate micro-events, 114 sniff-like windows, QC tables, and DeepFace samples.",
            "• GitHub repository was pushed with commit accd4f4 for the current pipeline and results.",
        ],
    },
    16: {
        5: [
            "• Fill condition as sweat/control for every included trial before any final comparison.",
            "• Complete missing baseline/sniff timing fields, or mark unusable trials as exclude=yes with a reason.",
            "• Manually confirm likely sniff windows from auto_candidate_sniffs.csv.",
            "• Review low-quality or unstable-head-pose videos before final reporting.",
            "• Keep protocol clarifications and QC notes explicit rather than inferred from filenames.",
        ],
        6: [
            "• Save and close the workbook, then export it back to CSV.",
            "• Re-run extract-features only after condition and timing validation pass.",
            "• Run analyze to generate trial-level features, paired tests, QC summaries, and model-ready tables.",
            "• Update GitHub and the final report after valid trial-level outputs are regenerated.",
            "• Report sweat/control effects only when enough condition-coded paired trials exist.",
        ],
    },
    17: {
        5: [
            "• Workbook is in micro_exp_analysis_bundle and was updated on 29 May 2026.",
            "• Trial Map currently contains 34 included trials.",
            "• Event Annotations contain 23 active trials with baseline/sniff timing filled.",
            "• Four trials are marked exclude_trial=yes.",
            "• The workbook was exported to CSV for validation without changing source videos.",
        ],
        6: [
            "• condition is still blank for all 34 trials, so sweat/control analysis cannot run yet.",
            "• 21 required timing cells are missing across active, non-excluded rows.",
            "• extract-features is correctly blocked until condition=sweat/control is filled.",
            "• Candidate sniff windows still need human approval before they become analysis onsets.",
            "• Current annotation status supports progress tracking, not final hypothesis interpretation.",
        ],
    },
    18: {
        5: [
            "• Full automatic extraction completed for all 15 representative videos.",
            "• MediaPipe landmark-derived AU proxies were extracted at full 100 fps.",
            "• DeepFace was sampled as an auxiliary emotion layer, not a micro-expression substitute.",
            "• Regenerated outputs include statistical_report.md, statistical_summary.xlsx, quality_full.csv, auto_candidate_events.csv, and auto_candidate_sniffs.csv.",
            "• Automatic events remain candidate markers until manually reviewed.",
        ],
        6: [
            "• The pipeline can process the full representative dataset, not just a smoke test.",
            "• Numeric feature files and DeepFace sample files exist for every representative video.",
            "• Event density and sniff candidates identify where human review should focus.",
            "• No sweat/control conclusions are reported until condition labels and onsets are validated.",
            "• Privacy was preserved: no participant frames, face images, or identifiable stills were added.",
        ],
    },
    19: {
        5: [
            "• Representative videos processed: 15/15",
            "• Frame-level rows in manifest: 450,805",
            "• Sampling target: full 100 fps",
            "• Candidate micro-events: 3,614",
            "• Candidate sniff-like windows: 114",
            "• DeepFace sample outputs: 15/15 videos",
        ],
        6: [
            "• Mean face-detection rate: 93.9%",
            "• Quality tiers: 13 good, 1 usable_review, 1 review_or_exclude",
            "• Review/exclude candidate: 2025-12-28_14-03-58",
            "• Usable with review: P2L42107",
            "• Current findings remain exploratory until manual annotation is complete.",
        ],
    },
    24: {
        3: ["Updated status from full extraction to annotation blockers and final-analysis readiness"],
        8: ["Implemented", "video index + duplicates", "19 files / 4 duplicates / 15 representatives"],
        12: ["Partly filled", "34 trials / 23 timed / 4 excluded", "add condition + missing timing"],
        16: ["Implemented", "100 fps MediaPipe + DeepFace auxiliary", "450,805 frames / 3,614 events / 114 sniff candidates"],
        20: ["Blocked by inputs", "automatic descriptive report exists", "run trial-level stats after valid coding"],
        26: [
            "available now: GitHub accd4f4  •  statistical_report  •  statistical_summary  •  quality_full  •  auto candidate events/sniffs",
            "still needed: condition-coded trials  •  manual sniff confirmation  •  extract-features/analyze  •  final supervisor-ready results",
        ],
        27: [
            "We now have a complete automatic extraction layer; final research conclusions still depend on condition labels, timing completion, and manual QC."
        ],
    },
}


def update_deck(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.startswith("ppt/slides/slide") and info.filename.endswith(".xml"):
                stem = Path(info.filename).stem
                try:
                    slide_num = int(stem.replace("slide", ""))
                except ValueError:
                    slide_num = -1
                if slide_num in UPDATES:
                    root = ET.fromstring(data)
                    for shape_index, paragraphs in UPDATES[slide_num].items():
                        replace_shape_text(root, shape_index, paragraphs)
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            zout.writestr(info, data)


def validate(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if not dst.exists():
        raise FileNotFoundError(dst)

    with zipfile.ZipFile(src, "r") as src_zip, zipfile.ZipFile(dst, "r") as dst_zip:
        src_slides = sorted(p for p in src_zip.namelist() if p.startswith("ppt/slides/slide") and p.endswith(".xml"))
        dst_slides = sorted(p for p in dst_zip.namelist() if p.startswith("ppt/slides/slide") and p.endswith(".xml"))
        if len(src_slides) != 25 or len(dst_slides) != 25:
            raise ValueError(f"expected 25 slides, got source={len(src_slides)}, output={len(dst_slides)}")

        required_strings = [
            "Representative videos processed: 15/15",
            "MediaPipe frames extracted: 450,805",
            "Candidate micro-events: 3,614",
            "Candidate sniff-like windows: 114",
            "Trial Map currently contains 34 included trials.",
            "condition is still blank for all 34 trials",
            "GitHub repository was pushed with commit accd4f4",
            "Quality tiers: 13 good, 1 usable_review, 1 review_or_exclude",
        ]
        all_text = []
        for slide_path in dst_slides:
            root = ET.fromstring(dst_zip.read(slide_path))
            all_text.extend(t.text or "" for t in root.findall(".//a:t", NS))
        combined = "\n".join(all_text)
        missing = [s for s in required_strings if s not in combined]
        if missing:
            raise ValueError(f"missing required deck text: {missing}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: update_project_status_deck.py SOURCE.pptx DESTINATION.pptx", file=sys.stderr)
        return 2
    src = Path(argv[1])
    dst = Path(argv[2])
    update_deck(src, dst)
    validate(src, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
