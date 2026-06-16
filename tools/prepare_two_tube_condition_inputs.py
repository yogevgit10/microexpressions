"""Prepare two-test-tube condition inputs for trial-level analysis."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "microexpressions_repo" / "results"
BUNDLE_ANNOTATIONS = Path(r"C:\Users\gitya\Downloads\micro_exp_analysis_bundle\inputs\annotations.csv")

ANALYSIS_COLUMNS = [
    "video_id",
    "source_path",
    "trial_id",
    "approach_onset_s",
    "sniff_onset_s",
    "sniff_offset_s",
    "response_s",
    "occlusion_notes",
    "quality_flag",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def trial_number(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("trial_id", "")))
    except ValueError:
        return -1


def has_sniff(row: dict[str, str]) -> bool:
    return bool(str(row.get("sniff_onset_s", "")).strip())


def is_ok(row: dict[str, str]) -> bool:
    return str(row.get("quality_flag", "")).strip().lower() == "ok"


def main() -> None:
    rows = read_rows(BUNDLE_ANNOTATIONS)
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)

    two_tube_video_ids = sorted(
        video_id
        for video_id, group in by_video.items()
        if len(group) == 2 and {trial_number(row) for row in group} == {1, 2}
    )
    four_trial_video_ids = sorted(
        video_id
        for video_id, group in by_video.items()
        if len(group) == 4 and {trial_number(row) for row in group} == {1, 2, 3, 4}
    )

    two_tube_rows = sorted(
        [row for video_id in two_tube_video_ids for row in by_video[video_id]],
        key=lambda row: (row["video_id"], trial_number(row)),
    )
    primary_video_ids = []
    for video_id in two_tube_video_ids:
        group = by_video[video_id]
        if all(is_ok(row) and has_sniff(row) for row in group):
            primary_video_ids.append(video_id)

    primary_rows = sorted(
        [row for video_id in primary_video_ids for row in by_video[video_id]],
        key=lambda row: (row["video_id"], trial_number(row)),
    )
    four_trial_rows = sorted(
        [row for video_id in four_trial_video_ids for row in by_video[video_id]],
        key=lambda row: (row["video_id"], trial_number(row)),
    )
    pending_or_excluded_rows = sorted(
        [
            row
            for row in two_tube_rows
            if row["video_id"] not in primary_video_ids
            or not is_ok(row)
            or not has_sniff(row)
        ],
        key=lambda row: (row["video_id"], trial_number(row)),
    )

    mapping_rows = []
    for row in two_tube_rows:
        trial_id = trial_number(row)
        mapping_rows.append(
            {
                "video_id": row["video_id"],
                "trial_id": trial_id,
                "condition": "control" if trial_id == 1 else "sweat",
                "protocol": "two_test_tube",
                "condition_label_note": "trial_1_placebo_control_trial_2_sweat",
                "analysis_status": "primary" if row["video_id"] in primary_video_ids else "not_primary",
                "quality_flag": row.get("quality_flag", ""),
                "sniff_onset_s": row.get("sniff_onset_s", ""),
            }
        )

    summary_rows = [
        {"item": "total_annotation_rows", "value": len(rows)},
        {"item": "total_annotation_videos", "value": len(by_video)},
        {"item": "two_tube_videos", "value": len(two_tube_video_ids)},
        {"item": "two_tube_rows", "value": len(two_tube_rows)},
        {"item": "primary_two_tube_videos", "value": len(primary_video_ids)},
        {"item": "primary_two_tube_rows", "value": len(primary_rows)},
        {"item": "four_trial_videos", "value": len(four_trial_video_ids)},
        {"item": "four_trial_rows", "value": len(four_trial_rows)},
        {"item": "pending_or_excluded_two_tube_rows", "value": len(pending_or_excluded_rows)},
        {"item": "condition_mapping_rows", "value": len(mapping_rows)},
    ]
    flag_counts = Counter(row.get("quality_flag", "") or "blank" for row in rows)
    for flag, count in sorted(flag_counts.items()):
        summary_rows.append({"item": f"annotation_quality_flag_{flag}", "value": count})

    write_rows(RESULTS_DIR / "annotations_full.csv", rows, list(rows[0].keys()))
    write_rows(RESULTS_DIR / "annotations.csv", primary_rows, ANALYSIS_COLUMNS)
    write_rows(RESULTS_DIR / "annotations_two_tube_primary.csv", primary_rows, list(rows[0].keys()))
    write_rows(RESULTS_DIR / "annotations_four_trial_exploratory.csv", four_trial_rows, list(rows[0].keys()))
    write_rows(RESULTS_DIR / "annotations_pending_or_excluded.csv", pending_or_excluded_rows, list(rows[0].keys()))
    write_rows(
        RESULTS_DIR / "condition_mapping.csv",
        mapping_rows,
        [
            "video_id",
            "trial_id",
            "condition",
            "protocol",
            "condition_label_note",
            "analysis_status",
            "quality_flag",
            "sniff_onset_s",
        ],
    )
    write_rows(RESULTS_DIR / "protocol_split_summary.csv", summary_rows, ["item", "value"])

    notes = [
        "# Two-Test-Tube Condition Mapping Notes",
        "",
        "- Applied only to videos with exactly two annotated trials.",
        "- `trial_id=1` is coded as `control` (placebo/control tube).",
        "- `trial_id=2` is coded as `sweat`.",
        "- Four-trial videos are retained separately as exploratory protocol rows.",
        "- Primary analysis includes only two-tube videos where both trials are `ok` and have `sniff_onset_s`.",
        "",
        f"- Primary paired videos: {len(primary_video_ids)}",
        f"- Primary paired rows: {len(primary_rows)}",
        f"- Condition mapping rows: {len(mapping_rows)}",
    ]
    (RESULTS_DIR / "condition_mapping_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Wrote primary annotations: {RESULTS_DIR / 'annotations.csv'}")
    print(f"Wrote condition mapping: {RESULTS_DIR / 'condition_mapping.csv'}")
    print(f"Primary paired videos: {len(primary_video_ids)}")
    print(f"Primary rows: {len(primary_rows)}")
    print(f"Four-trial exploratory rows: {len(four_trial_rows)}")


if __name__ == "__main__":
    main()
