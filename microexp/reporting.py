from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_optional(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _write_excel(out_dir: Path, tables: dict[str, pd.DataFrame | None]) -> Path | None:
    usable = {name: df for name, df in tables.items() if df is not None}
    if not usable:
        return None
    excel_path = out_dir / "microexp_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in usable.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return excel_path


def _condition_comparisons(
    out_dir: Path,
    trial_summary: pd.DataFrame | None,
    condition_mapping: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if trial_summary is None or condition_mapping is None:
        return None
    required = {"video_id", "trial_id", "condition"}
    if not required.issubset(set(condition_mapping.columns)):
        return None

    from scipy import stats

    mapping = condition_mapping.copy()
    mapping["condition"] = mapping["condition"].astype(str).str.strip().str.lower()
    merged = trial_summary.merge(mapping[["video_id", "trial_id", "condition"]], on=["video_id", "trial_id"], how="inner")
    if merged.empty:
        return None

    feature_cols = [
        col
        for col in merged.columns
        if col.endswith("_z_peak_abs") or col.endswith("_delta_mean") or col.endswith("_delta_auc_abs")
    ]
    rows = []
    for window in sorted(merged["window"].dropna().unique()):
        win = merged[merged["window"].eq(window)]
        for feature in feature_cols:
            pivot = win.pivot_table(index="video_id", columns="condition", values=feature, aggfunc="mean")
            if not {"sweat", "control"}.issubset(set(pivot.columns)):
                continue
            paired = pivot[["sweat", "control"]].dropna()
            if len(paired) < 2:
                continue
            diff = paired["sweat"] - paired["control"]
            t_stat, t_p = stats.ttest_rel(paired["sweat"], paired["control"], nan_policy="omit")
            try:
                if len(paired) >= 3 and diff.abs().sum() > 0:
                    w_stat, w_p = stats.wilcoxon(paired["sweat"], paired["control"])
                else:
                    w_stat, w_p = (float("nan"), float("nan"))
            except ValueError:
                w_stat, w_p = (float("nan"), float("nan"))
            rows.append(
                {
                    "window": window,
                    "feature": feature,
                    "n_pairs": int(len(paired)),
                    "sweat_mean": float(paired["sweat"].mean()),
                    "control_mean": float(paired["control"].mean()),
                    "mean_diff_sweat_minus_control": float(diff.mean()),
                    "paired_t_stat": float(t_stat),
                    "paired_t_p": float(t_p),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                }
            )

    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return None
    comparisons.to_csv(out_dir / "condition_comparison.csv", index=False)
    return comparisons


def _write_figures(out_dir: Path, trial_summary: pd.DataFrame | None, quality_report: pd.DataFrame | None) -> list[Path]:
    if trial_summary is None and quality_report is None:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    if quality_report is not None and not quality_report.empty:
        labels = quality_report["video_id"].astype(str) + " t" + quality_report["trial_id"].astype(str)
        plt.figure(figsize=(max(8, len(labels) * 0.35), 4.5))
        plt.bar(range(len(labels)), quality_report["face_detection_rate"])
        plt.xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
        plt.ylim(0, 1.05)
        plt.ylabel("Face detection rate")
        plt.tight_layout()
        path = fig_dir / "face_detection_rate.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)

    if trial_summary is not None and not trial_summary.empty:
        early = trial_summary[trial_summary["window"].eq("early_0_500ms")]
        metric_cols = [c for c in early.columns if c.endswith("_z_peak_abs")]
        if metric_cols:
            metric_means = early[metric_cols].mean(numeric_only=True).sort_values(ascending=False).head(12)
            plt.figure(figsize=(9, 4.8))
            plt.bar(range(len(metric_means)), metric_means.values)
            plt.xticks(range(len(metric_means)), [c.replace("_z_peak_abs", "") for c in metric_means.index], rotation=55, ha="right", fontsize=8)
            plt.ylabel("Mean early peak |z|")
            plt.tight_layout()
            path = fig_dir / "early_peak_abs_z.png"
            plt.savefig(path, dpi=160)
            plt.close()
            figures.append(path)

    return figures


def build_report(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = _read_optional(out_dir / "inventory.csv")
    annotations = _read_optional(out_dir / "annotations.csv")
    trial_summary = _read_optional(out_dir / "trial_summary.csv")
    micro_events = _read_optional(out_dir / "micro_events.csv")
    quality_report = _read_optional(out_dir / "quality_report.csv")
    deepface_emotions = _read_optional(out_dir / "deepface_emotions.csv")
    condition_mapping = _read_optional(out_dir / "condition_mapping.csv")
    condition_comparison = _condition_comparisons(out_dir, trial_summary, condition_mapping)

    figures = _write_figures(out_dir, trial_summary, quality_report)
    excel = _write_excel(
        out_dir,
        {
            "inventory": inventory,
            "annotations": annotations,
            "trial_summary": trial_summary,
            "micro_events": micro_events,
            "quality_report": quality_report,
            "deepface_emotions": deepface_emotions,
            "condition_mapping": condition_mapping,
            "condition_comparison": condition_comparison,
        },
    )

    lines = [
        "# Micro-Expression Analysis Report",
        "",
        "This report is blind to sweat/control condition unless `condition_mapping.csv` is present.",
        "",
        "## Inventory",
    ]
    if inventory is None or inventory.empty:
        lines.append("- No inventory found.")
    else:
        n_videos = len(inventory)
        n_reps = int(inventory.get("representative", pd.Series([True] * n_videos)).astype(bool).sum())
        n_dup = int(inventory.get("is_duplicate", pd.Series([False] * n_videos)).astype(bool).sum())
        lines.extend(
            [
                f"- Videos listed: {n_videos}",
                f"- Representative videos for analysis: {n_reps}",
                f"- Exact duplicate files detected: {n_dup}",
            ]
        )
        if "duration_s" in inventory:
            total_minutes = inventory["duration_s"].fillna(0).sum() / 60.0
            lines.append(f"- Total listed duration: {total_minutes:.1f} minutes")

    lines.extend(["", "## Annotation Status"])
    if annotations is None or annotations.empty:
        lines.append("- No annotations file found.")
    else:
        ready = pd.to_numeric(annotations["sniff_onset_s"], errors="coerce").notna()
        excluded = annotations["quality_flag"].fillna("").astype(str).str.lower().eq("exclude")
        lines.extend(
            [
                f"- Annotation rows: {len(annotations)}",
                f"- Trials with `sniff_onset_s`: {int(ready.sum())}",
                f"- Excluded trials: {int(excluded.sum())}",
            ]
        )
        if not ready.any():
            lines.append("- Trial-level facial analysis is pending manual/semi-manual onset coding.")

    lines.extend(["", "## Trial-Level Outputs"])
    if trial_summary is None:
        lines.append("- `trial_summary.csv` has not been generated yet.")
    else:
        lines.append(f"- Summary rows: {len(trial_summary)}")
    if micro_events is None:
        lines.append("- `micro_events.csv` has not been generated yet.")
    else:
        lines.append(f"- Micro-events detected: {len(micro_events)}")
        if not micro_events.empty and "metric" in micro_events:
            top = micro_events["metric"].value_counts().head(7)
            lines.append("- Events by metric: " + ", ".join(f"{k}={v}" for k, v in top.items()))
    if quality_report is None:
        lines.append("- `quality_report.csv` has not been generated yet.")
    else:
        mean_face = quality_report["face_detection_rate"].mean()
        lines.append(f"- Mean face detection rate: {mean_face:.3f}")
    if deepface_emotions is None:
        lines.append("- `deepface_emotions.csv` has not been generated yet.")
    else:
        lines.append(f"- DeepFace emotion summary rows: {len(deepface_emotions)}")
        if "deepface_dominant_emotion_mode" in deepface_emotions:
            modes = deepface_emotions["deepface_dominant_emotion_mode"].dropna().astype(str)
            modes = modes[modes.ne("")]
            if not modes.empty:
                top = modes.value_counts().head(5)
                lines.append("- DeepFace dominant modes: " + ", ".join(f"{k}={v}" for k, v in top.items()))

    lines.extend(["", "## Condition Mapping"])
    if condition_mapping is None:
        lines.append("- No `condition_mapping.csv` found. No sweat/control claims were made.")
    elif condition_comparison is None:
        lines.append("- `condition_mapping.csv` found, but no paired sweat/control comparisons could be computed yet.")
    else:
        lines.append(f"- Paired sweat/control comparison rows: {len(condition_comparison)}")
        top = condition_comparison.sort_values("paired_t_p", na_position="last").head(5)
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.window} / {row.feature}: n={row.n_pairs}, "
                f"mean diff={row.mean_diff_sweat_minus_control:.4g}, paired t p={row.paired_t_p:.4g}"
            )

    if figures:
        lines.extend(["", "## Figures"])
        for fig in figures:
            lines.append(f"- `{fig.relative_to(out_dir)}`")

    if excel:
        lines.extend(["", "## Workbook", f"- `{excel.name}`"])

    report_path = out_dir / "microexp_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
