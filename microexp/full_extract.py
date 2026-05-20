from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .events import EventDetectionConfig, detect_micro_events
from .facial_metrics import FaceMetricExtractor, KEY_EVENT_METRICS, METRIC_COLUMNS
from .inventory import build_inventory, load_inventory, require_cv2


FRAME_FEATURE_DIR = "frame_features_100fps"
DEEPFACE_SAMPLE_DIR = "deepface_samples"
STAT_FIGURE_DIR = "statistical_figures"

EVENT_METRICS = [
    "au06_cheek_raise_proxy",
    "au12_lip_corner_pull_proxy",
    "au09_nose_wrinkle_proxy",
    "au10_upper_lip_raise_proxy",
    "au24_lip_press_proxy",
    "au45_blink_proxy",
    "cheek_composite_proxy",
    "mouth_open_proxy",
    "mouth_asymmetry_proxy",
    "head_yaw_proxy",
]

SNIFF_CUE_METRICS = [
    "au09_nose_wrinkle_proxy",
    "au10_upper_lip_raise_proxy",
    "mouth_open_proxy",
    "au45_blink_proxy",
    "head_yaw_proxy",
    "cheek_composite_proxy",
]


def _quiet_progress() -> bool:
    return os.environ.get("MICROEXP_QUIET", "").strip().lower() in {"1", "true", "yes"}


def _resize_for_processing(cv2, frame, max_width: int | None):
    if not max_width or max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    new_height = int(round(height * (max_width / width)))
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


def _select_inventory(
    out_dir: Path,
    video_dir: Path,
    include_duplicates: bool,
    video_ids: list[str] | None,
    limit_videos: int | None,
) -> pd.DataFrame:
    inventory_path = out_dir / "inventory.csv"
    if inventory_path.exists():
        inventory = load_inventory(inventory_path)
    else:
        inventory = build_inventory(video_dir, out_dir, hash_files=True)

    selected = inventory.copy()
    if not include_duplicates and "representative" in selected.columns:
        selected = selected[selected["representative"].astype(bool)]
    if video_ids:
        selected = selected[selected["video_id"].isin(video_ids)]
    selected = selected.sort_values("video_id").reset_index(drop=True)
    if limit_videos is not None:
        selected = selected.head(limit_videos)
    return selected


def _flush_rows(path: Path, rows: list[dict], write_header: bool) -> bool:
    if not rows:
        return write_header
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, mode="a", header=write_header)
    rows.clear()
    return False


def extract_video_features(
    video_row: pd.Series,
    out_dir: Path,
    target_fps: float = 100.0,
    include_deepface: bool = False,
    deepface_every_s: float = 1.0,
    max_width: int | None = 960,
    force: bool = False,
    max_frames: int | None = None,
) -> dict:
    cv2 = require_cv2()
    video_id = str(video_row["video_id"])
    video_path = Path(str(video_row["source_path"]))
    feature_dir = out_dir / FRAME_FEATURE_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{video_id}.csv"

    deepface_path = None
    if include_deepface:
        deepface_dir = out_dir / DEEPFACE_SAMPLE_DIR
        deepface_dir.mkdir(parents=True, exist_ok=True)
        deepface_path = deepface_dir / f"{video_id}.csv"

    deepface_satisfied = not include_deepface or (deepface_path is not None and deepface_path.exists())
    if feature_path.exists() and not force and deepface_satisfied:
        existing = pd.read_csv(feature_path, usecols=lambda col: col in {"processed_fps", "source_fps"})
        processed_frames = sum(1 for _ in feature_path.open("r", encoding="utf-8")) - 1
        processed_fps = float(pd.to_numeric(existing.get("processed_fps"), errors="coerce").dropna().iloc[0]) if "processed_fps" in existing and existing["processed_fps"].notna().any() else np.nan
        source_fps = float(pd.to_numeric(existing.get("source_fps"), errors="coerce").dropna().iloc[0]) if "source_fps" in existing and existing["source_fps"].notna().any() else np.nan
        return {
            "video_id": video_id,
            "source_path": str(video_path),
            "feature_path": str(feature_path),
            "deepface_path": str(deepface_path) if deepface_path and deepface_path.exists() else "",
            "status": "skipped_existing",
            "source_fps": source_fps,
            "processed_fps": processed_fps,
            "processed_frames": processed_frames,
            "max_width": max_width or 0,
        }

    if feature_path.exists():
        feature_path.unlink()
    if deepface_path and deepface_path.exists():
        deepface_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS for video: {video_path}")

    frame_step = max(1, int(round(source_fps / target_fps))) if target_fps > 0 else 1
    processed_fps = source_fps / frame_step
    expected_process_frames = int(math.ceil(frame_count / frame_step)) if frame_count else None
    if max_frames is not None and expected_process_frames is not None:
        expected_process_frames = min(expected_process_frames, max_frames)

    extractor = FaceMetricExtractor()
    deepface_extractor = None
    if include_deepface:
        from .deepface_emotions import DeepFaceEmotionExtractor

        deepface_extractor = DeepFaceEmotionExtractor()

    rows: list[dict] = []
    deepface_rows: list[dict] = []
    write_header = True
    deepface_write_header = True
    processed_frames = 0
    read_frames = 0
    next_deepface_s = 0.0
    batch_size = 2500

    try:
        with tqdm(total=expected_process_frames, desc=f"extract {video_id}", leave=False, disable=_quiet_progress()) as bar:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    read_frames += 1
                    continue
                if max_frames is not None and processed_frames >= max_frames:
                    break

                timestamp_s = frame_idx / source_fps
                proc_frame = _resize_for_processing(cv2, frame, max_width=max_width)
                metric_row = extractor.process_bgr(cv2, proc_frame, frame_idx, timestamp_s)
                metric_row.update(
                    {
                        "video_id": video_id,
                        "source_file": video_path.name,
                        "source_fps": source_fps,
                        "processed_fps": processed_fps,
                        "frame_step": frame_step,
                    }
                )
                rows.append(metric_row)

                if deepface_extractor is not None and timestamp_s + 1e-9 >= next_deepface_s:
                    emotion_row = deepface_extractor.analyze_bgr(proc_frame)
                    emotion_row.update(
                        {
                            "video_id": video_id,
                            "source_file": video_path.name,
                            "frame_idx": frame_idx,
                            "timestamp_s": timestamp_s,
                        }
                    )
                    deepface_rows.append(emotion_row)
                    next_deepface_s += max(deepface_every_s, 1.0 / source_fps)

                processed_frames += 1
                read_frames += 1
                frame_idx += 1
                if len(rows) >= batch_size:
                    write_header = _flush_rows(feature_path, rows, write_header)
                if deepface_path and len(deepface_rows) >= 250:
                    deepface_write_header = _flush_rows(deepface_path, deepface_rows, deepface_write_header)
                if bar.total is not None:
                    bar.update(1)
    finally:
        extractor.close()
        cap.release()

    write_header = _flush_rows(feature_path, rows, write_header)
    if deepface_path:
        deepface_write_header = _flush_rows(deepface_path, deepface_rows, deepface_write_header)
        if deepface_write_header and not deepface_path.exists():
            pd.DataFrame().to_csv(deepface_path, index=False)

    return {
        "video_id": video_id,
        "source_path": str(video_path),
        "feature_path": str(feature_path),
        "deepface_path": str(deepface_path) if deepface_path else "",
        "status": "processed",
        "source_fps": source_fps,
        "processed_fps": processed_fps,
        "source_frames": frame_count,
        "processed_frames": processed_frames,
        "width": width,
        "height": height,
        "max_width": max_width or 0,
    }


def _rolling_z(values: pd.Series, fps: float) -> pd.Series:
    min_periods = max(5, int(round(fps * 0.5)))
    smooth_window = max(3, int(round(fps * 0.05)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    baseline_window = max(min_periods, int(round(fps * 2.0)))
    scale_window = max(baseline_window, int(round(fps * 8.0)))

    series = pd.to_numeric(values, errors="coerce").astype(float)
    series = series.interpolate(limit=max(1, int(round(fps * 0.2))), limit_direction="both")
    smooth = series.rolling(smooth_window, center=True, min_periods=1).median()
    baseline = smooth.rolling(baseline_window, center=True, min_periods=min_periods).median()
    residual = smooth - baseline
    scale = residual.rolling(scale_window, center=True, min_periods=min_periods).std()

    global_mad = float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    global_scale = 1.4826 * global_mad if np.isfinite(global_mad) and global_mad > 1e-9 else float(np.nanstd(residual))
    if not np.isfinite(global_scale) or global_scale <= 1e-9:
        global_scale = 1.0
    scale = scale.fillna(global_scale).replace(0, global_scale)
    z = residual / scale
    return z.replace([np.inf, -np.inf], np.nan)


def _metric_activity(values: pd.Series, fps: float) -> float:
    series = pd.to_numeric(values, errors="coerce").astype(float)
    diffs = series.diff().abs()
    if not diffs.notna().any():
        return float("nan")
    return float(diffs.mean() * fps)


def analyze_feature_file(feature_path: Path, video_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(feature_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"video_id": video_id, "status": "empty_features"}

    fps = float(pd.to_numeric(df.get("processed_fps"), errors="coerce").dropna().iloc[0])
    duration_s = float(pd.to_numeric(df["timestamp_s"], errors="coerce").max())
    event_rows = []
    z_cache: dict[str, pd.Series] = {}

    for metric in EVENT_METRICS:
        if metric not in df.columns:
            continue
        z = _rolling_z(df[metric].where(df["face_detected"].eq(1)), fps=fps)
        z_cache[metric] = z
        events = detect_micro_events(
            df["timestamp_s"].to_numpy(dtype=float),
            z.to_numpy(dtype=float),
            metric,
            config=EventDetectionConfig(threshold_z=3.0, min_duration_s=0.08, max_duration_s=0.80, max_gap_s=0.05),
        )
        if events.empty:
            continue
        events["video_id"] = video_id
        events["event_start_s"] = events.pop("event_start_rel_s")
        events["event_end_s"] = events.pop("event_end_rel_s")
        events["requires_manual_confirmation"] = True
        event_rows.extend(events.to_dict("records"))

    summary = {
        "video_id": video_id,
        "duration_s": duration_s,
        "processed_frames": int(len(df)),
        "processed_fps": fps,
        "face_detection_rate": float(df["face_detected"].mean()) if "face_detected" in df else np.nan,
    }
    for metric in METRIC_COLUMNS:
        if metric not in df:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        summary[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        summary[f"{metric}_std"] = float(vals.std()) if vals.notna().any() else np.nan
        summary[f"{metric}_p05"] = float(vals.quantile(0.05)) if vals.notna().any() else np.nan
        summary[f"{metric}_p50"] = float(vals.quantile(0.50)) if vals.notna().any() else np.nan
        summary[f"{metric}_p95"] = float(vals.quantile(0.95)) if vals.notna().any() else np.nan
        summary[f"{metric}_activity_per_s"] = _metric_activity(vals, fps=fps)

    quality = dict(summary)
    yaw = pd.to_numeric(df.get("head_yaw_proxy"), errors="coerce")
    roll = pd.to_numeric(df.get("head_roll_deg"), errors="coerce")
    quality.update(
        {
            "mean_abs_head_yaw_proxy": float(yaw.abs().mean()) if yaw.notna().any() else np.nan,
            "max_abs_head_yaw_proxy": float(yaw.abs().max()) if yaw.notna().any() else np.nan,
            "mean_abs_head_roll_deg": float(roll.abs().mean()) if roll.notna().any() else np.nan,
            "max_abs_head_roll_deg": float(roll.abs().max()) if roll.notna().any() else np.nan,
        }
    )
    quality["quality_tier"] = _quality_tier(quality)
    quality["occlusion_risk"] = _occlusion_risk(quality)

    events_df = pd.DataFrame(event_rows)
    sniffs_df = _candidate_sniffs_from_events(events_df, video_id, duration_s)
    return events_df, sniffs_df, quality


def _quality_tier(row: dict) -> str:
    face_rate = row.get("face_detection_rate", np.nan)
    yaw = row.get("mean_abs_head_yaw_proxy", np.nan)
    roll = row.get("mean_abs_head_roll_deg", np.nan)
    if pd.notna(face_rate) and face_rate >= 0.95 and (pd.isna(yaw) or yaw < 0.18) and (pd.isna(roll) or roll < 8):
        return "good"
    if pd.notna(face_rate) and face_rate >= 0.80:
        return "usable_review"
    return "review_or_exclude"


def _occlusion_risk(row: dict) -> str:
    face_rate = row.get("face_detection_rate", np.nan)
    yaw = row.get("max_abs_head_yaw_proxy", np.nan)
    roll = row.get("max_abs_head_roll_deg", np.nan)
    reasons = []
    if pd.notna(face_rate) and face_rate < 0.90:
        reasons.append("low_face_detection")
    if pd.notna(yaw) and yaw > 0.45:
        reasons.append("large_yaw")
    if pd.notna(roll) and roll > 18:
        reasons.append("large_roll")
    return ";".join(reasons) if reasons else "low"


def _candidate_sniffs_from_events(events: pd.DataFrame, video_id: str, duration_s: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    cue_events = events[events["metric"].isin(SNIFF_CUE_METRICS)].copy()
    if cue_events.empty:
        return pd.DataFrame()

    bin_s = 0.5
    cue_events["bin"] = (cue_events["event_start_s"] / bin_s).round().astype(int)
    grouped = (
        cue_events.groupby("bin")
        .agg(score=("peak_abs_z", "sum"), event_count=("metric", "count"))
        .reset_index()
        .sort_values(["score", "event_count"], ascending=False)
    )

    selected = []
    min_gap_s = 8.0
    for rec in grouped.to_dict("records"):
        center = float(rec["bin"] * bin_s)
        if any(abs(center - other["candidate_center_s"]) < min_gap_s for other in selected):
            continue
        window_events = cue_events[cue_events["event_start_s"].between(center - 2.0, center + 3.0, inclusive="both")]
        metrics = window_events["metric"].value_counts().head(4)
        selected.append(
            {
                "video_id": video_id,
                "candidate_center_s": center,
                "candidate_start_s": max(0.0, center - 2.0),
                "candidate_end_s": min(duration_s, center + 3.0),
                "score": float(rec["score"]),
                "event_count": int(rec["event_count"]),
                "dominant_metrics": ", ".join(f"{metric}:{count}" for metric, count in metrics.items()),
                "requires_manual_confirmation": True,
                "note": "Automatic sniff-like candidate from facial motion/event density.",
            }
        )
        if len(selected) >= 8:
            break
    return pd.DataFrame(selected).sort_values("candidate_center_s") if selected else pd.DataFrame()


def _load_deepface_samples(out_dir: Path) -> pd.DataFrame | None:
    deepface_dir = out_dir / DEEPFACE_SAMPLE_DIR
    if not deepface_dir.exists():
        return None
    frames = []
    for path in sorted(deepface_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "deepface_full.csv", index=False)
    return combined


def _pca_cluster(video_metric_summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame | None:
    if video_metric_summary.empty:
        return None

    base = video_metric_summary[["video_id"]].copy()
    numeric_cols = [
        col
        for col in video_metric_summary.columns
        if col != "video_id" and (col.endswith("_mean") or col.endswith("_std") or col.endswith("_activity_per_s"))
    ]
    features = video_metric_summary[["video_id", *numeric_cols]].copy()
    if events is not None and not events.empty:
        counts = events.pivot_table(index="video_id", columns="metric", values="event_start_s", aggfunc="count", fill_value=0)
        counts.columns = [f"event_count_{col}" for col in counts.columns]
        features = features.merge(counts.reset_index(), on="video_id", how="left")
    features = features.fillna(0.0)

    x = features.drop(columns=["video_id"]).to_numpy(dtype=float)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return None
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :2] * s[:2]
    out = base.copy()
    out["pca1"] = scores[:, 0]
    out["pca2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0

    if len(out) >= 3:
        try:
            from scipy.cluster.vq import kmeans2

            k = min(3, len(out))
            _, labels = kmeans2(scores[:, :2], k, minit="points", seed=7)
            out["cluster"] = labels.astype(int)
        except Exception:
            out["cluster"] = -1
    else:
        out["cluster"] = -1
    return out


def _write_figures(
    out_dir: Path,
    quality: pd.DataFrame,
    events: pd.DataFrame,
    video_metric_summary: pd.DataFrame,
    pca_scores: pd.DataFrame | None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    fig_dir = out_dir / STAT_FIGURE_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    if not quality.empty and "face_detection_rate" in quality:
        labels = quality["video_id"].astype(str)
        plt.figure(figsize=(max(8, len(labels) * 0.45), 4.5))
        plt.bar(range(len(labels)), quality["face_detection_rate"])
        plt.xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
        plt.ylim(0, 1.05)
        plt.ylabel("Face detection rate")
        plt.tight_layout()
        path = fig_dir / "quality_face_detection_rate.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)

    if events is not None and not events.empty:
        pivot = events.pivot_table(index="video_id", columns="metric", values="event_start_s", aggfunc="count", fill_value=0)
        plt.figure(figsize=(max(8, pivot.shape[1] * 0.85), max(5, pivot.shape[0] * 0.32)))
        plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        plt.colorbar(label="Candidate event count")
        plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=55, ha="right", fontsize=8)
        plt.yticks(range(pivot.shape[0]), pivot.index, fontsize=7)
        plt.tight_layout()
        path = fig_dir / "event_count_heatmap.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)

        plt.figure(figsize=(7, 4))
        events["duration_s"].dropna().hist(bins=30)
        plt.xlabel("Event duration (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        path = fig_dir / "event_duration_histogram.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)

    corr_cols = [c for c in video_metric_summary.columns if c.endswith("_mean") and c.replace("_mean", "") in METRIC_COLUMNS]
    if len(corr_cols) >= 2:
        corr = video_metric_summary[corr_cols].corr()
        plt.figure(figsize=(8, 7))
        plt.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(label="Correlation")
        labels = [c.replace("_mean", "") for c in corr_cols]
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        plt.yticks(range(len(labels)), labels, fontsize=7)
        plt.tight_layout()
        path = fig_dir / "metric_correlation_heatmap.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)

    if pca_scores is not None and not pca_scores.empty:
        plt.figure(figsize=(7, 5))
        clusters = pca_scores.get("cluster", pd.Series([-1] * len(pca_scores)))
        plt.scatter(pca_scores["pca1"], pca_scores["pca2"], c=clusters, cmap="tab10", s=55)
        for row in pca_scores.itertuples(index=False):
            plt.text(row.pca1, row.pca2, str(row.video_id), fontsize=7)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        path = fig_dir / "pca_cluster_plot.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figures.append(path)

    return figures


def _write_excel(out_dir: Path, tables: dict[str, pd.DataFrame | None]) -> Path:
    path = out_dir / "statistical_summary.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        wrote = False
        for name, table in tables.items():
            if table is None or table.empty:
                continue
            table.to_excel(writer, sheet_name=name[:31], index=False)
            wrote = True
        if not wrote:
            pd.DataFrame({"message": ["No statistical tables generated."]}).to_excel(writer, sheet_name="summary", index=False)
    return path


def _write_statistical_report(
    out_dir: Path,
    manifest: pd.DataFrame,
    quality: pd.DataFrame,
    events: pd.DataFrame,
    sniffs: pd.DataFrame,
    pca_scores: pd.DataFrame | None,
    figures: list[Path],
) -> Path:
    lines = [
        "# Statistical Micro-Expression Report",
        "",
        "This is an exploratory automatic analysis. It does not claim sweat/control effects unless a validated condition mapping is added.",
        "",
        "## Extraction",
        f"- Videos in manifest: {len(manifest)}",
        f"- Processed videos: {int(manifest['status'].isin(['processed', 'skipped_existing']).sum()) if 'status' in manifest else len(manifest)}",
    ]
    if "processed_frames" in manifest:
        total_frames = pd.to_numeric(manifest["processed_frames"], errors="coerce").fillna(0).sum()
        lines.append(f"- Processed frames listed in manifest: {int(total_frames)}")

    lines.extend(["", "## Quality"])
    if quality.empty:
        lines.append("- No quality table generated.")
    else:
        lines.append(f"- Mean face detection rate: {quality['face_detection_rate'].mean():.3f}")
        lines.append("- Quality tiers: " + ", ".join(f"{k}={v}" for k, v in quality["quality_tier"].value_counts().items()))

    lines.extend(["", "## Candidate Micro-Events"])
    if events.empty:
        lines.append("- No automatic candidate events detected.")
    else:
        lines.append(f"- Candidate events: {len(events)}")
        top = events["metric"].value_counts().head(10)
        lines.append("- Events by metric: " + ", ".join(f"{k}={v}" for k, v in top.items()))
        lines.append(f"- Median duration: {events['duration_s'].median():.3f}s")
        lines.append(f"- Median peak |z|: {events['peak_abs_z'].median():.2f}")

    lines.extend(["", "## Candidate Sniff-Like Windows"])
    if sniffs.empty:
        lines.append("- No sniff-like candidates generated.")
    else:
        lines.append(f"- Candidate sniff-like windows: {len(sniffs)}")
        lines.append("- These windows require manual confirmation before event-based claims.")

    lines.extend(["", "## PCA / Clustering"])
    if pca_scores is None or pca_scores.empty:
        lines.append("- PCA/clustering skipped because too few usable video summaries were available.")
    else:
        lines.append(f"- PCA rows: {len(pca_scores)}")
        lines.append("- Clusters are exploratory and unsupervised.")

    if figures:
        lines.extend(["", "## Figures"])
        for fig in figures:
            lines.append(f"- `{fig.relative_to(out_dir)}`")

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "- Automatic events are candidates, not final FACS annotations.",
            "- Use `auto_candidate_sniffs.csv` to prioritize manual review and fill `annotations.csv` with validated `sniff_onset_s` values.",
            "- After validated onsets and condition mapping are added, rerun event-based trial analysis for stronger inference.",
        ]
    )
    path = out_dir / "statistical_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def extract_all(
    video_dir: Path,
    out_dir: Path,
    target_fps: float = 100.0,
    include_deepface: bool = False,
    deepface_every_s: float = 1.0,
    include_duplicates: bool = False,
    video_ids: list[str] | None = None,
    limit_videos: int | None = None,
    max_width: int | None = 960,
    force: bool = False,
    max_frames: int | None = None,
) -> dict[str, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_inventory(out_dir, video_dir, include_duplicates, video_ids, limit_videos)
    if selected.empty:
        raise RuntimeError("No videos selected for extraction.")

    manifest_rows = []
    event_frames = []
    sniff_frames = []
    quality_rows = []

    for row in selected.to_dict("records"):
        result = extract_video_features(
            pd.Series(row),
            out_dir=out_dir,
            target_fps=target_fps,
            include_deepface=include_deepface,
            deepface_every_s=deepface_every_s,
            max_width=max_width,
            force=force,
            max_frames=max_frames,
        )
        manifest_rows.append(result)

        feature_path = Path(result["feature_path"])
        events, sniffs, quality = analyze_feature_file(feature_path, result["video_id"])
        if not events.empty:
            event_frames.append(events)
        if not sniffs.empty:
            sniff_frames.append(sniffs)
        quality_rows.append(quality)

    manifest = pd.DataFrame(manifest_rows)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    sniffs = pd.concat(sniff_frames, ignore_index=True) if sniff_frames else pd.DataFrame()
    quality = pd.DataFrame(quality_rows)
    video_metric_summary = quality[
        [col for col in quality.columns if col == "video_id" or col.endswith("_mean") or col.endswith("_std") or col.endswith("_activity_per_s")]
    ].copy()
    pca_scores = _pca_cluster(video_metric_summary, events)
    deepface_full = _load_deepface_samples(out_dir) if include_deepface else None

    manifest_path = out_dir / "frame_features_manifest.csv"
    events_path = out_dir / "auto_candidate_events.csv"
    sniffs_path = out_dir / "auto_candidate_sniffs.csv"
    quality_path = out_dir / "quality_full.csv"
    summary_path = out_dir / "video_metric_summary.csv"
    pca_path = out_dir / "pca_cluster_scores.csv"
    corr_path = out_dir / "metric_correlations.csv"

    manifest.to_csv(manifest_path, index=False)
    events.to_csv(events_path, index=False)
    sniffs.to_csv(sniffs_path, index=False)
    quality.to_csv(quality_path, index=False)
    video_metric_summary.to_csv(summary_path, index=False)
    if pca_scores is not None:
        pca_scores.to_csv(pca_path, index=False)
    else:
        pca_path = None

    corr_cols = [c for c in video_metric_summary.columns if c.endswith("_mean") and c.replace("_mean", "") in METRIC_COLUMNS]
    correlations = video_metric_summary[corr_cols].corr() if len(corr_cols) >= 2 else pd.DataFrame()
    correlations.to_csv(corr_path)

    figures = _write_figures(out_dir, quality, events, video_metric_summary, pca_scores)
    excel_path = _write_excel(
        out_dir,
        {
            "manifest": manifest,
            "quality_full": quality,
            "candidate_events": events,
            "candidate_sniffs": sniffs,
            "video_metric_summary": video_metric_summary,
            "pca_cluster_scores": pca_scores,
            "metric_correlations": correlations.reset_index().rename(columns={"index": "metric"}),
            "deepface_full": deepface_full,
        },
    )
    report_path = _write_statistical_report(out_dir, manifest, quality, events, sniffs, pca_scores, figures)

    return {
        "manifest": manifest_path,
        "events": events_path,
        "sniffs": sniffs_path,
        "quality": quality_path,
        "summary": summary_path,
        "correlations": corr_path,
        "pca": pca_path,
        "excel": excel_path,
        "report": report_path,
    }
