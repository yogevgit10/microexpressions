"""Run trial-level analysis from previously extracted 100 fps frame features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT / "microexpressions_repo"
RESULTS_DIR = REPO_DIR / "results"
sys.path.insert(0, str(REPO_DIR))

from microexp.analyze import WINDOWS  # noqa: E402
from microexp.deepface_emotions import summarize_deepface_windows  # noqa: E402
from microexp.events import EventDetectionConfig, detect_micro_events, robust_baseline_z  # noqa: E402
from microexp.facial_metrics import KEY_EVENT_METRICS, METRIC_COLUMNS  # noqa: E402


def auc_abs(times: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(np.abs(values[valid]), times[valid]))
    return float(np.trapz(np.abs(values[valid]), times[valid]))  # type: ignore[attr-defined]


def add_baseline_normalization(frame_metrics: pd.DataFrame) -> pd.DataFrame:
    df = frame_metrics.copy()
    baseline_mask = df["t_rel_s"].between(-1.0, -0.5, inclusive="both") & (df["face_detected"] == 1)
    baseline_array = baseline_mask.to_numpy()
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        baseline_values = values[baseline_array]
        if np.isfinite(baseline_values).sum() >= 3:
            center = float(np.nanmedian(baseline_values))
            df[f"{metric}_delta"] = values - center
        else:
            df[f"{metric}_delta"] = np.nan
        df[f"{metric}_z"] = robust_baseline_z(values, baseline_array)
    return df


def summarize_trial(row: pd.Series, metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for window_name, (start, end) in WINDOWS.items():
        win = metrics[metrics["t_rel_s"].between(start, end, inclusive="left")]
        summary: dict[str, object] = {
            "video_id": row["video_id"],
            "trial_id": int(row["trial_id"]),
            "condition": row.get("condition", ""),
            "window": window_name,
            "window_start_rel_s": start,
            "window_end_rel_s": end,
            "n_frames": int(len(win)),
            "face_detection_rate": float(win["face_detected"].mean()) if len(win) else np.nan,
            "quality_flag": row.get("quality_flag", ""),
            "occlusion_notes": row.get("occlusion_notes", ""),
        }
        for metric in METRIC_COLUMNS:
            raw = pd.to_numeric(win.get(metric), errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(win.get(f"{metric}_delta"), errors="coerce").to_numpy(dtype=float)
            z = pd.to_numeric(win.get(f"{metric}_z"), errors="coerce").to_numpy(dtype=float)
            times = pd.to_numeric(win.get("t_rel_s"), errors="coerce").to_numpy(dtype=float)
            summary[f"{metric}_raw_mean"] = float(np.nanmean(raw)) if np.isfinite(raw).any() else np.nan
            summary[f"{metric}_delta_mean"] = float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan
            summary[f"{metric}_z_peak_abs"] = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else np.nan
            summary[f"{metric}_delta_auc_abs"] = auc_abs(times, delta)
        rows.append(summary)
    return rows


def quality_row(row: pd.Series, metrics: pd.DataFrame) -> dict[str, object]:
    face_rate = float(metrics["face_detected"].mean()) if len(metrics) else np.nan
    baseline = metrics[metrics["t_rel_s"].between(-1.0, -0.5, inclusive="both")]
    yaw = pd.to_numeric(metrics.get("head_yaw_proxy"), errors="coerce").to_numpy(dtype=float)
    roll = pd.to_numeric(metrics.get("head_roll_deg"), errors="coerce").to_numpy(dtype=float)
    return {
        "video_id": row["video_id"],
        "trial_id": int(row["trial_id"]),
        "condition": row.get("condition", ""),
        "n_processed_frames": int(len(metrics)),
        "face_detection_rate": face_rate,
        "baseline_frames": int(len(baseline)),
        "baseline_face_frames": int((baseline["face_detected"] == 1).sum()) if len(baseline) else 0,
        "mean_abs_head_yaw_proxy": float(np.nanmean(np.abs(yaw))) if np.isfinite(yaw).any() else np.nan,
        "max_abs_head_yaw_proxy": float(np.nanmax(np.abs(yaw))) if np.isfinite(yaw).any() else np.nan,
        "mean_abs_head_roll_deg": float(np.nanmean(np.abs(roll))) if np.isfinite(roll).any() else np.nan,
        "max_abs_head_roll_deg": float(np.nanmax(np.abs(roll))) if np.isfinite(roll).any() else np.nan,
        "quality_flag": row.get("quality_flag", ""),
        "occlusion_notes": row.get("occlusion_notes", ""),
    }


def event_rows(row: pd.Series, metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = EventDetectionConfig()
    times = metrics["t_rel_s"].to_numpy(dtype=float)
    for metric in KEY_EVENT_METRICS:
        events = detect_micro_events(times, metrics[f"{metric}_z"].to_numpy(dtype=float), metric, config=config)
        for event in events.to_dict("records"):
            event.update(
                {
                    "video_id": row["video_id"],
                    "trial_id": int(row["trial_id"]),
                    "condition": row.get("condition", ""),
                }
            )
            rows.append(event)
    return rows


def load_primary_annotations() -> pd.DataFrame:
    annotations = pd.read_csv(RESULTS_DIR / "annotations.csv")
    mapping = pd.read_csv(RESULTS_DIR / "condition_mapping.csv")
    mapping = mapping[["video_id", "trial_id", "condition"]].copy()
    mapping["trial_id"] = pd.to_numeric(mapping["trial_id"], errors="coerce").astype("Int64")
    annotations["trial_id"] = pd.to_numeric(annotations["trial_id"], errors="coerce").astype("Int64")
    merged = annotations.merge(mapping, on=["video_id", "trial_id"], how="left")
    ready = merged[pd.to_numeric(merged["sniff_onset_s"], errors="coerce").notna()].copy()
    ready = ready[ready["quality_flag"].fillna("").astype(str).str.lower().eq("ok")]
    return ready.sort_values(["video_id", "trial_id"]).reset_index(drop=True)


def main() -> None:
    annotations = load_primary_annotations()
    if annotations.empty:
        raise RuntimeError("No primary two-test-tube annotations are ready.")

    frame_dir = RESULTS_DIR / "frame_features_100fps"
    trial_frame_dir = RESULTS_DIR / "frame_metrics"
    trial_frame_dir.mkdir(parents=True, exist_ok=True)

    deepface_full_path = RESULTS_DIR / "deepface_full.csv"
    deepface_full = pd.read_csv(deepface_full_path) if deepface_full_path.exists() else pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    micro_event_rows: list[dict[str, object]] = []
    deepface_rows: list[dict[str, object]] = []

    for row in annotations.to_dict("records"):
        row_s = pd.Series(row)
        video_id = str(row_s["video_id"])
        trial_id = int(row_s["trial_id"])
        sniff_onset = float(row_s["sniff_onset_s"])
        feature_path = frame_dir / f"{video_id}.csv"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        features = pd.read_csv(feature_path)
        features["timestamp_s"] = pd.to_numeric(features["timestamp_s"], errors="coerce")
        features["t_rel_s"] = features["timestamp_s"] - sniff_onset
        trial_metrics = features[features["t_rel_s"].between(-1.0, 5.0, inclusive="both")].copy()
        trial_metrics = add_baseline_normalization(trial_metrics)
        for metadata_col in ("video_id", "trial_id", "condition"):
            if metadata_col in trial_metrics.columns:
                trial_metrics = trial_metrics.drop(columns=[metadata_col])
        trial_metrics.insert(0, "condition", row_s["condition"])
        trial_metrics.insert(0, "trial_id", trial_id)
        trial_metrics.insert(0, "video_id", video_id)
        trial_metrics.to_csv(trial_frame_dir / f"{video_id}_trial_{trial_id}.csv", index=False)

        summary_rows.extend(summarize_trial(row_s, trial_metrics))
        quality_rows.append(quality_row(row_s, trial_metrics))
        micro_event_rows.extend(event_rows(row_s, trial_metrics))

        if not deepface_full.empty:
            emotion_frames = deepface_full[deepface_full["video_id"].astype(str).eq(video_id)].copy()
            if not emotion_frames.empty:
                emotion_frames["timestamp_s"] = pd.to_numeric(emotion_frames["timestamp_s"], errors="coerce")
                emotion_frames["t_rel_s"] = emotion_frames["timestamp_s"] - sniff_onset
                emotion_frames = emotion_frames[emotion_frames["t_rel_s"].between(-1.0, 5.0, inclusive="both")]
            deepface_rows.extend(summarize_deepface_windows(row_s, emotion_frames, WINDOWS))

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "trial_summary.csv", index=False)
    pd.DataFrame(micro_event_rows).to_csv(RESULTS_DIR / "micro_events.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(RESULTS_DIR / "quality_report.csv", index=False)
    pd.DataFrame(deepface_rows).to_csv(RESULTS_DIR / "deepface_emotions.csv", index=False)

    print(f"Primary annotations analyzed: {len(annotations)}")
    print(f"Paired videos: {annotations['video_id'].nunique()}")
    print(f"Trial summary rows: {len(summary_rows)}")
    print(f"Micro-events: {len(micro_event_rows)}")
    print(f"Quality rows: {len(quality_rows)}")
    print(f"DeepFace summary rows: {len(deepface_rows)}")


if __name__ == "__main__":
    main()
