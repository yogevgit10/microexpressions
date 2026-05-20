from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .events import EventDetectionConfig, detect_micro_events, robust_baseline_z
from .facial_metrics import FaceMetricExtractor, KEY_EVENT_METRICS, METRIC_COLUMNS
from .inventory import load_inventory, require_cv2


WINDOWS = {
    "baseline_-1000_-500ms": (-1.0, -0.5),
    "early_0_500ms": (0.0, 0.5),
    "short_500_2000ms": (0.5, 2.0),
    "late_2_5s": (2.0, 5.0),
}


ANNOTATION_COLUMNS = [
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


def create_annotations_template(inventory: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = inventory.copy()
    if "representative" in df.columns:
        df = df[df["representative"].astype(bool)]

    rows = []
    for row in df.itertuples(index=False):
        for trial_id in (1, 2):
            rows.append(
                {
                    "video_id": getattr(row, "video_id"),
                    "source_path": getattr(row, "source_path"),
                    "trial_id": trial_id,
                    "approach_onset_s": "",
                    "sniff_onset_s": "",
                    "sniff_offset_s": "",
                    "response_s": "",
                    "occlusion_notes": "",
                    "quality_flag": "pending",
                }
            )
    annotations = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(out_path, index=False)
    return annotations


def load_annotations(path: Path) -> pd.DataFrame:
    annotations = pd.read_csv(path)
    for col in ANNOTATION_COLUMNS:
        if col not in annotations.columns:
            annotations[col] = ""
    for col in ["trial_id", "approach_onset_s", "sniff_onset_s", "sniff_offset_s", "response_s"]:
        annotations[col] = pd.to_numeric(annotations[col], errors="coerce")
    annotations["quality_flag"] = annotations["quality_flag"].fillna("").astype(str)
    annotations["occlusion_notes"] = annotations["occlusion_notes"].fillna("").astype(str)
    return annotations[ANNOTATION_COLUMNS]


def annotations_ready(annotations: pd.DataFrame) -> bool:
    return bool(pd.to_numeric(annotations["sniff_onset_s"], errors="coerce").notna().any())


def _trial_bounds(row: pd.Series) -> tuple[float, float]:
    sniff = float(row["sniff_onset_s"])
    start = max(0.0, sniff - 1.0)
    response = row.get("response_s", np.nan)
    end = sniff + 5.0
    if pd.notna(response) and float(response) > sniff:
        end = min(end, float(response))
    return start, end


def _read_trial_metrics(
    video_path: Path,
    row: pd.Series,
    include_deepface: bool = False,
    deepface_every_s: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS for video: {video_path}")

    start_s, end_s = _trial_bounds(row)
    start_frame = max(0, int(np.floor(start_s * fps)))
    end_frame = max(start_frame, int(np.ceil(end_s * fps)))
    sniff_onset = float(row["sniff_onset_s"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    extractor = FaceMetricExtractor()
    deepface_extractor = None
    if include_deepface:
        from .deepface_emotions import DeepFaceEmotionExtractor

        deepface_extractor = DeepFaceEmotionExtractor()
    rows = []
    deepface_rows = []
    next_deepface_sample_s = start_s
    try:
        frame_idx = start_frame
        with tqdm(total=end_frame - start_frame + 1, desc=f"{row['video_id']} t{int(row['trial_id'])}", leave=False) as bar:
            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_s = frame_idx / fps
                metric_row = extractor.process_bgr(cv2, frame, frame_idx, timestamp_s)
                metric_row["t_rel_s"] = timestamp_s - sniff_onset
                rows.append(metric_row)
                if deepface_extractor is not None and timestamp_s + 1e-9 >= next_deepface_sample_s:
                    emotion_row = deepface_extractor.analyze_bgr(frame)
                    emotion_row.update(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_s": timestamp_s,
                            "t_rel_s": timestamp_s - sniff_onset,
                        }
                    )
                    deepface_rows.append(emotion_row)
                    next_deepface_sample_s += max(deepface_every_s, 1.0 / fps)
                frame_idx += 1
                bar.update(1)
    finally:
        extractor.close()
        cap.release()

    return pd.DataFrame(rows), pd.DataFrame(deepface_rows)


def _add_baseline_normalization(frame_metrics: pd.DataFrame) -> pd.DataFrame:
    df = frame_metrics.copy()
    baseline_mask = df["t_rel_s"].between(-1.0, -0.5, inclusive="both") & (df["face_detected"] == 1)
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        baseline_values = values[baseline_mask.to_numpy()]
        if np.isfinite(baseline_values).sum() >= 3:
            center = float(np.nanmedian(baseline_values))
            df[f"{metric}_delta"] = values - center
        else:
            df[f"{metric}_delta"] = np.nan
        df[f"{metric}_z"] = robust_baseline_z(values, baseline_mask.to_numpy())
    return df


def _auc_abs(times: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapz(np.abs(values[valid]), times[valid]))


def _summarize_trial(row: pd.Series, metrics: pd.DataFrame) -> list[dict]:
    rows = []
    for window_name, (start, end) in WINDOWS.items():
        mask = metrics["t_rel_s"].between(start, end, inclusive="left")
        win = metrics[mask]
        summary = {
            "video_id": row["video_id"],
            "trial_id": int(row["trial_id"]),
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
            summary[f"{metric}_delta_auc_abs"] = _auc_abs(times, delta)
        rows.append(summary)
    return rows


def _quality_row(row: pd.Series, metrics: pd.DataFrame) -> dict:
    face_rate = float(metrics["face_detected"].mean()) if len(metrics) else np.nan
    baseline = metrics[metrics["t_rel_s"].between(-1.0, -0.5, inclusive="both")]
    yaw = pd.to_numeric(metrics.get("head_yaw_proxy"), errors="coerce").to_numpy(dtype=float)
    roll = pd.to_numeric(metrics.get("head_roll_deg"), errors="coerce").to_numpy(dtype=float)
    return {
        "video_id": row["video_id"],
        "trial_id": int(row["trial_id"]),
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


def _event_rows(row: pd.Series, metrics: pd.DataFrame) -> list[dict]:
    rows = []
    config = EventDetectionConfig()
    times = metrics["t_rel_s"].to_numpy(dtype=float)
    for metric in KEY_EVENT_METRICS:
        events = detect_micro_events(times, metrics[f"{metric}_z"].to_numpy(dtype=float), metric, config=config)
        for event in events.to_dict("records"):
            event.update({"video_id": row["video_id"], "trial_id": int(row["trial_id"])})
            rows.append(event)
    return rows


def analyze_trials(
    out_dir: Path,
    annotations_path: Path | None = None,
    include_deepface: bool = False,
    deepface_every_s: float = 0.25,
) -> dict[str, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = annotations_path or (out_dir / "annotations.csv")
    inventory_path = out_dir / "inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing inventory: {inventory_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotations_path}")

    annotations = load_annotations(annotations_path)
    ready = annotations[pd.to_numeric(annotations["sniff_onset_s"], errors="coerce").notna()].copy()
    ready = ready[~ready["quality_flag"].str.lower().eq("exclude")]
    if ready.empty:
        return {
            "trial_summary": None,
            "micro_events": None,
            "quality_report": None,
            "deepface_emotions": None,
            "message": "No annotated trials with sniff_onset_s were found.",
        }

    frame_dir = out_dir / "frame_metrics"
    deepface_frame_dir = out_dir / "deepface_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if include_deepface:
        deepface_frame_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    quality_rows: list[dict] = []
    event_rows: list[dict] = []
    deepface_rows: list[dict] = []

    for row in tqdm(list(ready.to_dict("records")), desc="analyzing trials"):
        row_s = pd.Series(row)
        video_path = Path(str(row_s["source_path"]))
        metrics, emotion_frames = _read_trial_metrics(
            video_path,
            row_s,
            include_deepface=include_deepface,
            deepface_every_s=deepface_every_s,
        )
        metrics = _add_baseline_normalization(metrics)
        trial_id = int(row_s["trial_id"])
        metrics.insert(0, "trial_id", trial_id)
        metrics.insert(0, "video_id", row_s["video_id"])
        metrics.to_csv(frame_dir / f"{row_s['video_id']}_trial_{trial_id}.csv", index=False)
        if include_deepface:
            if not emotion_frames.empty:
                emotion_frames.insert(0, "trial_id", trial_id)
                emotion_frames.insert(0, "video_id", row_s["video_id"])
            emotion_frames.to_csv(deepface_frame_dir / f"{row_s['video_id']}_trial_{trial_id}.csv", index=False)
            from .deepface_emotions import summarize_deepface_windows

            deepface_rows.extend(summarize_deepface_windows(row_s, emotion_frames, WINDOWS))
        summary_rows.extend(_summarize_trial(row_s, metrics))
        quality_rows.append(_quality_row(row_s, metrics))
        event_rows.extend(_event_rows(row_s, metrics))

    trial_summary = pd.DataFrame(summary_rows)
    micro_events = pd.DataFrame(event_rows)
    quality_report = pd.DataFrame(quality_rows)
    deepface_emotions = pd.DataFrame(deepface_rows)

    trial_summary_path = out_dir / "trial_summary.csv"
    micro_events_path = out_dir / "micro_events.csv"
    quality_report_path = out_dir / "quality_report.csv"
    deepface_emotions_path = out_dir / "deepface_emotions.csv" if include_deepface else None
    trial_summary.to_csv(trial_summary_path, index=False)
    micro_events.to_csv(micro_events_path, index=False)
    quality_report.to_csv(quality_report_path, index=False)
    if deepface_emotions_path is not None:
        deepface_emotions.to_csv(deepface_emotions_path, index=False)

    return {
        "trial_summary": trial_summary_path,
        "micro_events": micro_events_path,
        "quality_report": quality_report_path,
        "deepface_emotions": deepface_emotions_path,
        "message": "Analysis complete.",
    }
