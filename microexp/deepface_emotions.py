from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


EMOTION_COLUMNS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def deepface_home() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "deepface"


class DeepFaceEmotionExtractor:
    """Thin wrapper that keeps DeepFace optional and locally cached."""

    def __init__(self, detector_backend: str = "opencv") -> None:
        os.environ.setdefault("DEEPFACE_HOME", str(deepface_home()))
        deepface_home().mkdir(parents=True, exist_ok=True)
        try:
            from deepface import DeepFace  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "DeepFace is not installed. Install it with: "
                "python -m pip install deepface"
            ) from exc
        self._deepface = DeepFace
        self.detector_backend = detector_backend

    def analyze_bgr(self, frame) -> dict:
        try:
            result = self._deepface.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True,
            )
        except Exception as exc:  # DeepFace can fail on individual frames.
            row = {
                "deepface_ok": 0,
                "deepface_error": str(exc)[:500],
                "deepface_dominant_emotion": "",
            }
            for emotion in EMOTION_COLUMNS:
                row[f"deepface_{emotion}"] = np.nan
            return row

        if isinstance(result, list):
            result = result[0] if result else {}
        emotions = result.get("emotion", {}) if isinstance(result, dict) else {}
        row = {
            "deepface_ok": 1,
            "deepface_error": "",
            "deepface_dominant_emotion": result.get("dominant_emotion", "") if isinstance(result, dict) else "",
        }
        for emotion in EMOTION_COLUMNS:
            row[f"deepface_{emotion}"] = float(emotions.get(emotion, np.nan))
        return row


def summarize_deepface_windows(row: pd.Series, emotion_frames: pd.DataFrame, windows: dict[str, tuple[float, float]]) -> list[dict]:
    rows: list[dict] = []
    if emotion_frames.empty:
        return rows

    for window_name, (start, end) in windows.items():
        win = emotion_frames[emotion_frames["t_rel_s"].between(start, end, inclusive="left")]
        summary = {
            "video_id": row["video_id"],
            "trial_id": int(row["trial_id"]),
            "window": window_name,
            "window_start_rel_s": start,
            "window_end_rel_s": end,
            "deepface_sampled_frames": int(len(win)),
            "deepface_success_rate": float(win["deepface_ok"].mean()) if len(win) else np.nan,
            "deepface_dominant_emotion_mode": "",
        }
        valid = win[win["deepface_ok"].eq(1)]
        if not valid.empty and "deepface_dominant_emotion" in valid:
            modes = valid["deepface_dominant_emotion"].dropna().astype(str)
            if not modes.empty:
                summary["deepface_dominant_emotion_mode"] = modes.mode().iloc[0]
        for emotion in EMOTION_COLUMNS:
            col = f"deepface_{emotion}"
            values = pd.to_numeric(valid.get(col), errors="coerce")
            summary[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            summary[f"{col}_max"] = float(values.max()) if values.notna().any() else np.nan
        rows.append(summary)
    return rows

