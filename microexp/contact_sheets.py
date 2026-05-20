from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inventory import require_cv2


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def _labeled_thumbnail(cv2, frame, label: str, width: int):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid frame dimensions")
    new_h = int(round(h * (width / w)))
    thumb = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
    banner_h = 34
    banner = np.zeros((banner_h, width, 3), dtype=np.uint8)
    cv2.putText(
        banner,
        label,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, thumb])


def make_contact_sheet(
    video_path: Path,
    output_path: Path,
    every_s: float = 5.0,
    thumb_width: int = 320,
    cols: int = 4,
) -> Path:
    cv2 = require_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    times = list(np.arange(0.0, max(duration, every_s), every_s))
    thumbs = []

    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ok, frame = cap.read()
        if not ok:
            continue
        thumbs.append(_labeled_thumbnail(cv2, frame, _format_time(float(t)), thumb_width))

    cap.release()
    if not thumbs:
        raise RuntimeError(f"No frames could be sampled from {video_path}")

    cell_h = max(img.shape[0] for img in thumbs)
    cell_w = thumb_width
    padded = []
    for img in thumbs:
        pad_h = cell_h - img.shape[0]
        if pad_h:
            img = np.vstack([img, np.zeros((pad_h, cell_w, 3), dtype=np.uint8)])
        padded.append(img)

    rows = []
    total_rows = int(math.ceil(len(padded) / cols))
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    for row_idx in range(total_rows):
        row_imgs = padded[row_idx * cols : (row_idx + 1) * cols]
        while len(row_imgs) < cols:
            row_imgs.append(blank.copy())
        rows.append(np.hstack(row_imgs))

    sheet = np.vstack(rows)
    ok = cv2.imwrite(str(output_path), sheet)
    if not ok:
        raise RuntimeError(f"Could not write contact sheet: {output_path}")
    return output_path


def build_contact_sheets(
    inventory: pd.DataFrame,
    out_dir: Path,
    every_s: float = 5.0,
    representative_only: bool = True,
) -> list[Path]:
    sheets_dir = out_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    df = inventory.copy()
    if representative_only and "representative" in df.columns:
        df = df[df["representative"].astype(bool)]

    outputs = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="contact sheets"):
        video_id = getattr(row, "video_id")
        source_path = Path(getattr(row, "source_path"))
        output_path = sheets_dir / f"{video_id}.jpg"
        outputs.append(make_contact_sheet(source_path, output_path, every_s=every_s))
    return outputs

