from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return cv2


def stable_video_id(path: Path) -> str:
    text = path.stem.strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def iter_video_files(video_dir: Path) -> Iterable[Path]:
    for path in sorted(video_dir.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "fps": None,
            "frame_count": None,
            "duration_s": None,
            "width": None,
            "height": None,
            "readable": False,
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_s = frame_count / fps if fps > 0 else None
    return {
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count or None,
        "duration_s": duration_s,
        "width": width or None,
        "height": height or None,
        "readable": True,
    }


def build_inventory(video_dir: Path, out_dir: Path, hash_files: bool = True) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in iter_video_files(video_dir):
        stat = path.stat()
        row = {
            "video_id": stable_video_id(path),
            "file_name": path.name,
            "source_path": str(path),
            "size_bytes": stat.st_size,
            "modified_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
        }
        row.update(video_metadata(path))
        row["sha256"] = sha256_file(path) if hash_files else None
        rows.append(row)

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory.to_csv(out_dir / "inventory.csv", index=False)
        return inventory

    inventory["duplicate_group"] = ""
    inventory["is_duplicate"] = False
    inventory["representative"] = True

    if hash_files and "sha256" in inventory:
        duplicate_hashes = (
            inventory.groupby("sha256", dropna=True)
            .filter(lambda group: len(group) > 1)["sha256"]
            .dropna()
            .unique()
        )
        for idx, digest in enumerate(sorted(duplicate_hashes), start=1):
            mask = inventory["sha256"] == digest
            group_name = f"dup_{idx:02d}"
            group_indices = inventory.index[mask].tolist()
            inventory.loc[mask, "duplicate_group"] = group_name
            inventory.loc[mask, "is_duplicate"] = True
            inventory.loc[group_indices[1:], "representative"] = False

    inventory.to_csv(out_dir / "inventory.csv", index=False)
    return inventory


def load_inventory(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

