from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventDetectionConfig:
    threshold_z: float = 3.0
    min_duration_s: float = 0.08
    max_duration_s: float = 0.80
    max_gap_s: float = 0.05
    min_start_s: float = 0.0


def robust_baseline_z(values, baseline_mask) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    baseline = values[baseline_mask & np.isfinite(values)]
    if baseline.size < 3:
        return np.full_like(values, np.nan, dtype=float)
    center = float(np.nanmedian(baseline))
    mad = float(np.nanmedian(np.abs(baseline - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        std = float(np.nanstd(baseline))
        scale = std if std > 1e-9 else np.nan
    if not np.isfinite(scale):
        return np.full_like(values, np.nan, dtype=float)
    return (values - center) / scale


def _median_dt(times: np.ndarray) -> float:
    finite = times[np.isfinite(times)]
    if finite.size < 2:
        return 0.0
    diffs = np.diff(np.sort(finite))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.nanmedian(diffs))


def detect_micro_events(
    times_s,
    z_values,
    metric: str,
    config: EventDetectionConfig | None = None,
) -> pd.DataFrame:
    config = config or EventDetectionConfig()
    times = np.asarray(times_s, dtype=float)
    z = np.asarray(z_values, dtype=float)
    active = (
        np.isfinite(times)
        & np.isfinite(z)
        & (times >= config.min_start_s)
        & (np.abs(z) >= config.threshold_z)
    )
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return pd.DataFrame()

    dt = _median_dt(times)
    groups: list[list[int]] = []
    current = [int(active_indices[0])]
    for idx in active_indices[1:]:
        prev = current[-1]
        if times[idx] - times[prev] <= config.max_gap_s:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    groups.append(current)

    rows = []
    for group in groups:
        start = float(times[group[0]])
        end = float(times[group[-1]] + dt)
        duration = end - start
        if duration < config.min_duration_s or duration > config.max_duration_s:
            continue
        group_z = z[group]
        peak_abs = float(np.nanmax(np.abs(group_z)))
        peak_signed = float(group_z[int(np.nanargmax(np.abs(group_z)))])
        rows.append(
            {
                "metric": metric,
                "event_start_rel_s": start,
                "event_end_rel_s": end,
                "duration_s": duration,
                "peak_abs_z": peak_abs,
                "peak_signed_z": peak_signed,
                "n_frames": len(group),
            }
        )
    return pd.DataFrame(rows)

