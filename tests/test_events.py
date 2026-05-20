from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from microexp.events import EventDetectionConfig, detect_micro_events, robust_baseline_z


def test_micro_event_detector_finds_short_event():
    times = np.arange(-1.0, 2.0, 0.01)
    values = np.zeros_like(times)
    values[(times >= 0.20) & (times < 0.36)] = 5.0
    baseline = (times >= -1.0) & (times <= -0.5)
    z = robust_baseline_z(values, baseline)
    # Flat baselines can make robust z undefined, so use direct z-like values for detector.
    z = values.copy()
    events = detect_micro_events(times, z, "synthetic", EventDetectionConfig(threshold_z=3.0))
    assert len(events) == 1
    duration = float(events.iloc[0]["duration_s"])
    assert 0.14 <= duration <= 0.18


def test_micro_event_detector_rejects_long_event():
    times = np.arange(-1.0, 2.0, 0.01)
    z = np.zeros_like(times)
    z[(times >= 0.10) & (times < 1.20)] = 4.0
    events = detect_micro_events(times, z, "synthetic", EventDetectionConfig(threshold_z=3.0))
    assert len(events) == 0


if __name__ == "__main__":
    test_micro_event_detector_finds_short_event()
    test_micro_event_detector_rejects_long_event()
    print("event tests passed")

