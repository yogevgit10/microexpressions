# Micro Expression Video Analysis Pipeline

This project analyzes the MP4 videos in:

`C:\Users\gitya\Downloads\Micro Exp Papers`

It follows the plan from `flfinal.docx`: blind, event-based analysis around manually or semi-manually coded sniff onsets, using OpenCV + MediaPipe landmarks as geometric proxies for key facial Action Units.

## What It Produces

All outputs are written under `microexp_analysis\results` by default:

- `inventory.csv`: video duration, fps, resolution, file hash, duplicate groups.
- `contact_sheets\*.jpg`: timestamped thumbnails for semi-manual onset coding.
- `annotations.csv`: two-trial annotation template per unique video.
- `trial_summary.csv`: per-trial features by time window, after annotations are filled.
- `micro_events.csv`: short 80-800 ms robust-z events per metric.
- `quality_report.csv`: face detection and head-pose quality indicators.
- `deepface_emotions.csv`: optional DeepFace emotion summaries by analysis window.
- `microexp_report.md`: readable analysis report.
- `microexp_summary.xlsx`: Excel workbook with available result tables.

Full automatic extraction also writes:

- `frame_features_100fps\*.csv`: per-video frame-level MediaPipe/AU-proxy time series.
- `auto_candidate_events.csv`: automatic micro-expression candidates, 80-800 ms.
- `auto_candidate_sniffs.csv`: sniff-like candidate windows that need manual confirmation.
- `quality_full.csv`: full-video quality metrics.
- `video_metric_summary.csv`: per-video statistical features.
- `statistical_report.md`: exploratory statistical report.
- `statistical_summary.xlsx`: workbook with quality, events, sniff candidates, PCA/clustering and correlations.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pipeline expects the MediaPipe Tasks model at:

`models\face_landmarker.task`

It has been downloaded into this project folder. If the folder is recreated, download the official MediaPipe Face Landmarker model again before running analysis.

## First Run

```powershell
.\.venv\Scripts\python.exe -m microexp.cli run-all `
  --video-dir "C:\Users\gitya\Downloads\Micro Exp Papers" `
  --out-dir ".\results"
```

The first run creates `inventory.csv`, contact sheets, and `annotations.csv`.

## Full Automatic Extraction

This extracts MediaPipe/AU-proxy features at 100fps for the representative videos. DeepFace, when enabled, is sampled more slowly because it is a broad emotion classifier and is much heavier than the landmark model.

Smoke test on the short video:

```powershell
.\.venv\Scripts\python.exe -m microexp.cli extract-all `
  --video-dir "C:\Users\gitya\Downloads\Micro Exp Papers" `
  --out-dir ".\results" `
  --video-id "2026-01-08_10-25-28" `
  --target-fps 100 `
  --deepface `
  --force
```

Full run on all representative videos:

```powershell
.\.venv\Scripts\python.exe -m microexp.cli extract-all `
  --video-dir "C:\Users\gitya\Downloads\Micro Exp Papers" `
  --out-dir ".\results" `
  --target-fps 100 `
  --deepface
```

Notes:

- Exact duplicates are skipped by default; use `--include-duplicates` only if you intentionally want duplicate files processed too.
- Frames are resized to width 960 before model inference by default for speed; use `--max-width 0` to process original resolution.
- Automatic events and sniff windows are candidates, not final human-confirmed annotations.

## Annotation Workflow

Open `results\annotations.csv` and fill at least:

- `approach_onset_s`: when the tube starts approaching the nose.
- `sniff_onset_s`: effective sniff onset. This is required for analysis.
- `sniff_offset_s`: optional but recommended.
- `response_s`: optional response/choice time.
- `quality_flag`: use `ok`, `exclude`, or a short status such as `occluded`.
- `occlusion_notes`: note hands, tube, self-touch, talking, glasses adjustment, etc.

Do not encode sweat/control condition in this file unless you have a separate randomization table. The pipeline intentionally remains blind.

After filling annotations:

```powershell
.\.venv\Scripts\python.exe -m microexp.cli analyze --out-dir ".\results"
.\.venv\Scripts\python.exe -m microexp.cli report --out-dir ".\results"
```

To include optional DeepFace emotion summaries:

```powershell
.\.venv\Scripts\python.exe -m microexp.cli analyze --out-dir ".\results" --deepface
.\.venv\Scripts\python.exe -m microexp.cli report --out-dir ".\results"
```

DeepFace is used only for broad emotion predictions (`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`). The AU-style micro-expression proxies still come from MediaPipe landmarks.

## Optional Condition Mapping

If a randomization table becomes available, save it as `results\condition_mapping.csv` with columns:

`video_id,trial_id,condition`

where `condition` is `sweat` or `control`. Then rerun `report`; it will generate `condition_comparison.csv` with paired t-test and Wilcoxon summaries where possible.

## Smoke Tests

```powershell
.\.venv\Scripts\python.exe tests\test_events.py
.\.venv\Scripts\python.exe -m microexp.cli smoke-test `
  --video-dir "C:\Users\gitya\Downloads\Micro Exp Papers" `
  --out-dir ".\results"
.\.venv\Scripts\python.exe -m microexp.cli smoke-test `
  --video-dir "C:\Users\gitya\Downloads\Micro Exp Papers" `
  --out-dir ".\results" `
  --deepface
```
