# Project Inputs

Open this folder in VS Code:

`C:\Users\gitya\Documents\Codex\2026-04-28\files-mentioned-by-the-user-micro\microexp_analysis`

## Source Files

- Videos: `C:\Users\gitya\Downloads\Micro Exp Papers`
- Word file used for method/context: `C:\Users\gitya\Downloads\flfinal.docx`
- MediaPipe model: `models\face_landmarker.task`

The MP4 files are not copied into this project folder because they are large. The pipeline reads them from the original Downloads folder and writes all derived outputs into `results`.

## Main Outputs

- `results\inventory.csv`
- `results\annotations.csv`
- `results\contact_sheets`
- `results\microexp_report.md`
- `results\microexp_summary.xlsx`

## VS Code Tasks

Use `Terminal -> Run Task...` and choose:

- `MicroExp: smoke test`
- `MicroExp: smoke test + DeepFace`
- `MicroExp: run all setup`
- `MicroExp: extract short video 100fps`
- `MicroExp: extract all 100fps`
- `MicroExp: analyze annotated trials`
- `MicroExp: analyze annotated trials + DeepFace`
- `MicroExp: rebuild report`
- `MicroExp: event tests`

Before running `MicroExp: analyze annotated trials`, fill `results\annotations.csv` with `sniff_onset_s` values.

`MicroExp: extract all 100fps` can take a long time because it processes about 75 minutes of representative video frame-by-frame.
