# Micro-Expression Analysis Report

This report uses `condition_mapping.csv` for paired sweat/control comparisons.

## Inventory
- Videos listed: 19
- Representative videos for analysis: 15
- Files in exact-duplicate groups: 8
- Non-representative duplicate files excluded from analysis: 4
- Total listed duration: 84.2 minutes

## Annotation Status
- Annotation rows: 14
- Trials with `sniff_onset_s`: 14
- Excluded trials: 0

## Trial-Level Outputs
- Summary rows: 56
- Micro-events detected: 169
- Events by metric: au24_lip_press_proxy=44, cheek_composite_proxy=26, au12_lip_corner_pull_proxy=25, au06_cheek_raise_proxy=24, au45_blink_proxy=18, au10_upper_lip_raise_proxy=17, au09_nose_wrinkle_proxy=15
- Mean face detection rate: 0.986
- DeepFace emotion summary rows: 56
- DeepFace dominant modes: fear=27, sad=13, neutral=13, happy=2, angry=1

## Condition Mapping
- Paired sweat/control comparison rows: 132
- late_2_5s / cheek_composite_proxy_delta_auc_abs: n=7, mean diff=-0.03651, dz=-1.44, paired t p=0.008997
- early_0_500ms / head_roll_deg_delta_mean: n=7, mean diff=5.234, dz=0.94, paired t p=0.04739
- short_500_2000ms / head_roll_deg_delta_mean: n=7, mean diff=5.17, dz=0.822, paired t p=0.07249
- early_0_500ms / au10_upper_lip_raise_proxy_delta_mean: n=7, mean diff=0.04467, dz=0.803, paired t p=0.07774
- short_500_2000ms / mouth_asymmetry_proxy_delta_auc_abs: n=7, mean diff=-0.04706, dz=-0.789, paired t p=0.08198

## Figures
- `figures\face_detection_rate.png`
- `figures\early_peak_abs_z.png`

## Workbook
- `microexp_summary.xlsx`
