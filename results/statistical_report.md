# Statistical Micro-Expression Report

This is an exploratory automatic analysis. It does not claim sweat/control effects unless a validated condition mapping is added.

## Extraction
- Videos in manifest: 15
- Processed videos: 15
- Processed frames listed in manifest: 450805

## Quality
- Mean face detection rate: 0.939
- Quality tiers: good=13, review_or_exclude=1, usable_review=1

## Candidate Micro-Events
- Candidate events: 3614
- Events by metric: au45_blink_proxy=643, au06_cheek_raise_proxy=423, au09_nose_wrinkle_proxy=415, cheek_composite_proxy=378, au10_upper_lip_raise_proxy=324, au24_lip_press_proxy=302, mouth_open_proxy=302, head_yaw_proxy=298, mouth_asymmetry_proxy=269, au12_lip_corner_pull_proxy=260
- Median duration: 0.150s
- Median peak |z|: 4.14

## Candidate Sniff-Like Windows
- Candidate sniff-like windows: 114
- These windows require manual confirmation before event-based claims.

## PCA / Clustering
- PCA rows: 15
- Clusters are exploratory and unsupervised.

## Figures
- `statistical_figures\quality_face_detection_rate.png`
- `statistical_figures\event_count_heatmap.png`
- `statistical_figures\event_duration_histogram.png`
- `statistical_figures\metric_correlation_heatmap.png`
- `statistical_figures\pca_cluster_plot.png`

## Interpretation Boundary
- Automatic events are candidates, not final FACS annotations.
- Use `auto_candidate_sniffs.csv` to prioritize manual review and fill `annotations.csv` with validated `sniff_onset_s` values.
- After validated onsets and condition mapping are added, rerun event-based trial analysis for stronger inference.
