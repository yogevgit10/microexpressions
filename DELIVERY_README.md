# Micro-Expressions Project Delivery

Generated: 2026-06-16

This package contains the current project outputs for the facial micro-expression
analysis project. It intentionally excludes source videos, virtual environments,
caches, model weights, and private/contact-sheet style visual material.

## Main Contents

- `microexpressions_repo/results/`
  - Full analysis outputs, including frame-level feature CSVs, trial-level
    summaries, quality reports, condition mapping, paired sweat/control
    comparisons, and summary workbooks.
- `article_draft/`
  - Updated article sources and outputs:
    - `main.tex`
    - `main.pdf`
    - `microexpressions_article_verified.docx`
    - `microexpressions_article_verified.pdf`
    - `references.bib`
    - non-identifying figures
- `tools/`
  - Utility scripts used to prepare the two-test-tube condition mapping,
    rerun trial-level analysis from existing 100 fps frame features, and
    rebuild the article draft.
- `microexpressions_repo/microexp/`
  - Current pipeline source code.

## Current Analysis Status

- Representative videos processed: 15/15.
- Frame-level observations: 450,805.
- Automatic candidate micro-events: 3,614.
- Candidate sniff-like windows: 114.
- Two-test-tube condition mapping:
  - trial 1 = control / placebo
  - trial 2 = sweat
- Primary paired condition subset:
  - 7 complete paired videos
  - 14 primary trial rows
  - 132 paired sweat/control comparison rows

## Interpretation Boundaries

- The paired sweat/control results are preliminary and exploratory.
- Four-trial odor-sequence videos are kept separate and are not included in the
  main two-test-tube sweat/control comparison.
- Automatic AU proxies and candidate events are not FACS-certified labels and
  should be interpreted with manual review and quality-control notes.
