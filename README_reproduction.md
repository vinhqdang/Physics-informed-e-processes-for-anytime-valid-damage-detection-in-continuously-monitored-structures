# PHASE submission package

Manuscript: "Physics-informed e-processes for anytime-valid damage detection in
continuously monitored structures", for Artificial Intelligence for a
Sustainable Built Environment (special issue: Physics-Informed AI for
Infrastructure Resilience).

## Layout

```
manuscript_anonymised.pdf   double-blind submission file (44 pp, 9 figures, 6 tables)
title_page.pdf / .tex       separate title page: author details and declarations
main.tex, refs.bib          LaTeX source of the manuscript
figures/                    nine figures (PNG)
code/                       simulation, monitors, experiments, Z24 pipeline
submission_text/            cover letter and ScholarOne form fields (plain text)
```

## Rebuilding the PDF

Figures are found through `\graphicspath{{figures/}}`, so compile from this
folder:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Verified from a clean copy of this package: 44 pages, no undefined references.

## Simulation study (code/)

| File | Role |
|---|---|
| model.py | shear-frame digital twin, environmental driver, damage profiles |
| monitors.py | PHASE, ablations, and baselines |
| stage.py | Monte Carlo driver, one stage per call; writes part_*.json |
| aggregate.py | reads part_*.json, writes summary.json and the simulation figures |
| schematics.py | draws the pipeline, frame and bridge schematics |
| summary.json | every number reported for the simulation study |

Stages, with fixed seeds: `geometry`, `calib`, `healthy`, `scen0`..`scen6`,
`long`, `long1`, `long2`, `burn120`, `burn240`, `burn480`, `burn1460`, `paths`,
for example `python3 stage.py calib`. Run `calib` first. The full study takes
roughly an hour single-threaded. The intermediate part_*.json files are not
shipped (summary.json holds the aggregated results); `stage.py` regenerates
them and `aggregate.py` then rebuilds summary.json and the figures.

## Z24 bridge study (code/)

| File | Role |
|---|---|
| z24_modal.py | frequency domain decomposition and band-limited modal tracking |
| z24_experiment.py | monitors and experiments R1a-R5 |
| z24_run_all.py | reproduces the full Z24 table, both modalities, ten monitors |
| z24_figure.py | draws fig_z24.png |
| z24_modal.npz | extracted modal features, ambient vibration (690 epochs) |
| z24_modal_fvt.npz | extracted modal features, forced vibration (711 epochs) |
| z24_full.json | results of z24_run_all.py (200 orderings per experiment) |

Everything in the Z24 section reproduces from the two .npz files:

    cd code && python3 z24_run_all.py        # about 4 minutes

Set `REPS=20` for a quick check.

### Data provenance

The raw Z24 progressive damage test acceleration records are NOT included. They
are KU Leuven research data and must be obtained from KU Leuven. The analysis
used the ambient (avt) and forced (fvt) records for test states 01-10, in
Parquet form. To rebuild the features from raw records, set `ROOT` in
z24_modal.py, delete the .npz files, and run `python3 z24_modal.py`.

### Interpretation caveats (also stated in the paper, Section 5)

- The released files carry no environmental covariate, so the surrogate reduces
  to a running intercept and the identifiability analysis cannot be exercised.
- Each structural state was measured on a different day; the two undamaged
  references differ by 0.9-1.4% in tracked frequencies, comparable to the damage
  signatures. Experiment R5 shows a nominally undamaged new day also alarms.
  R2 and R4 therefore show earlier response to real change, not calibrated
  separation of damage from environment.
- Only three modes were tracked consistently across all ten states.

## Figures

All nine figures are the authors' own work (three schematics and six plots from
the experiments); no third-party imagery is used. The figure environment for the
bridge in main.tex contains a commented-out two-panel block for photographs, if
rights are later cleared (e.g. Reynders and De Roeck, 2009, DOI
10.1002/9780470061626.shm165). Emerald requires non-exclusive, worldwide, print
and electronic rights for the life of the work.

## Environment

Python 3, numpy, scipy, pandas, matplotlib, pyarrow (only to read raw Z24
Parquet). No GPU or network needed once the .npz files are present.
