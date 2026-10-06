# PHASE manuscript package (revision)

Manuscript: "Physics-informed e-processes for anytime-valid damage detection in
continuously monitored structures", submitted to Artificial Intelligence for a
Sustainable Built Environment (special issue: Physics-Informed AI for
Infrastructure Resilience). This folder holds the revised version.

## Layout

```
main.tex, refs.bib          LaTeX source of the revised manuscript (changes shown in blue)
figures/                    nine figures (PNG)
code/                       simulation, monitors, experiments, Z24 pipeline
revision/                   response to reviewers, highlighted and clean PDFs
manuscript_anonymised.pdf   double-blind file as first submitted (version 1)
```

## Rebuilding the PDF

Figures are found through `\graphicspath{{figures/}}`, so compile from this
folder. Revised text appears in blue; the clean version has no colour:

    pdflatex main && bibtex main && pdflatex main && pdflatex main
    pdflatex -jobname=main_clean "\def\CLEANBUILD{1}\input{main}"     # clean (run the three-pass sequence)

## Simulation study (code/)

| File | Role |
|---|---|
| model.py | shear-frame digital twin, environmental driver, damage profiles |
| monitors.py | PHASE, variants (-ND, -C, -R, -E, -BB, -Omega, -S, -NA), comparators |
| twin.py | digital-twin error experiment: monitor built from a perturbed structural model |
| stage.py | Monte Carlo driver, one stage per call; writes part_*.json |
| aggregate.py | reads part_*.json, writes summary.json (counts, exact intervals) and the simulation figures |
| identifiability.py | rank structure, confounded subspace, coherence, nuisance-model sensitivity; writes identifiability.json |
| make_tables.py | generates the LaTeX table bodies from the json files |
| schematics.py | draws the pipeline, frame and bridge schematics |
| summary.json | every number reported for the simulation study |

Stages, with fixed seeds: `geometry`, `calib`, `healthy`, `scen0`..`scen6`,
`long`, `long1`, `long2`, `burn120`, `burn240`, `burn480`, `burn720`,
`burn960`, `burn1200`, `burn1460`, `late` (late damage onset), `twin` (digital-twin error), `rate3s`
(trigger rate of the 3-sigma chart), `paths`, for example `python3 stage.py calib`.
Run `geometry` and `calib` first. The full study takes roughly an hour on four
cores. The intermediate part_*.json files are not shipped (summary.json holds
the aggregated results); `stage.py` regenerates them and `aggregate.py` then
rebuilds summary.json and the figures. The stages reproduce the numbers of
version 1 exactly for every monitor that existed there.

Record accounting: 150 calibration + 300 healthy two-year + 300 healthy
four-year + 7 x 150 damage scenarios + 7 x 200 commissioning sweep + 4 x 100
late-onset + 4 x 100 twin-error = 4,000 records (3,300 two-year, 700 four-year), plus three records
drawn for the illustrative figure.

## Z24 bridge study (code/)

| File | Role |
|---|---|
| z24_modal.py | frequency domain decomposition and band-limited modal tracking |
| z24_experiment.py | monitors and experiments R1a-R5 |
| z24_run_all.py | reproduces the full Z24 table, both modalities, ten monitors |
| z24_sensitivity.py | chronological, reversed, circular-shift, set-up-shuffle orderings; exchangeability diagnostics; writes z24_sensitivity.json (set `ORDERS=poolblock OUT=z24_pool.json` for the set-up shuffle across commissioning days) |
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

## KW51 study (code/)

| File | Role |
|---|---|
| kw51_experiment.py | PHASE (reduced: regression surrogate, orthant cone) and comparators on the KW51 tracked-mode data; `--download` fetches trackedmodes.zip from Zenodo record 3745914 (CC BY-NC-SA 4.0; not redistributed here) |
| kw51_figure.py | draws fig_kw51.png |
| kw51_results.json | all 120 commissioning configurations (2 epoch lengths x 4 surrogates x 15 windows) and the evidence paths of the plotted configurations |

    cd code && python3 kw51_experiment.py --download && python3 kw51_experiment.py   # about 3 minutes

### Interpretation caveats (also stated in the paper, Section 5)

- The released files carry no environmental covariate, so the surrogate reduces
  to a running intercept and the identifiability analysis cannot be exercised.
- Each structural state was measured on a different day; the two undamaged
  references differ by 0.9-1.4% in tracked frequencies, comparable to the damage
  signatures. Experiment R5 shows a nominally undamaged new day also alarms
  when commissioning epochs from the two reference days are mixed. R2 and R4
  therefore show a response to real change that depends on how commissioning
  data are ordered (z24_sensitivity.py), not calibrated separation of damage
  from environment. The recovery experiment R3 is insensitive to ordering.
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
