# EDA summary — 2026-09-03 14:10

Rules and thresholds live in config.py (decisions, not constants).

| rule | columns retired |
|---|---|
| constant: n_unique == 1 | 116 |
| missing: NaN fraction > 0.5 | 28 |
| near-zero variance: dom_frac > 0.99 or cv < 0.01 or n_unique < 5; deviants not enriched (rescue needs n_deviant >= 5 and enrichment >= 2.0) | 19 |
| correlated: |r| > 0.95; dropped the more-missing member of the pair | 170 |

Active sensors after retirement: **257** (legacy flat-file run: 257 — must match)

Supervised-rule note: the NZV enrichment rescue uses the target. Computed on the full frame until config.CUTOFF is set; then on the CV pool only, applied globally.

Data artifacts written by this run: none. Registry flips applied; this folder is documentation-grade evidence.
