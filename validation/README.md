# Square-Domain Paper Validation

These results reproduce Section 4.3 / Table 3 of Liu et al., JCP 467 (2022) 111436.

| Case | KnR | KnN | Ntheta | Nphi | GSIS steps | CIS steps | Paper GSIS | Paper CIS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| knr1e-3_knn1e5 | 0.001 | 100000 | 20 | 40 | 43 | timeout | 43 | >1e6 |
| knr1e-2_knn1e5 | 0.01 | 100000 | 40 | 80 | 24 | 13234 | 24 | 13234 |
| knr1e-1_knn1e5 | 0.1 | 100000 | 40 | 80 | 26 | 269 | 26 | 269 |
| knr1_knn1 | 1 | 1 | 80 | 160 | 28 | 29 | 28 | 29 |
| knr10_knn1e-2 | 10 | 0.01 | 40 | 80 | 47 | 1883 | 47 | 1883 |

Generated files:

- `summary.csv`: per-run iteration, residual, timing, and Fourier-limit metrics.
- `scheme_comparison.csv`: GSIS/CIS pairwise temperature-field comparisons.
- `figures/*_temperature_contours.png`: side-by-side temperature contours.
- `results/<case>/<scheme>/`: residual CSV, Tecplot field, ParaView collection, Fourier reference, stdout, and run metadata.
