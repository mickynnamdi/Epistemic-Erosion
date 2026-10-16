# results_summary/

Output of `scripts/make_tables.py` (and, for Table A16, `scripts/analysis/detector_table.py`) on the paper's own generations, with every sample rebuilt by the
scripts in this repository. Stages that need a 70B model or a paid API were run on the stored model
and judge outputs of the paper (`--stage analyze`).

| File | Contents |
|---|---|
| `PAPER_CHECK.md` | every value printed in the paper that the package recomputes, next to the reproduced value: 774 match, 9 approximate (token-probe numerical precision) |
| `table1_models.csv` | Table 1 |
| `tableA18_distortion_types.csv` | Table A18 |
| `tablesA5_A6_A7_mitigation.csv` | Tables A5, A6, A7 (EFS, SD, delta, Cohen's d, Mann-Whitney p) |
| `table2_hosted.csv` | Table 2 |
| `tableA4_length_quintiles.csv` | Table A4 |
| `tableA16_detector.csv` | Table A16 (from the paper's 12 detector runs) |

Aggregate numbers only: no text, no identifiers.
