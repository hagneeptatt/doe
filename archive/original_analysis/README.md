# Original analysis

The repo as it was before it became `doekit`. Nothing in this folder has been changed, only moved.

| Path | Contents |
|---|---|
| `main_doe.py` | The original single-file analysis: quadratic fit, surface plots, target search |
| `mew_doe_results.csv` | Melt electrowriting study: 5 factors, 46 runs, response `dia` |
| `bioprint_doe_*_results_*_valve.csv` | Jetting study: `press` and `vot` against `vol` (volume per drop), for HAMA and media, with small and large valves |
| `figures/` | The figures `main_doe.py` produced from the MEW data |

## Running the original script

It reads its CSV from, and writes its figures to, the current folder. It also needs `scikit-learn` and `seaborn`, which the `doe` environment doesn't include:

```bash
pip install scikit-learn seaborn
cd archive/original_analysis
python main_doe.py
```

## Re-analysing this data with doekit

`configs/examples/mew_fibre_diameter.yaml` and `configs/examples/jetting_hama_large_valve.yaml` read these CSVs directly, so they don't need to be copied anywhere. `tests/test_validation.py` checks that the new toolkit gives the same predictions as this script's model on the MEW data.
