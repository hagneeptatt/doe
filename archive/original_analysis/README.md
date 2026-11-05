# Original analysis

`main_doe.py` is the original single-file analysis that doekit grew out of: a quadratic fit, surface plots and a target search. It has been moved here, but not changed.

The original experimental data isn't in the public repository, so this script won't run from a fresh copy of the project. It needs a CSV called `mew_doe_results.csv` in the folder you run it from, with a label in the first column, the factors next, and the response last.

It also needs `scikit-learn` and `seaborn`, which the `doe` environment doesn't include.

For new work, use doekit instead. See the [README](../../README.md).
