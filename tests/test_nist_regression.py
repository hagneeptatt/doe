"""Check the regression engine against NIST's certified results.

NIST's Statistical Reference Datasets (StRD) for linear least squares come with results
certified to 15 significant digits, calculated with 500-digit arithmetic. They're the
standard way to test the accuracy of statistical software, and some are deliberately hard:
Longley's data are nearly collinear, and Filip's 10th-degree polynomial defeats many
packages. The data files in reference_data/nist_strd are copied unchanged from
https://www.itl.nist.gov/div898/strd/lls/lls.shtml

Accuracy is measured as NIST recommends, by the log relative error: the number of
significant digits that agree with the certified value.

The two datasets without an intercept (NoInt1 and NoInt2) aren't included, because doekit
always fits an intercept, except in mixture models.
"""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from doekit.analysis import fit_metrics, fit_response
from doekit.study import study_from_config

FOLDER = Path(__file__).resolve().parent / "reference_data" / "nist_strd"

# Name, then the model: a polynomial degree in x, or "linear" for several predictors.
DATASETS = [
    ("Norris", 1),  # lower difficulty
    ("Pontius", 2),  # lower difficulty
    ("Longley", "linear"),  # higher difficulty: nearly collinear predictors
    ("Wampler1", 5),  # higher difficulty: an exact fit
    ("Wampler2", 5),
    ("Wampler3", 5),
    ("Wampler4", 5),
    ("Wampler5", 5),
    ("Filip", 10),  # higher difficulty: many packages fail this one
]
# NIST's results have 15 digits. When this was written, doekit matched at least 7.8 digits on
# every fitted value and 8.9 on every summary statistic, so 7 leaves a little room for
# differences between computers.
MINIMUM_DIGITS = 7


def agreeing_digits(value: float, certified: float) -> float:
    """NIST's log relative error, using the absolute error where the certified value is 0."""
    if value == certified:
        return 15.0
    error = abs(value - certified) / (abs(certified) if certified != 0 else 1.0)
    return float(min(15.0, -np.log10(error)))


def read_dataset(name: str):
    text = (FOLDER / f"{name}.dat").read_text()
    lines = text.splitlines()
    start, end = (int(v) for v in re.search(r"Data\s+\(lines (\d+) to (\d+)\)", text).groups())
    columns = lines[start - 2].split()[1:]  # the line before the data names the columns
    raw = [line.split() for line in lines[start - 1 : end]]
    frame = pd.DataFrame([[float(v) for v in row] for row in raw], columns=columns)
    exact = [[Fraction(v) for v in row] for row in raw]  # the data exactly as written
    certified = {
        "beta": [Fraction(v) for v in re.findall(r"^\s+B\d+\s+(\S+)\s+\S+\s*$", text, flags=re.M)],
        "sd": float(re.search(r"Residual\s+Standard Deviation\s+(\S+)", text).group(1)),
        "r2": float(re.search(r"R-Squared\s+(\S+)", text).group(1)),
        "ss_regression": float(re.search(r"Regression\s+\d+\s+(\S+)", text).group(1)),
        "ss_residual": float(re.search(r"Residual\s+\d+\s+(\S+)", text).group(1)),
    }
    return frame, exact, columns, certified


def fit_dataset(name: str, model):
    frame, exact, columns, certified = read_dataset(name)
    predictors = [column for column in columns if column != "y"]
    factors = [{"name": x, "low": float(frame[x].min()), "high": float(frame[x].max())} for x in predictors]
    study = study_from_config({"name": name.lower(), "factors": factors, "responses": [{"name": "y"}]})
    if model == "linear":
        terms = [(x,) for x in predictors]
    else:
        terms = [("x",) * power for power in range(1, model + 1)]
    fit = fit_response(study, frame, study.responses[0], terms=terms)

    # The certified fitted values, worked out exactly from the certified coefficients.
    position = {column: i for i, column in enumerate(columns)}
    fitted = []
    for row in exact:
        if model == "linear":
            values = [Fraction(1)] + [row[position[x]] for x in predictors]
        else:
            values = [row[position["x"]] ** power for power in range(model + 1)]
        fitted.append(float(sum(b * v for b, v in zip(certified["beta"], values))))
    return fit, certified, np.array(fitted)


@pytest.mark.parametrize("name, model", DATASETS)
def test_regression_matches_nist_certified_values(name, model):
    fit, certified, certified_fitted = fit_dataset(name, model)
    metrics = fit_metrics(fit)
    ss_regression = float(np.sum((fit.fitted - fit.y.mean()) ** 2))
    found = {
        "residual standard deviation": agreeing_digits(metrics["rmse"], certified["sd"]),
        "R-squared": agreeing_digits(metrics["r2"], certified["r2"]),
        "regression sum of squares": agreeing_digits(ss_regression, certified["ss_regression"]),
        "residual sum of squares": agreeing_digits(fit.sse, certified["ss_residual"]),
        # Fitted values carry the certified coefficients, whatever coding doekit uses inside.
        "fitted values (worst)": min(agreeing_digits(a, b) for a, b in zip(fit.fitted, certified_fitted)),
    }
    poor = {what: round(value, 1) for what, value in found.items() if value < MINIMUM_DIGITS}
    assert not poor, f"{name}: fewer than {MINIMUM_DIGITS} digits agree for {poor}"
