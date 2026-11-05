"""Check the statistics against published results, where the right answer is known.

If these fail, the numbers the toolkit reports can't be trusted, whatever else passes.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from doekit.analysis import (
    anova_table,
    coefficient_table,
    fit_metrics,
    fit_response,
    lack_of_fit,
    lenth_analysis,
)
from doekit.optimise import find_optimum
from doekit.runsheet import read_runsheet
from doekit.study import load_study, study_from_config

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA = ROOT / "archive/private"


def _coef(fit, term):
    table = coefficient_table(fit)
    return float(table.loc[table["term"] == term, "coef"].iloc[0])


@pytest.fixture
def chemical_yield():
    """Montgomery, Design and Analysis of Experiments: central composite, chemical yield."""
    rows = [
        (-1, -1, 76.5), (-1, 1, 77.0), (1, -1, 78.0), (1, 1, 79.5),
        (0, 0, 79.9), (0, 0, 80.3), (0, 0, 80.0), (0, 0, 79.7), (0, 0, 79.8),
        (1.414, 0, 78.4), (-1.414, 0, 75.6), (0, 1.414, 78.5), (0, -1.414, 77.0),
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=["x1", "x2", "y"])
    frame["time"] = 85 + 5 * frame["x1"]
    frame["temp"] = 175 + 5 * frame["x2"]
    study = study_from_config(
        {
            "name": "chemical_yield",
            "seed": 3,
            "factors": [{"name": "time", "low": 80, "high": 90}, {"name": "temp", "low": 170, "high": 180}],
            "responses": [{"name": "y", "goal": "maximise", "low": 75, "high": 81}],
            "model": "quadratic",
        }
    )
    return study, frame


def test_response_surface_coefficients_match_published(chemical_yield):
    study, frame = chemical_yield
    fit = fit_response(study, frame, study.responses[0])
    assert _coef(fit, "Intercept") == pytest.approx(79.94, abs=0.005)
    assert _coef(fit, "time") == pytest.approx(0.995, abs=0.001)
    assert _coef(fit, "temp") == pytest.approx(0.515, abs=0.001)
    assert _coef(fit, "time^2") == pytest.approx(-1.376, abs=0.001)
    assert _coef(fit, "temp^2") == pytest.approx(-1.001, abs=0.001)
    assert _coef(fit, "time:temp") == pytest.approx(0.25, abs=0.001)
    assert fit_metrics(fit)["r2"] == pytest.approx(0.9827, abs=0.0005)


def test_optimum_matches_published_stationary_point(chemical_yield):
    study, frame = chemical_yield
    fit = fit_response(study, frame, study.responses[0])
    best = find_optimum(study, [fit], seed=3).iloc[0]
    assert best["time"] == pytest.approx(86.95, abs=0.05)
    assert best["temp"] == pytest.approx(176.53, abs=0.05)
    assert best["y_predicted"] == pytest.approx(80.21, abs=0.01)


def test_lack_of_fit_uses_replicates_as_pure_error(chemical_yield):
    study, frame = chemical_yield
    fit = fit_response(study, frame, study.responses[0])
    result = lack_of_fit(fit)
    centre = np.array([79.9, 80.3, 80.0, 79.7, 79.8])
    assert result["ss_pure_error"] == pytest.approx(np.sum((centre - centre.mean()) ** 2))
    assert result["df_pure_error"] == 4
    assert result["df_lack_of_fit"] == 3
    assert result["ss_lack_of_fit"] + result["ss_pure_error"] == pytest.approx(fit.sse)


def test_predicted_r2_equals_real_leave_one_out(chemical_yield):
    study, frame = chemical_yield
    fit = fit_response(study, frame, study.responses[0])
    press = 0.0
    for row in range(len(frame)):
        train = frame.drop(index=row).reset_index(drop=True)
        model = fit_response(study, train, study.responses[0])
        press += (frame.loc[row, "y"] - model.predict_mean(frame.iloc[[row]])[0]) ** 2
    assert fit_metrics(fit)["press"] == pytest.approx(press)


def test_factorial_effects_and_lenth_match_published():
    """Montgomery: the 2^4 filtration rate experiment, with no degrees of freedom left."""
    y = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]
    grid = [(a, b, c, d) for d in (-1, 1) for c in (-1, 1) for b in (-1, 1) for a in (-1, 1)]
    frame = pd.DataFrame(grid, columns=list("ABCD")).astype(float)
    frame["y"] = y
    study = study_from_config(
        {"name": "filtration", "factors": [{"name": n, "low": -1, "high": 1} for n in "ABCD"], "responses": [{"name": "y"}]}
    )
    terms = [term for size in range(1, 5) for term in itertools.combinations("ABCD", size)]
    fit = fit_response(study, frame, study.responses[0], terms=terms)
    published = {
        "A": 21.625, "B": 3.125, "C": 9.875, "D": 14.625, "A:B": 0.125, "A:C": -18.125, "A:D": 16.625,
        "B:C": 2.375, "B:D": -0.375, "C:D": -1.125, "A:B:C": 1.875, "A:B:D": 4.125, "A:C:D": -1.625,
        "B:C:D": -2.625, "A:B:C:D": 1.375,
    }  # fmt: skip
    for term, effect in published.items():
        assert 2 * _coef(fit, term) == pytest.approx(effect), term
    lenth = lenth_analysis(coefficient_table(fit))
    assert set(lenth["active"]) == {"A", "C", "D", "A:C", "A:D"}
    assert lenth["pse"] == pytest.approx(2.625)


def test_mixture_coefficients_match_published():
    """Cornell, Experiments with Mixtures: yarn elongation on a {3,2} simplex lattice."""
    rows = [
        (1, 0, 0, 11.0), (1, 0, 0, 12.4), (0.5, 0.5, 0, 15.0), (0.5, 0.5, 0, 14.8), (0.5, 0.5, 0, 16.1),
        (0, 1, 0, 8.8), (0, 1, 0, 10.0), (0, 0.5, 0.5, 10.0), (0, 0.5, 0.5, 9.7), (0, 0.5, 0.5, 11.8),
        (0, 0, 1, 16.8), (0, 0, 1, 16.0), (0.5, 0, 0.5, 17.7), (0.5, 0, 0.5, 16.4), (0.5, 0, 0.5, 16.6),
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=["a", "b", "c", "y"])
    study = study_from_config(
        {
            "name": "yarn",
            "factors": [{"name": n, "type": "mixture"} for n in "abc"],
            "responses": [{"name": "y"}],
            "model": {"mixture": "quadratic"},
        }
    )
    fit = fit_response(study, frame, study.responses[0])
    published = {"a": 11.7, "b": 9.4, "c": 16.4, "a:b": 19.0, "a:c": 11.4, "b:c": -9.6}
    for term, value in published.items():
        assert _coef(fit, term) == pytest.approx(value), term
    # With no intercept column, an uncorrected R2 would be misleadingly close to 1.
    y = frame["y"].to_numpy()
    assert fit_metrics(fit)["r2"] == pytest.approx(1 - fit.sse / np.sum((y - y.mean()) ** 2))
    assert fit_metrics(fit)["r2"] == pytest.approx(0.951, abs=0.001)
    table = anova_table(fit).set_index("source")
    assert table.loc["Linear mixture", "df"] == 2
    assert table.loc["Total (corrected)", "sum_sq"] == pytest.approx(134.856, abs=0.001)


@pytest.mark.skipif(not PRIVATE_DATA.exists(), reason="the original data isn't in the public repository")
def test_new_model_reproduces_the_original_script_on_mew_data():
    """The fit behind archive/original_analysis/main_doe.py must be reproduced exactly."""
    study = load_study(PRIVATE_DATA / "configs/mew_fibre_diameter.yaml")
    frame = read_runsheet(study)
    fit = fit_response(study, frame, study.responses[0])

    # The original: a degree-2 polynomial in raw units, fitted by least squares.
    x = frame[study.factor_names].to_numpy()
    n = x.shape[1]
    columns = [np.ones(len(x))] + [x[:, i] for i in range(n)]
    columns += [x[:, i] * x[:, j] for i, j in itertools.combinations_with_replacement(range(n), 2)]
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, frame["dia"].to_numpy(), rcond=None)
    original = design @ beta

    assert fit.fitted == pytest.approx(original, abs=1e-8)
    assert fit_metrics(fit)["r2"] == pytest.approx(0.95899, abs=1e-4)
