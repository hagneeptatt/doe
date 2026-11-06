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
from doekit.designs import create_design
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


# ---------------------------------------------------------------------- categorical factors
def battery_data() -> pd.DataFrame:
    """Montgomery, Design and Analysis of Experiments, Example 5.1: battery life.

    Three plate materials and three temperatures (15, 70, 125 F), with four batteries each.
    The published effects, ANOVA and regression are reproduced in Purdue's STAT 514 notes,
    https://www.stat.purdue.edu/~chakrabp/STAT514/Notes/factorial.pdf
    """
    data = {
        ("1", 15): [130, 155, 74, 180], ("1", 70): [34, 40, 80, 75], ("1", 125): [20, 70, 82, 58],
        ("2", 15): [150, 188, 159, 126], ("2", 70): [136, 122, 106, 115], ("2", 125): [25, 70, 58, 45],
        ("3", 15): [138, 110, 168, 160], ("3", 70): [174, 120, 150, 139], ("3", 125): [96, 104, 82, 60],
    }  # fmt: skip
    rows = [(material, temp, life) for (material, temp), lives in data.items() for life in lives]
    frame = pd.DataFrame(rows, columns=["material", "temp", "life"])
    frame["temperature"] = frame["temp"].map({15: "F15", 70: "F70", 125: "F125"})
    return frame


def test_two_categorical_factors_match_published_effects_and_anova():
    """Effect coding gives the textbook effects: each level's mean minus the grand mean."""
    study = study_from_config(
        {
            "name": "battery",
            "factors": [{"name": "material", "levels": ["1", "2", "3"]}, {"name": "temperature", "levels": ["F15", "F70", "F125"]}],
            "responses": [{"name": "life"}],
            "model": "interaction",
        }
    )
    fit = fit_response(study, battery_data(), study.responses[0])
    published = {
        "Intercept": 105.5278, "material[1]": -22.3612, "material[2]": 2.8055,
        "temperature[F15]": 39.3055, "temperature[F70]": 2.0555,
        "material[1]:temperature[F15]": 12.2779, "material[1]:temperature[F70]": -27.9721,
        "material[2]:temperature[F15]": 8.1112, "material[2]:temperature[F70]": 9.3612,
    }  # fmt: skip
    for term, value in published.items():
        assert _coef(fit, term) == pytest.approx(value, abs=6e-4), term

    table = anova_table(fit).set_index("source")
    for source, ss, df in [("material", 10683.72222, 2), ("temperature", 39118.72222, 2), ("material:temperature", 9613.77778, 4), ("Residual", 18230.75, 27)]:
        assert table.loc[source, "sum_sq"] == pytest.approx(ss, abs=1e-4), source
        assert table.loc[source, "df"] == df
    assert table.loc["material", "F"] == pytest.approx(7.91, abs=0.005)
    assert table.loc["temperature", "F"] == pytest.approx(28.97, abs=0.005)
    assert table.loc["material:temperature", "p"] == pytest.approx(0.0186, abs=5e-5)
    assert fit_metrics(fit)["r2"] == pytest.approx(0.765210, abs=5e-7)
    assert fit_metrics(fit)["rmse"] == pytest.approx(25.98486, abs=5e-6)


def test_categorical_factor_with_a_curved_numeric_factor_matches_published_regression():
    """The same data, with temperature as a number: SAS's coefficients and standard errors.

    The notes code material as 1, 0, -1 and 0, 1, -1, and temperature as (temp - 70) / 55,
    which is exactly doekit's coding.
    """
    study = study_from_config(
        {
            "name": "battery",
            "factors": [{"name": "material", "levels": ["1", "2", "3"]}, {"name": "temp", "low": 15, "high": 125}],
            "responses": [{"name": "life"}],
            "model": {"terms": ["material", "temp", "material*temp", "temp^2", "material*temp^2"]},
        }
    )
    fit = fit_response(study, battery_data(), study.responses[0])
    table = coefficient_table(fit).set_index("term")
    published = {
        "Intercept": (107.58333, 7.50118), "material[1]": (-50.33333, 10.60827), "material[2]": (12.16667, 10.60827),
        "temp": (-40.33333, 5.30414), "material[1]:temp": (1.70833, 7.50118), "material[2]:temp": (-12.79167, 7.50118),
        "temp^2": (-3.08333, 9.18704), "material[1]:temp^2": (41.95833, 12.99243), "material[2]:temp^2": (-14.04167, 12.99243),
    }  # fmt: skip
    for term, (coef, std_err) in published.items():
        assert table.loc[term, "coef"] == pytest.approx(coef, abs=6e-6), term
        assert table.loc[term, "std_err"] == pytest.approx(std_err, abs=6e-6), term


# ---------------------------------------------------------------------- a two-stage experiment
# Myers, Montgomery & Anderson-Cook, Response Surface Methodology, Table 7.6. A factorial
# with centre points was run first (block 1). Its first-order model showed lack of fit, so
# axial points and more centre points were added later (block 2). The published analysis
# is in Lenth (2009), Response-surface methods in R, using rsm, Journal of Statistical
# Software 32(7), and its output is printed in the rsm package's vignette.
CHEMICAL_REACTION = [
    (80, 170, 1, 80.5), (80, 180, 1, 81.5), (90, 170, 1, 82.0), (90, 180, 1, 83.5),
    (85, 175, 1, 83.9), (85, 175, 1, 84.3), (85, 175, 1, 84.0),
    (85, 175, 2, 79.7), (85, 175, 2, 79.8), (85, 175, 2, 79.5),
    (92.07, 175, 2, 78.4), (77.93, 175, 2, 75.6), (85, 182.07, 2, 78.5), (85, 167.93, 2, 77.0),
]  # fmt: skip


def reaction_study(model: str):
    return study_from_config(
        {
            "name": "reaction",
            "seed": 1,
            "factors": [{"name": "Time", "low": 80, "high": 90}, {"name": "Temp", "low": 170, "high": 180}],
            "responses": [{"name": "Yield", "goal": "maximise", "low": 75, "high": 85}],
            "model": model,
        }
    )


def reaction_data() -> pd.DataFrame:
    return pd.DataFrame(CHEMICAL_REACTION, columns=["Time", "Temp", "block", "Yield"])


def test_first_block_shows_the_published_lack_of_fit():
    study = reaction_study("linear")
    first = reaction_data().query("block == 1").drop(columns="block")
    fit = fit_response(study, first, study.responses[0])
    table = coefficient_table(fit).set_index("term")
    for term, (coef, std_err) in {"Intercept": (82.81429, 0.54719), "Time": (0.875, 0.72386), "Temp": (0.625, 0.72386)}.items():
        assert table.loc[term, "coef"] == pytest.approx(coef, abs=6e-6), term
        assert table.loc[term, "std_err"] == pytest.approx(std_err, abs=6e-6), term
    assert fit_metrics(fit)["r2"] == pytest.approx(0.3555, abs=5e-5)
    result = lack_of_fit(fit)
    assert (result["df_lack_of_fit"], result["df_pure_error"]) == (2, 2)
    assert result["ss_pure_error"] == pytest.approx(0.0867, abs=5e-5)
    assert result["f"] == pytest.approx(95.7335, abs=5e-5)
    assert result["p"] == pytest.approx(0.01034, abs=5e-6)


def test_both_blocks_match_the_published_second_order_analysis():
    study = reaction_study("quadratic")
    fit = fit_response(study, reaction_data(), study.responses[0])
    table = coefficient_table(fit).set_index("term")
    published = {
        "Time": (0.932541, 0.057699), "Temp": (0.577712, 0.057699), "Time:Temp": (0.125, 0.081592),
        "Time^2": (-1.308555, 0.060064), "Temp^2": (-0.933442, 0.060064),
    }  # fmt: skip
    for term, (coef, std_err) in published.items():
        assert table.loc[term, "coef"] == pytest.approx(coef, abs=6e-7), term
        assert table.loc[term, "std_err"] == pytest.approx(std_err, abs=6e-7), term

    # rsm reports block 2 minus block 1 (-4.457530, SE 0.087226). doekit's effect coding
    # reports each block's difference from the average, which is half of that.
    assert table.loc["block[1]", "coef"] == pytest.approx(4.457530 / 2, abs=6e-7)
    assert table.loc["block[1]", "std_err"] == pytest.approx(0.087226 / 2, abs=6e-7)
    assert table.loc["Intercept", "coef"] == pytest.approx(84.095427 - 4.457530 / 2, abs=6e-7)

    metrics = fit_metrics(fit)
    assert metrics["r2"] == pytest.approx(0.9981, abs=5e-5)
    assert metrics["adj_r2"] == pytest.approx(0.9964, abs=5e-5)
    result = lack_of_fit(fit)
    assert (result["df_lack_of_fit"], result["df_pure_error"]) == (3, 4)
    assert result["f"] == pytest.approx(0.5307, abs=5e-5)
    assert result["p"] == pytest.approx(0.6851, abs=5e-5)
    anova = anova_table(fit).set_index("source")
    assert anova.loc["Residual", "sum_sq"] == pytest.approx(0.186, abs=5e-4)
    # The model's sum of squares after allowing for blocks: 9.626 + 0.063 + 17.791 in rsm.
    # rsm's block row adds blocks first (Type I), while doekit adjusts every term for all
    # the others (Type III), so that row is expected to differ slightly and isn't compared.
    assert anova.loc["Model", "sum_sq"] == pytest.approx(9.626 + 0.063 + 17.791, abs=2e-3)


def test_best_settings_match_the_published_stationary_point():
    study = reaction_study("quadratic")
    fit = fit_response(study, reaction_data(), study.responses[0])
    best = find_optimum(study, [fit], seed=1).iloc[0]
    assert best["Time"] == pytest.approx(86.86148, abs=1e-3)
    assert best["Temp"] == pytest.approx(176.67190, abs=1e-3)


def test_adding_runs_after_the_first_block_gives_the_textbook_design():
    """Faced with the first block alone, adding 4 runs for a curved model in a new session
    should give the axial points of a face-centred central composite design. Brute force
    over every possible choice confirms that these are the only best runs."""
    from doekit.augment import augment_design

    first = reaction_data().query("block == 1").drop(columns="block")
    for criterion in ("D", "I"):
        result = augment_design(reaction_study("quadratic"), first, 4, criterion=criterion)
        coded = {(round((t - 85) / 5, 6), round((p - 175) / 5, 6)) for t, p in result.new_runs[["Time", "Temp"]].to_numpy()}
        assert coded == {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)}, criterion
        assert (result.new_runs["block"] == 2).all()


# ---------------------------------------------------------------------- several responses at once
def test_desirability_optimum_matches_published():
    """Myers & Montgomery's reaction with two responses: maximise conversion (80 to 97) while
    keeping thermal activity near 57.5 (55 to 60), within the cube of the axial points.

    The fitted equations and the optimum come from the vignette of the R package
    desirability (Kuhn), which follows Derringer & Suich (1980):
    https://cran.r-project.org/web/packages/desirability/vignettes/desirability.pdf
    The equations are recreated exactly from noise-free runs on a central composite design.
    """

    def conversion(t, temp, cat):
        return (81.09 + 1.0284 * t + 4.043 * temp + 6.2037 * cat - 1.8366 * t**2 + 2.9382 * temp**2 - 5.1915 * cat**2
                + 2.2150 * t * temp + 11.375 * t * cat - 3.875 * temp * cat)  # fmt: skip

    def activity(t, temp, cat):
        return (59.85 + 3.583 * t + 0.2546 * temp + 2.2298 * cat + 0.83479 * t**2 + 0.07484 * temp**2 + 0.05716 * cat**2
                - 0.3875 * t * temp - 0.375 * t * cat + 0.3125 * temp * cat)  # fmt: skip

    axial = 1.682
    names = ["time", "temperature", "catalyst"]
    points = list(itertools.product([-1.0, 1.0], repeat=3))
    points += [tuple(sign * axial if j == i else 0.0 for j in range(3)) for i in range(3) for sign in (-1, 1)]
    points += [(0.0, 0.0, 0.0)] * 6
    frame = pd.DataFrame(points, columns=names)
    frame["conversion"] = conversion(*(frame[n] for n in names))
    frame["activity"] = activity(*(frame[n] for n in names))
    study = study_from_config(
        {
            "name": "reaction",
            "seed": 1,
            "factors": [{"name": n, "low": -axial, "high": axial} for n in names],
            "responses": [
                {"name": "conversion", "goal": "maximise", "low": 80, "high": 97},
                {"name": "activity", "goal": "target", "low": 55, "high": 60, "target": 57.5},
            ],
            "model": "quadratic",
        }
    )
    fits = [fit_response(study, frame, response) for response in study.responses]
    best = find_optimum(study, fits, seed=1).iloc[0]
    assert best["desirability"] == pytest.approx(0.9425094, abs=1e-5)
    assert best["time"] == pytest.approx(-0.5117132, abs=2e-3)
    assert best["temperature"] == pytest.approx(1.682, abs=1e-6)
    assert best["catalyst"] == pytest.approx(-0.5863863, abs=2e-3)
    assert best["conversion_predicted"] == pytest.approx(95.1, abs=0.05)
    assert best["activity_predicted"] == pytest.approx(57.5, abs=0.05)


# ---------------------------------------------------------------------- design tables
def test_half_fraction_matches_published_aliases_and_effects():
    """Montgomery, Example 8.1: the filtration experiment as a 2^(4-1) with I = ABCD."""
    factors = [{"name": n, "low": -1, "high": 1} for n in "ABCD"]
    design = create_design(study_from_config({"name": "f", "factors": factors, "design": {"type": "fractional_factorial", "resolution": 4}}))
    assert len(design.frame) == 8
    assert (design.frame[list("ABCD")].prod(axis=1) == 1).all()  # I = ABCD
    assert "A:B = C:D; A:C = B:D; A:D = B:C" in " ".join(design.notes)

    runs = {(-1, -1, -1, -1): 45, (1, -1, -1, 1): 100, (-1, 1, -1, 1): 45, (1, 1, -1, -1): 65,
            (-1, -1, 1, 1): 75, (1, -1, 1, -1): 60, (-1, 1, 1, -1): 80, (1, 1, 1, 1): 96}  # fmt: skip
    frame = pd.DataFrame([(*point, y) for point, y in runs.items()], columns=[*"ABCD", "y"]).astype(float)
    study = study_from_config({"name": "f", "factors": factors, "responses": [{"name": "y"}]})
    terms = [("A",), ("B",), ("C",), ("D",), ("A", "B"), ("A", "C"), ("A", "D")]
    fit = fit_response(study, frame, study.responses[0], terms=terms)
    published = {"A": 19.0, "B": 1.5, "C": 14.0, "D": 16.5, "A:B": -1.0, "A:C": -18.5, "A:D": 19.0}
    for term, effect in published.items():
        assert 2 * _coef(fit, term) == pytest.approx(effect), term


def test_box_behnken_matches_the_published_table():
    """Box & Behnken (1960): for 3 factors, the 12 midpoints of the cube's edges, plus centre points."""
    factors = [{"name": n, "low": -1, "high": 1} for n in "ABC"]
    design = create_design(study_from_config({"name": "b", "factors": factors, "design": {"type": "box_behnken"}}))
    points = {tuple(int(round(v)) for v in row) for row in design.frame[list("ABC")].to_numpy()}
    published = {point for point in itertools.product([-1, 0, 1], repeat=3) if sorted(map(abs, point)) == [0, 1, 1]}
    assert points - {(0, 0, 0)} == published


@pytest.mark.parametrize("count", [2, 3, 4])
def test_rotatable_central_composite_uses_the_published_axial_distance(count):
    """Box & Hunter (1957): a rotatable design puts its axial points at (2^k)^(1/4)."""
    factors = [{"name": f"x{i}", "low": -1, "high": 1} for i in range(count)]
    settings = {"type": "central_composite", "variant": "circumscribed", "alpha": "rotatable"}
    design = create_design(study_from_config({"name": "c", "factors": factors, "design": settings}))
    values = design.frame[[f"x{i}" for i in range(count)]]
    assert values.abs().max().max() == pytest.approx((2**count) ** 0.25, abs=1e-3)
    face = create_design(study_from_config({"name": "c", "factors": factors, "design": {"type": "central_composite"}}))
    assert face.frame[[f"x{i}" for i in range(count)]].abs().max().max() == pytest.approx(1.0)
