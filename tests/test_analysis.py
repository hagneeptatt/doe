"""Check model building, reduction, interpretation, prediction and optimisation."""

import numpy as np
import pandas as pd
import pytest

from doekit.analysis import (
    _removable_terms,
    anova_table,
    curvature_test,
    fit_all,
    fit_metrics,
    fit_response,
    interpret_fit,
    run_diagnostics,
    suggest_transform,
)
from doekit.expressions import evaluate, names_in
from doekit.model import find_aliases, model_matrix, model_terms, parse_term, term_label
from doekit.optimise import Limits, desirability, find_optimum, overall_desirability
from doekit.study import Response, StudyError, study_from_config


def linear_study(**extra):
    return study_from_config(
        {
            "name": "t",
            "factors": [{"name": "a", "low": 0, "high": 10}, {"name": "b", "low": 100, "high": 200}],
            "responses": [{"name": "y", "goal": "maximise", "low": 0, "high": 100}],
            **extra,
        }
    )


def grid_frame(function, seed=0, noise=0.0, levels=5):
    values = np.linspace(0, 1, levels)
    a, b = np.meshgrid(values * 10, 100 + values * 100)
    frame = pd.DataFrame({"a": a.ravel(), "b": b.ravel()})
    frame["y"] = function(frame) + np.random.default_rng(seed).normal(0, noise, len(frame))
    return frame


# ------------------------------------------------------------------ expressions
def test_expressions_do_arithmetic():
    assert evaluate("2 * a + b ** 2", {"a": 3, "b": 2}) == 10
    assert evaluate("exp(0) + log(1) + sqrt(4)", {}) == 3
    assert names_in("gelma + 2 * exp(alginate)") == {"gelma", "alginate"}


def test_expressions_refuse_code():
    for attempt in ["__import__('os').system('ls')", "a.__class__", "open('x')", "[a for a in b]", "lambda: 1"]:
        with pytest.raises(ValueError):
            evaluate(attempt, {"a": 1, "b": 2})


# ------------------------------------------------------------------ coding and model terms
def test_process_factors_are_coded_from_minus_one_to_plus_one():
    study = linear_study()
    coded = study.coded(pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [100.0, 150.0, 200.0]}))
    assert coded["a"] == pytest.approx([-1, 0, 1])
    assert coded["b"] == pytest.approx([-1, 0, 1])


def test_mixture_components_become_pseudo_components():
    study = study_from_config(
        {
            "name": "t",
            "factors": [
                {"name": "a", "type": "mixture", "low": 0.2},
                {"name": "b", "type": "mixture", "low": 0.1},
                {"name": "c", "type": "mixture", "low": 0.1},
            ],
        }
    )
    assert study.pseudo_range == pytest.approx(0.6)
    coded = study.coded(pd.DataFrame({"a": [0.8], "b": [0.1], "c": [0.1]}))
    assert coded["a"] == pytest.approx([1.0])
    assert coded["b"] == pytest.approx([0.0])


def test_unreachable_mixture_limits_are_tightened():
    study = study_from_config(
        {
            "name": "t",
            "factors": [
                {"name": "a", "type": "mixture", "low": 0.5, "high": 1.0},
                {"name": "b", "type": "mixture", "low": 0.1, "high": 0.3},
                {"name": "c", "type": "mixture", "low": 0.1, "high": 0.3},
            ],
        }
    )
    assert study.factor("a").high == pytest.approx(0.8)  # b and c need at least 0.2 between them


def test_impossible_mixture_limits_are_refused():
    with pytest.raises(StudyError, match="no room"):
        study_from_config(
            {"name": "t", "factors": [{"name": "a", "type": "mixture", "low": 0.6}, {"name": "b", "type": "mixture", "low": 0.5}]}
        )


def test_categorical_effect_coding_sums_to_zero():
    study = study_from_config({"name": "t", "factors": [{"name": "c", "levels": ["x", "y", "z"]}], "model": "linear"})
    frame = pd.DataFrame({"c": ["x", "y", "z"]})
    matrix = model_matrix(study, frame, model_terms(study))
    assert matrix.names == ["Intercept", "c[x]", "c[y]"]
    assert matrix.values[:, 1] == pytest.approx([1, 0, -1])
    assert matrix.values[:, 2] == pytest.approx([0, 1, -1])


def test_scheffe_models_have_no_intercept_and_cross_with_process():
    study = study_from_config(
        {
            "name": "t",
            "factors": [
                {"name": "a", "type": "mixture"},
                {"name": "b", "type": "mixture"},
                {"name": "p", "low": 0, "high": 1},
            ],
            "model": {"mixture": "quadratic", "process": "quadratic", "mixture_process": "linear"},
        }
    )
    terms = model_terms(study)
    assert terms == [("a",), ("b",), ("a", "b"), ("a", "p"), ("b", "p"), ("p", "p")]
    frame = pd.DataFrame({"a": [0.5], "b": [0.5], "p": [0.5]})
    assert "Intercept" not in model_matrix(study, frame, terms).names


def test_term_text_round_trip():
    study = linear_study()
    assert parse_term("a*b", study) == ("a", "b")
    assert parse_term("a^2", study) == ("a", "a")
    assert term_label(("a", "a", "b")) == "a^2:b"
    with pytest.raises(StudyError, match="unknown factor"):
        parse_term("zzz", study)


def test_two_level_factors_get_no_curvature_terms():
    study = study_from_config(
        {
            "name": "t",
            "factors": [{"name": "a", "low": 0, "high": 1}, {"name": "g", "levels": [22, 27]}],
            "model": "quadratic",
        }
    )
    assert ("g", "g") not in model_terms(study)
    assert ("a", "a") in model_terms(study)


def test_alias_detection_includes_the_intercept():
    study = linear_study()
    frame = pd.DataFrame({"a": [0.0, 0.0, 10.0, 10.0], "b": [100.0, 200.0, 100.0, 200.0]})
    matrix = model_matrix(study, frame, [("a",), ("b",), ("a", "a")])
    assert ("Intercept", "a^2") in find_aliases(matrix)


# ------------------------------------------------------------------ fitting
def test_noise_free_truth_is_recovered_exactly():
    study = linear_study(model="quadratic")
    frame = grid_frame(lambda f: 50 + 2 * (f["a"] - 5) - 0.03 * (f["b"] - 150) + 0.4 * (f["a"] - 5) ** 2)
    fit = fit_response(study, frame, study.responses[0])
    assert fit.predict_mean(frame) == pytest.approx(frame["y"].to_numpy())
    assert fit_metrics(fit)["r2"] == pytest.approx(1)


def test_runs_without_results_are_left_out_and_reported():
    study = linear_study(model="linear")
    frame = grid_frame(lambda f: f["a"] + 0.1 * f["b"], noise=0.5)
    frame.loc[[0, 5], "y"] = np.nan
    fit = fit_response(study, frame, study.responses[0])
    assert len(fit.y) == len(frame) - 2
    assert fit.total_runs == len(frame)
    assert any("2 of 25 runs" in check.message for check in interpret_fit(fit))


def test_inestimable_model_gives_a_clear_error():
    study = linear_study(model="quadratic")
    frame = pd.DataFrame({"a": [0.0, 0.0, 10.0, 10.0, 5.0], "b": [100.0, 200.0, 100.0, 200.0, 150.0], "y": [1.0, 2, 3, 4, 5]})
    with pytest.raises(StudyError, match="can't be estimated"):
        fit_response(study, frame, study.responses[0])


def test_anova_partial_tests_match_coefficient_tests():
    study = linear_study(model="quadratic")
    frame = grid_frame(lambda f: 2 * f["a"] + 0.1 * f["b"] + 0.3 * f["a"] ** 2, noise=1.0)
    fit = fit_response(study, frame, study.responses[0])
    table = anova_table(fit).set_index("source")
    for term in ["a", "b", "a:b", "a^2", "b^2"]:
        position = fit.matrix.names.index(term)
        assert table.loc[term, "p"] == pytest.approx(np.asarray(fit.result.pvalues)[position])
        assert table.loc[term, "F"] == pytest.approx(np.asarray(fit.result.tvalues)[position] ** 2)


def test_model_reduction_finds_the_real_terms():
    study = linear_study(model={"process": "quadratic", "reduce": True, "alpha": 0.05})
    frame = grid_frame(lambda f: 4 * (f["a"] - 5) + 0.5 * (f["a"] - 5) ** 2, noise=0.5, seed=1, levels=6)
    fit = fit_response(study, frame, study.responses[0])
    kept = [term_label(term) for term in fit.terms]
    assert "a" in kept and "a^2" in kept
    assert "b^2" not in kept and "a:b" not in kept
    assert fit.removed_terms


def test_reduction_keeps_the_model_hierarchical():
    study = linear_study()
    assert ("a",) not in _removable_terms(study, [("a",), ("b",), ("a", "b")])
    assert ("a", "b") in _removable_terms(study, [("a",), ("b",), ("a", "b")])
    assert ("a",) not in _removable_terms(study, [("a",), ("a", "a")])


def test_reduction_never_removes_mixture_components():
    study = study_from_config(
        {"name": "t", "factors": [{"name": "a", "type": "mixture"}, {"name": "b", "type": "mixture"}, {"name": "p", "low": 0, "high": 1}]}
    )
    terms = [("a",), ("b",), ("a", "b"), ("a", "p"), ("b", "p"), ("p", "p")]
    removable = _removable_terms(study, terms)
    assert ("a",) not in removable and ("b",) not in removable
    assert ("a", "p") in removable and ("b", "p") in removable
    # The last term carrying p's main effect stays while p^2 is still in the model.
    assert ("a", "p") not in _removable_terms(study, [("a",), ("b",), ("a", "p"), ("p", "p")])


def test_log_transform_predictions_come_back_in_original_units():
    study = study_from_config(
        {
            "name": "t",
            "factors": [{"name": "a", "low": 0, "high": 10}, {"name": "b", "low": 100, "high": 200}],
            "responses": [{"name": "y", "transform": "log"}],
            "model": "linear",
        }
    )
    frame = grid_frame(lambda f: np.exp(1 + 0.2 * f["a"]))
    fit = fit_response(study, frame, study.responses[0])
    assert fit.predict(frame)["predicted"].to_numpy() == pytest.approx(frame["y"].to_numpy())
    bad = frame.copy()
    bad.loc[0, "y"] = -1.0
    with pytest.raises(StudyError, match="log"):
        fit_response(study, bad, study.responses[0])


def test_prediction_intervals_are_wider_than_confidence_intervals():
    study = linear_study(model="quadratic")
    frame = grid_frame(lambda f: f["a"] + 0.1 * f["b"], noise=1.0)
    fit = fit_response(study, frame, study.responses[0])
    prediction = fit.predict(frame.iloc[:3])
    assert (prediction["pi_high"] > prediction["ci_high"]).all()
    assert (prediction["pi_low"] < prediction["ci_low"]).all()


def test_prediction_interval_coverage_is_about_right():
    """Over many simulated experiments, 95% intervals should contain about 95% of new runs."""
    study = linear_study(model="quadratic")
    generator = np.random.default_rng(0)
    hits, total = 0, 0
    point = pd.DataFrame({"a": [7.0], "b": [120.0]})

    def truth(frame):
        return 3 * frame["a"] + 0.1 * frame["b"]

    for seed in range(300):
        frame = grid_frame(truth, seed=seed, noise=2.0, levels=3)
        prediction = fit_response(study, frame, study.responses[0]).predict(point).iloc[0]
        new_run = truth(point).iloc[0] + generator.normal(0, 2.0)
        hits += prediction["pi_low"] <= new_run <= prediction["pi_high"]
        total += 1
    assert 0.91 <= hits / total <= 0.98


# ------------------------------------------------------------------ interpretation
def _checks(fit):
    return {check.topic: check for check in interpret_fit(fit)}


def test_a_good_fit_gets_a_clean_verdict():
    study = linear_study(model="quadratic")
    frame = grid_frame(lambda f: 3 * f["a"] + 0.1 * f["b"] + 0.5 * f["a"] ** 2, noise=1.0, seed=2)
    repeats = frame.iloc[[0, 12, 24]].copy()
    repeats["y"] = repeats["y"] + [0.5, -0.5, 0.3]
    frame = pd.concat([frame, repeats], ignore_index=True)
    checks = _checks(fit_response(study, frame, study.responses[0]))
    assert checks["Overall model"].status == "OK"
    assert checks["Predictive power"].status == "OK"
    assert checks["Lack of fit"].status == "OK"


def test_missing_curvature_is_flagged_as_lack_of_fit():
    study = linear_study(model="linear")
    frame = grid_frame(lambda f: 2 * (f["a"] - 5) ** 2, noise=0.2, seed=3)
    frame = pd.concat([frame] * 2, ignore_index=True)
    frame["y"] = frame["y"] + np.random.default_rng(4).normal(0, 0.2, len(frame))
    assert _checks(fit_response(study, frame, study.responses[0]))["Lack of fit"].status == "CHECK"


def test_noise_alone_is_not_declared_a_real_model():
    study = linear_study(model="linear")
    frame = grid_frame(lambda f: 0 * f["a"], noise=1.0, seed=8)
    assert _checks(fit_response(study, frame, study.responses[0]))["Overall model"].status == "CHECK"


def test_a_bad_outlier_is_flagged():
    study = linear_study(model="linear")
    frame = grid_frame(lambda f: f["a"] + 0.1 * f["b"], noise=0.3, seed=1)
    frame["run"] = np.arange(1, len(frame) + 1)
    frame.loc[12, "y"] += 15
    fit = fit_response(study, frame, study.responses[0])
    diagnostics = run_diagnostics(fit)
    flagged = diagnostics.loc[diagnostics["studentised_residual"].abs() > 3.5, "run"].tolist()
    assert 13 in flagged
    assert "Outliers" in _checks(fit)


def test_drift_over_run_order_is_flagged():
    study = linear_study(model="linear")
    frame = grid_frame(lambda f: f["a"] + 0.1 * f["b"], levels=6)
    frame["run"] = np.random.default_rng(1).permutation(len(frame)) + 1
    frame["y"] = frame["y"] + 0.4 * frame["run"]
    assert "Run order" in _checks(fit_response(study, frame, study.responses[0]))


def test_box_cox_suggests_log_when_scatter_grows_with_the_mean():
    study = linear_study(model="linear")
    frame = grid_frame(lambda f: np.exp(1 + 0.3 * f["a"]), levels=6)
    frame["y"] = frame["y"] * np.exp(np.random.default_rng(1).normal(0, 0.15, len(frame)))
    assert suggest_transform(fit_response(study, frame, study.responses[0]))["suggestion"] == "log"


def test_curvature_test_compares_centre_points_with_corners():
    study = linear_study(model="interaction")
    corners = [(0.0, 100.0), (0.0, 200.0), (10.0, 100.0), (10.0, 200.0)]
    frame = pd.DataFrame(corners + [(5.0, 150.0)] * 4, columns=["a", "b"])
    frame["y"] = [10.0, 10.2, 10.1, 9.9, 20.0, 20.2, 19.8, 20.1]
    result = curvature_test(fit_response(study, frame, study.responses[0]))
    assert result["difference"] == pytest.approx(10.05 - 20.025, abs=1e-6)
    assert result["p"] < 0.01


def test_fit_all_skips_empty_responses_with_a_note():
    study = study_from_config(
        {
            "name": "t",
            "factors": [{"name": "a", "low": 0, "high": 10}, {"name": "b", "low": 100, "high": 200}],
            "responses": [{"name": "y"}, {"name": "z"}],
            "model": "linear",
        }
    )
    frame = grid_frame(lambda f: f["a"], noise=0.1)
    frame["z"] = np.nan
    fits, notes = fit_all(study, frame)
    assert [fit.response.name for fit in fits] == ["y"]
    assert "'z'" in notes[0]


# ------------------------------------------------------------------ desirability and optimisation
@pytest.mark.parametrize(
    "goal, target, value, expected",
    [
        ("maximise", None, 50, 0.5),
        ("maximise", None, -10, 0.0),
        ("maximise", None, 120, 1.0),
        ("minimise", None, 25, 0.75),
        ("minimise", None, 120, 0.0),
        ("target", 40, 40, 1.0),
        ("target", 40, 20, 0.5),
        ("target", 40, 70, 0.5),
        ("target", 40, 120, 0.0),
        ("in_range", None, 50, 1.0),
        ("in_range", None, 120, 0.0),
    ],
)
def test_desirability_values(goal, target, value, expected):
    response = Response(name="y", goal=goal, low=0, high=100, target=target)
    limits = Limits(response, 0.0, 100.0, target)
    d, _ = desirability(limits, np.array([float(value)]))
    assert d[0] == pytest.approx(expected)


def test_shape_bends_the_desirability_curve():
    response = Response(name="y", goal="maximise", low=0, high=100, shape=2)
    d, _ = desirability(Limits(response, 0.0, 100.0, None), np.array([50.0]))
    assert d[0] == pytest.approx(0.25)


def test_overall_desirability_is_a_weighted_geometric_mean():
    study = study_from_config(
        {
            "name": "t",
            "factors": [{"name": "a", "low": 0, "high": 1}, {"name": "b", "low": 0, "high": 1}],
            "responses": [
                {"name": "y", "goal": "maximise", "low": 0, "high": 1, "importance": 3},
                {"name": "z", "goal": "maximise", "low": 0, "high": 1, "importance": 1},
            ],
            "model": "linear",
        }
    )
    grid = pd.DataFrame({"a": [0, 0, 1, 1, 0.5], "b": [0, 1, 0, 1, 0.5]}).astype(float)
    frame = grid.assign(y=grid["a"], z=grid["b"])
    fits = [fit_response(study, frame, response) for response in study.responses]
    limits = [Limits(response, 0.0, 1.0, None) for response in study.responses]
    score, _, _ = overall_desirability(fits, limits, pd.DataFrame({"a": [0.5], "b": [0.25]}))
    assert score[0] == pytest.approx((0.5**3 * 0.25) ** (1 / 4))
    zero, _, _ = overall_desirability(fits, limits, pd.DataFrame({"a": [0.0], "b": [1.0]}))
    assert zero[0] == pytest.approx(0)  # one unacceptable response fails the whole setting


def _quadratic_fit(goal="maximise", **limits):
    study = study_from_config(
        {
            "name": "t",
            "seed": 1,
            "factors": [{"name": "a", "low": 0, "high": 10}, {"name": "b", "low": 0, "high": 10}],
            "responses": [{"name": "y", "goal": goal, **limits}],
            "model": "quadratic",
        }
    )
    values = np.linspace(0, 10, 5)
    a, b = np.meshgrid(values, values)
    frame = pd.DataFrame({"a": a.ravel(), "b": b.ravel()})
    frame["y"] = 100 - (frame["a"] - 3) ** 2 - 2 * (frame["b"] - 7) ** 2
    return study, fit_response(study, frame, study.responses[0])


def test_optimiser_finds_a_known_peak():
    study, fit = _quadratic_fit(low=0, high=100)
    best = find_optimum(study, [fit]).iloc[0]
    assert best["a"] == pytest.approx(3, abs=0.01)
    assert best["b"] == pytest.approx(7, abs=0.01)
    assert best["y_predicted"] == pytest.approx(100, abs=0.01)


def test_optimiser_hits_a_target():
    study, fit = _quadratic_fit(goal="target", target=80, low=50, high=95)
    best = find_optimum(study, [fit]).iloc[0]
    assert best["y_predicted"] == pytest.approx(80, abs=0.01)
    assert best["desirability"] == pytest.approx(1, abs=0.001)


def test_optimiser_is_reproducible():
    study, fit = _quadratic_fit(goal="target", target=80, low=50, high=95)
    pd.testing.assert_frame_equal(find_optimum(study, [fit], seed=5), find_optimum(study, [fit], seed=5))


def test_optimiser_keeps_mixtures_on_their_total_and_within_limits():
    study = study_from_config(
        {
            "name": "t",
            "seed": 1,
            "factors": [
                {"name": "a", "type": "mixture", "low": 0.2, "high": 0.7},
                {"name": "b", "type": "mixture", "low": 0.1, "high": 0.5},
                {"name": "c", "type": "mixture", "low": 0.1, "high": 0.4},
            ],
            "responses": [{"name": "y", "goal": "maximise", "low": 0, "high": 10}],
            "model": {"mixture": "quadratic"},
        }
    )
    grid = pd.DataFrame(
        [(0.7, 0.2, 0.1), (0.7, 0.1, 0.2), (0.5, 0.1, 0.4), (0.2, 0.4, 0.4), (0.2, 0.5, 0.3), (0.4, 0.5, 0.1), (0.4, 0.3, 0.3), (0.5, 0.3, 0.2)],
        columns=list("abc"),
    )
    grid["y"] = 2 * grid["a"] + 3 * grid["b"] + 9 * grid["c"]  # the best blend has c at its limit
    fit = fit_response(study, grid, study.responses[0])
    best = find_optimum(study, [fit]).iloc[0]
    assert best[["a", "b", "c"]].sum() == pytest.approx(1)
    assert best["c"] == pytest.approx(0.4, abs=0.002)
    assert study.is_feasible(pd.DataFrame([best])).all()


def test_optimiser_chooses_the_best_categorical_level_and_respects_constraints():
    study = study_from_config(
        {
            "name": "t",
            "seed": 1,
            "factors": [
                {"name": "a", "low": 0, "high": 10},
                {"name": "b", "low": 0, "high": 10},
                {"name": "c", "levels": ["worse", "better"]},
            ],
            "constraints": [{"expr": "a + b", "max": 12}],
            "responses": [{"name": "y", "goal": "maximise", "low": 0, "high": 40}],
            "model": "linear",
        }
    )
    grid = pd.MultiIndex.from_product([[0.0, 5, 10], [0.0, 5, 10], ["worse", "better"]], names=list("abc")).to_frame(index=False)
    grid["y"] = grid["a"] + 2 * grid["b"] + 5 * (grid["c"] == "better")
    fit = fit_response(study, grid, study.responses[0])
    best = find_optimum(study, [fit]).iloc[0]
    assert best["c"] == "better"
    assert best["a"] + best["b"] == pytest.approx(12, abs=0.01)
    assert best["b"] == pytest.approx(10, abs=0.01)


def test_optimising_needs_at_least_one_goal():
    study = study_from_config(
        {
            "name": "t",
            "factors": [{"name": "a", "low": 0, "high": 10}, {"name": "b", "low": 100, "high": 200}],
            "responses": [{"name": "y"}],
            "model": "linear",
        }
    )
    frame = grid_frame(lambda f: f["a"])
    with pytest.raises(StudyError, match="goal"):
        find_optimum(study, [fit_response(study, frame, study.responses[0])])
