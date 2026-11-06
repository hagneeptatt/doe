"""Check that adding runs chooses well, and never disturbs the runs already done."""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from doekit.analysis import coefficient_table, fit_response
from doekit.augment import augment_design, compare_quality, minimum_extra_runs
from doekit.designs import create_design
from doekit.designs.optimal import _swap_gains, design_score, exchange_search
from doekit.evaluate import evaluate_design
from doekit.model import model_matrix
from doekit.runsheet import build_runsheet, read_runsheet
from doekit.simulate import fill_blank_results
from doekit.study import StudyError, study_from_config

ROOT = Path(__file__).resolve().parents[1]
THREE = [{"name": f"x{i + 1}", "low": 0, "high": 10} for i in range(3)]
RESPONSE = [{"name": "y", "goal": "maximise", "low": 0, "high": 100}]


def study_with(factors=THREE, model="quadratic", design=None, **extra):
    config = {"name": "t", "seed": 1, "factors": factors, "responses": RESPONSE, "model": model}
    if design:
        config["design"] = design
    return study_from_config({**config, **extra})


def truth(frame):
    return 50 + 2 * frame["x1"] - 1.5 * frame["x2"] + 0.5 * frame["x3"] + 0.3 * frame["x1"] * frame["x2"] - 0.4 * (frame["x1"] - 5) ** 2


def finished_factorial(center_points=3):
    """A two-level screening experiment that has been done, with its results."""
    study = study_with(design={"type": "full_factorial", "center_points": center_points}, model="interaction")
    design = create_design(study)
    sheet = build_runsheet(study, design)
    sheet["y"] = truth(sheet)
    return sheet


# ------------------------------------------------------------------ the search with fixed runs
def _small_problem():
    study = study_from_config({"name": "t", "factors": THREE[:2], "model": "quadratic"})
    grid = pd.DataFrame(list(itertools.product([0.0, 5.0, 10.0], repeat=2)), columns=["x1", "x2"])
    terms = [("x1",), ("x2",), ("x1", "x2"), ("x1", "x1"), ("x2", "x2")]
    C = model_matrix(study, grid, terms, include_blocks=False).values
    fixed = C[[0, 2, 6, 8]]  # the four corners, as if already done
    return C, fixed


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_search_with_fixed_runs_finds_the_true_best(criterion):
    """Compare with brute force over every possible set of 3 new runs."""
    C, fixed = _small_problem()
    moments = C.T @ C / len(C)
    brute = min(
        design_score(C, np.array(subset), criterion, moments, fixed)
        for subset in itertools.combinations_with_replacement(range(9), 3)
    )
    chosen = exchange_search(C, 3, criterion, np.random.default_rng(0), starts=30, fixed=fixed, moments=moments)
    assert design_score(C, chosen, criterion, moments, fixed) == pytest.approx(brute, rel=1e-6)


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_swap_formula_with_fixed_runs_matches_direct_calculation(criterion):
    C, fixed = _small_problem()
    moments = C.T @ C / len(C)
    chosen = np.array([1, 3, 4, 5])
    gains = _swap_gains(C, chosen, criterion, moments, fixed)
    before = design_score(C, chosen, criterion, moments, fixed)
    for i, j in [(0, 7), (3, 7), (2, 6)]:
        trial = chosen.copy()
        trial[i] = j
        after = design_score(C, trial, criterion, moments, fixed)
        if criterion == "D":
            assert 1 + gains[i, j] == pytest.approx(np.exp(before - after), rel=1e-5)
        else:
            assert gains[i, j] == pytest.approx(before - after, rel=1e-4, abs=1e-6)


def test_chosen_runs_beat_random_ones():
    """The point of the feature: its runs should be better than picking by hand or at random."""
    sheet = finished_factorial()
    study = study_with()
    chosen = augment_design(study, sheet, 8, new_block=False)
    best = evaluate_design(study, chosen.design_after, chosen.terms)
    generator = np.random.default_rng(0)
    for _ in range(40):
        random_runs = pd.DataFrame(generator.choice([0.0, 5.0, 10.0], size=(8, 3)), columns=["x1", "x2", "x3"])
        trial = pd.concat([chosen.design_before, random_runs], ignore_index=True)
        quality = evaluate_design(study, trial, chosen.terms)
        if quality.estimable:
            assert np.mean(best.prediction_se**2) <= np.mean(quality.prediction_se**2) + 1e-9


# ------------------------------------------------------------------ the existing runs
def test_existing_runs_and_results_are_never_changed():
    sheet = finished_factorial()
    sheet.loc[0, "notes"] = "nozzle replaced"
    result = augment_design(study_with(), sheet, 6)
    original = sheet[["run", "x1", "x2", "x3", "y", "notes"]].reset_index(drop=True)
    kept = result.sheet.iloc[: len(sheet)][["run", "x1", "x2", "x3", "y", "notes"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, original, check_dtype=False)
    assert result.sheet["y"].iloc[len(sheet) :].isna().all()
    assert len(result.sheet) == len(sheet) + 6


def test_runs_and_blocks_are_numbered_correctly():
    sheet = finished_factorial()
    result = augment_design(study_with(), sheet, 6)
    assert result.sheet["run"].tolist() == list(range(1, len(sheet) + 7))
    assert (result.sheet["block"].iloc[: len(sheet)] == 1).all()
    assert (result.sheet["block"].iloc[len(sheet) :] == 2).all()
    assert result.sheet["std_order"].is_unique

    # A second round goes in a third block, and picks different runs.
    second = augment_design(study_with(), result.sheet, 4)
    assert second.block == 3
    assert (second.sheet["block"].iloc[-4:] == 3).all()
    assert second.sheet["run"].tolist() == list(range(1, len(sheet) + 11))


def test_same_session_adds_no_block():
    sheet = finished_factorial()
    result = augment_design(study_with(), sheet, 6, new_block=False)
    assert "block" not in result.sheet
    assert result.block is None


def test_failed_runs_stay_in_the_sheet_but_do_not_count():
    sheet = finished_factorial()
    sheet.loc[2, "y"] = np.nan
    sheet.loc[2, "notes"] = "print failed"
    result = augment_design(study_with(), sheet, 6)
    assert result.ignored_runs == 1
    assert result.counted_runs == len(sheet) - 1
    assert len(result.design_before) == len(sheet) - 1
    assert result.sheet.loc[2, "notes"] == "print failed"
    assert any("no result" in note for note in result.notes)


def test_a_plan_with_no_results_yet_can_be_enlarged():
    study = study_with(design={"type": "box_behnken"})
    sheet = build_runsheet(study, create_design(study))
    result = augment_design(study, sheet, 5, new_block=False)
    assert result.counted_runs == len(sheet)
    assert result.ignored_runs == 0


def test_the_same_inputs_give_the_same_runs():
    sheet = finished_factorial()
    first = augment_design(study_with(), sheet, 6)
    second = augment_design(study_with(), sheet, 6)
    pd.testing.assert_frame_equal(first.new_runs, second.new_runs)


# ------------------------------------------------------------------ what the new runs make possible
def test_screening_study_can_be_upgraded_to_find_curvature():
    """The classic route: a two-level experiment, then runs that make a quadratic model possible."""
    sheet = finished_factorial()
    study = study_with(model="quadratic")
    before = evaluate_design(study, sheet[["x1", "x2", "x3"]], [t for t in augment_design(study, sheet, 6).terms])
    assert not before.estimable

    result = augment_design(study, sheet, 6)
    filled = result.sheet.copy()
    filled["y"] = truth(filled)
    fit = fit_response(study, filled, study.responses[0])
    # With no noise, the model has to recover the truth exactly at any setting.
    point = pd.DataFrame({"x1": [3.0], "x2": [7.0], "x3": [2.0]})
    assert fit.predict_mean(point)[0] == pytest.approx(truth(point).iloc[0], abs=1e-8)


def test_block_effect_protects_the_factor_effects():
    """A shift between sessions must not change the estimated effects."""
    sheet = finished_factorial()
    study = study_with(model="quadratic")
    result = augment_design(study, sheet, 8)
    clean = result.sheet.copy()
    clean["y"] = truth(clean)
    shifted = clean.copy()
    shifted.loc[shifted["block"] == 2, "y"] += 15.0

    def effects(frame):
        table = coefficient_table(fit_response(study, frame, study.responses[0]))
        return table[~table["group"].isin(["block", "Intercept"])].set_index("term")["coef"]

    pd.testing.assert_series_equal(effects(clean), effects(shifted))


def test_minimum_runs_is_exactly_what_is_needed():
    sheet = finished_factorial()
    study = study_with(model="quadratic")
    needed = minimum_extra_runs(study, sheet)
    assert needed >= 1
    assert len(augment_design(study, sheet, needed).new_runs) == needed
    with pytest.raises(StudyError, match=f"at least {needed} new runs"):
        augment_design(study, sheet, needed - 1 if needed > 1 else 0) if needed > 1 else augment_design(study, sheet, needed, replicate_runs=1)


def test_repeats_copy_earlier_runs_and_add_replicates():
    sheet = finished_factorial(center_points=0)
    study = study_with(model="interaction")
    result = augment_design(study, sheet, 4, replicate_runs=2, new_block=False)
    settings = {tuple(row) for row in sheet[["x1", "x2", "x3"]].to_numpy()}
    new = [tuple(row) for row in result.new_runs[["x1", "x2", "x3"]].to_numpy()]
    assert sum(point in settings for point in new) >= 2
    after = evaluate_design(study, result.design_after, result.terms)
    before = evaluate_design(study, result.design_before, result.terms)
    assert after.pure_error_df >= before.pure_error_df + 2


def test_comparison_table_shows_the_gain():
    sheet = finished_factorial()
    study = study_with(model="quadratic")
    table = compare_quality(study, augment_design(study, sheet, 8)).set_index("")
    assert table.loc["Spare runs to check the model", "Now"] == "can't fit this model"
    assert int(table.loc["Runs", "With the new runs"]) == len(sheet) + 8
    assert float(table.loc["Typical prediction error (x noise)", "With the new runs"]) > 0


# ------------------------------------------------------------------ limits on the new runs
def test_region_limits_only_the_new_runs():
    sheet = finished_factorial()
    result = augment_design(study_with(), sheet, 6, region={"x1": (2.0, 6.0)})
    assert result.new_runs["x1"].between(2.0, 6.0).all()
    assert result.sheet["x1"].iloc[: len(sheet)].tolist() == sheet["x1"].tolist()
    assert result.sheet["x1"].max() == 10.0


@pytest.mark.parametrize(
    "region, message",
    [
        ({"x1": (-5.0, 6.0)}, "beyond its limits"),
        ({"x1": (6.0, 2.0)}, "must be below"),
        ({"zz": (0.0, 1.0)}, "unknown factors"),
    ],
)
def test_bad_regions_are_explained(region, message):
    with pytest.raises(StudyError, match=message):
        augment_design(study_with(), finished_factorial(), 6, region=region)


def test_constraints_apply_to_new_runs():
    study = study_with(constraints=[{"expr": "x1 + x2", "max": 12}])
    sheet = pd.DataFrame({"x1": [0.0, 0.0, 10.0, 2.0], "x2": [0.0, 10.0, 0.0, 2.0], "x3": [0.0, 10.0, 5.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0]})
    result = augment_design(study, sheet, 14, new_block=False)
    assert (result.new_runs["x1"] + result.new_runs["x2"] <= 12 + 1e-9).all()


def test_blends_stay_on_their_total_and_within_limits():
    factors = [
        {"name": "a", "type": "mixture", "low": 0.2, "high": 0.7},
        {"name": "b", "type": "mixture", "low": 0.1, "high": 0.5},
        {"name": "c", "type": "mixture", "low": 0.1, "high": 0.4},
        {"name": "p", "low": 1, "high": 5},
    ]
    study = study_from_config(
        {"name": "t", "seed": 1, "factors": factors, "responses": RESPONSE, "design": {"type": "d_optimal", "runs": 16}, "model": {"mixture": "quadratic", "process": "linear"}}
    )
    sheet = build_runsheet(study, create_design(study))
    sheet["y"] = np.arange(len(sheet), dtype=float)
    result = augment_design(study, sheet, 6)
    assert result.new_runs[["a", "b", "c"]].sum(axis=1).to_numpy() == pytest.approx(np.ones(6), abs=1e-6)
    assert study.is_feasible(result.new_runs).all()


def test_categorical_and_discrete_factors_use_allowed_levels():
    factors = [{"name": "p", "low": 0, "high": 10}, {"name": "g", "levels": [22, 25, 27]}, {"name": "c", "levels": ["x", "y"]}]
    study = study_with(factors=factors, design={"type": "d_optimal", "runs": 18})
    sheet = build_runsheet(study, create_design(study))
    sheet["y"] = np.arange(len(sheet), dtype=float)
    result = augment_design(study, sheet, 6)
    assert set(result.new_runs["g"]) <= {22.0, 25.0, 27.0}
    assert set(result.new_runs["c"]) <= {"x", "y"}


def test_impossible_models_and_bad_inputs_are_explained():
    factors = [{"name": "p", "low": 0, "high": 10}, {"name": "g", "levels": [22, 27]}]
    study = study_with(factors=factors, model={"terms": ["p", "g", "g^2"]})
    sheet = pd.DataFrame({"p": [0.0, 10.0, 5.0], "g": [22.0, 27.0, 22.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(StudyError, match="can't be estimated even with more runs"):
        augment_design(study, sheet, 5)
    with pytest.raises(StudyError, match="at least 1 run"):
        augment_design(study_with(), finished_factorial(), 0)
    with pytest.raises(StudyError, match="repeats"):
        augment_design(study_with(), finished_factorial(), 2, replicate_runs=3)
    with pytest.raises(StudyError, match="Unknown goal"):
        augment_design(study_with(), finished_factorial(), 2, criterion="X")


def test_a_broken_block_column_is_explained():
    sheet = finished_factorial()
    sheet["block"] = 1.0
    sheet.loc[1, "block"] = np.nan
    with pytest.raises(StudyError, match="block column"):
        augment_design(study_with(), sheet, 6)


# ------------------------------------------------------------------ simulated results
def test_simulation_only_fills_blanks():
    config = (ROOT / "configs/examples/hydrogel_concentrations_simulated.yaml")
    from doekit.study import load_study

    study = load_study(config)
    design = create_design(study)
    sheet = build_runsheet(study, design)
    first = fill_blank_results(study, sheet)
    first.loc[0, "viability"] = 12.5  # a real measurement that must survive
    first.loc[0, "notes"] = "measured"
    result = augment_design(study, first, 3, new_block=False)
    second = fill_blank_results(study, result.sheet)
    assert second.loc[0, "viability"] == 12.5
    assert second.loc[0, "notes"] == "measured"
    assert second[study.response_names].notna().all().all()
    assert second["notes"].iloc[-1].startswith("SIMULATED")


# ------------------------------------------------------------------ the script
def test_script_adds_runs_and_keeps_a_copy(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    config = tmp_path / "study.yaml"
    config.write_text(
        "name: demo\nseed: 3\nfactors:\n"
        "  - {name: a, low: 0, high: 10}\n  - {name: b, low: 0, high: 10}\n  - {name: c, low: 0, high: 10}\n"
        "responses:\n  - {name: y, goal: maximise, low: 0, high: 10}\n"
        "design: {type: box_behnken}\nmodel: {process: quadratic}\n"
    )
    create = subprocess.run([sys.executable, str(ROOT / "scripts/1_create_design.py"), str(config)], capture_output=True, text=True)
    assert create.returncode == 0, create.stderr
    run_sheet = tmp_path / "data/demo/runsheet.csv"
    original = pd.read_csv(run_sheet)
    original["y"] = np.arange(len(original), dtype=float)
    original.to_csv(run_sheet, index=False)

    added = subprocess.run(
        [sys.executable, str(ROOT / "scripts/add_runs.py"), str(config), "--runs", "5", "--repeats", "1", "--region", "a=2:8"],
        capture_output=True,
        text=True,
    )
    assert added.returncode == 0, added.stderr
    assert "What they add" in added.stdout
    after = pd.read_csv(run_sheet)
    assert len(after) == len(original) + 5
    assert after["y"].iloc[: len(original)].tolist() == original["y"].tolist()
    assert after["y"].iloc[len(original) :].isna().all()
    backups = list(run_sheet.parent.glob("runsheet_before_adding_runs_*.csv"))
    assert len(backups) == 1
    assert len(pd.read_csv(backups[0])) == len(original)

    # The study still analyses, using the runs that have results.
    from doekit.study import load_study

    study = load_study(config)
    fit = fit_response(study, read_runsheet(study), study.responses[0])
    assert fit.total_runs == len(after)
    assert len(fit.y) == len(original)

    bad = subprocess.run([sys.executable, str(ROOT / "scripts/add_runs.py"), str(config), "--runs", "5", "--region", "a"], capture_output=True, text=True)
    assert bad.returncode == 1 and "Problem:" in bad.stderr and "Traceback" not in bad.stderr


def test_repeats_alone_can_link_a_new_session():
    """A repeat in a new block estimates the block effect, so no fresh runs are needed for it."""
    sheet = finished_factorial()
    study = study_with(model="interaction")
    assert minimum_extra_runs(study, sheet) == 1
    assert minimum_extra_runs(study, sheet, replicate_runs=2) == 2
    result = augment_design(study, sheet, 2, replicate_runs=2)
    settings = {tuple(row) for row in sheet[["x1", "x2", "x3"]].to_numpy()}
    assert all(tuple(row) in settings for row in result.new_runs[["x1", "x2", "x3"]].to_numpy())
    assert evaluate_design(study, result.design_after, result.terms).estimable


def test_more_repeats_than_distinct_runs_go_round_again():
    sheet = finished_factorial(center_points=0).head(5)
    study = study_with(model="linear")
    result = augment_design(study, sheet, 8, replicate_runs=8, new_block=False)
    counts = result.new_runs[["x1", "x2", "x3"]].value_counts()
    assert counts.max() - counts.min() <= 1


def test_repeats_must_be_inside_the_region():
    sheet = finished_factorial(center_points=0)
    with pytest.raises(StudyError, match="can be repeated"):
        augment_design(study_with(model="interaction"), sheet, 4, replicate_runs=2, region={"x1": (2.0, 6.0)})


def test_new_runs_are_chosen_knowing_about_the_repeats():
    """With a repeat already taking a slot, the fresh runs mustn't simply copy it."""
    sheet = finished_factorial()
    study = study_with()
    result = augment_design(study, sheet, 8, replicate_runs=2)
    quality = evaluate_design(study, result.design_after, result.terms)
    assert quality.estimable
    assert quality.pure_error_df >= evaluate_design(study, result.design_before, result.terms).pure_error_df + 1


def test_region_can_limit_a_blend_component():
    factors = [
        {"name": "a", "type": "mixture", "low": 0.2, "high": 0.7},
        {"name": "b", "type": "mixture", "low": 0.1, "high": 0.5},
        {"name": "c", "type": "mixture", "low": 0.1, "high": 0.4},
    ]
    study = study_from_config(
        {"name": "t", "seed": 1, "factors": factors, "responses": RESPONSE, "design": {"type": "extreme_vertices"}, "model": {"mixture": "quadratic"}}
    )
    sheet = build_runsheet(study, create_design(study))
    sheet["y"] = np.arange(len(sheet), dtype=float)
    result = augment_design(study, sheet, 4, region={"a": (0.3, 0.5)}, new_block=False)
    assert result.new_runs["a"].between(0.3 - 1e-6, 0.5 + 1e-6).all()
    assert result.new_runs[["a", "b", "c"]].sum(axis=1).to_numpy() == pytest.approx(np.ones(4), abs=1e-6)
    assert len(pd.concat([result.sheet.iloc[: len(sheet)]])) == len(sheet)


def test_region_can_limit_a_discrete_factor():
    factors = [{"name": "p", "low": 0, "high": 10}, {"name": "g", "levels": [22, 25, 27, 30]}]
    study = study_with(factors=factors, design={"type": "d_optimal", "runs": 14})
    sheet = build_runsheet(study, create_design(study))
    sheet["y"] = np.arange(len(sheet), dtype=float)
    result = augment_design(study, sheet, 4, region={"g": (25.0, 30.0)})
    assert set(result.new_runs["g"]) <= {25.0, 27.0, 30.0}
    with pytest.raises(StudyError, match="at least 2 of its levels"):
        augment_design(study, sheet, 4, region={"g": (26.0, 28.0)})


def test_categorical_factors_cannot_have_a_region():
    factors = [{"name": "p", "low": 0, "high": 10}, {"name": "c", "levels": ["x", "y"]}]
    study = study_with(factors=factors, model="linear", design={"type": "full_factorial"})
    sheet = build_runsheet(study, create_design(study))
    sheet["y"] = np.arange(len(sheet), dtype=float)
    with pytest.raises(StudyError, match="categorical"):
        augment_design(study, sheet, 2, region={"c": (0.0, 1.0)})


def test_widening_a_limit_lets_new_runs_reach_it():
    sheet = finished_factorial()
    wider = [{"name": "x1", "low": 0, "high": 20}, *THREE[1:]]
    study = study_with(factors=wider)
    result = augment_design(study, sheet, 8)
    assert result.new_runs["x1"].max() > 10
    assert result.sheet["x1"].iloc[: len(sheet)].max() == 10
