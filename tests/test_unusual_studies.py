"""Check that unusual study shapes work from design to best settings, including every figure.

The toolkit is meant for any process, so a study with one factor, or only categorical
factors, or percentages rather than fractions, has to work as well as the typical case.
"""

import numpy as np
import pytest

from doekit.analysis import fit_all
from doekit.designs import create_design
from doekit.evaluate import evaluate_design
from doekit.optimise import find_optimum
from doekit.study import study_from_config
from doekit.workflow import save_fit_outputs, save_optimum_outputs

RESPONSES = [{"name": "y", "goal": "maximise"}, {"name": "z", "goal": "target", "target": 50, "low": 20, "high": 80}]


STUDIES = {
    "one_continuous_factor": dict(
        factors=[{"name": "temp", "low": 20, "high": 40, "units": "°C"}],
        design={"type": "full_factorial", "levels": 5, "repeats": 2},
        model={"process": "quadratic"},
    ),
    "two_component_blend": dict(
        factors=[{"name": "a", "type": "mixture"}, {"name": "b", "type": "mixture"}],
        design={"type": "simplex_lattice", "degree": 4, "replicate_runs": 3},
        model={"mixture": "quadratic"},
    ),
    "four_component_blend": dict(
        factors=[{"name": n, "type": "mixture", "low": 0.05, "high": 0.6} for n in "abcd"],
        design={"type": "i_optimal"},
        model={"mixture": "quadratic"},
    ),
    "blend_in_percent": dict(
        mixture_total=100,
        factors=[
            {"name": "a", "type": "mixture", "low": 20, "high": 70, "units": "%"},
            {"name": "b", "type": "mixture", "low": 10, "high": 50, "units": "%"},
            {"name": "c", "type": "mixture", "low": 10, "high": 40, "units": "%"},
        ],
        design={"type": "extreme_vertices", "check_points": True, "replicate_runs": 3},
        model={"mixture": "quadratic"},
    ),
    "only_categorical_factors": dict(
        factors=[{"name": "c", "levels": ["x", "y", "z"]}, {"name": "d", "levels": ["p", "q"]}],
        design={"type": "full_factorial", "repeats": 2},
        model={"process": "interaction"},
    ),
    "numeric_with_discrete_and_categorical": dict(
        factors=[
            {"name": "p", "low": 1, "high": 5},
            {"name": "g", "levels": [22, 25, 27]},
            {"name": "c", "levels": ["x", "y"]},
        ],
        design={"type": "d_optimal", "runs": 24},
        model={"process": "quadratic"},
    ),
    "blend_with_categorical_process_factor": dict(
        factors=[
            {"name": "a", "type": "mixture", "low": 0.2},
            {"name": "b", "type": "mixture", "low": 0.1},
            {"name": "c", "type": "mixture", "low": 0.1},
            {"name": "x", "levels": ["k", "m"]},
        ],
        design={"type": "i_optimal"},
        model={"mixture": "quadratic", "process": "linear", "mixture_process": "linear"},
    ),
    "screening_with_no_spare_runs": dict(
        factors=[{"name": f"x{i}", "low": 0, "high": 1} for i in range(7)],
        design={"type": "plackett_burman"},
        model={"process": "linear"},
    ),
    "units_on_a_negative_scale": dict(
        factors=[{"name": "t", "low": -80, "high": -20}, {"name": "ph", "low": 4, "high": 9}],
        design={"type": "central_composite"},
        model={"process": "quadratic"},
    ),
    "doehlert_two_factors": dict(
        factors=[{"name": "t", "low": 0, "high": 1}, {"name": "u", "low": 0, "high": 1}],
        design={"type": "doehlert", "center_points": 3},
        model={"process": "quadratic"},
    ),
    "sobol_with_a_constraint": dict(
        factors=[{"name": "t", "low": 0, "high": 1}, {"name": "u", "low": 0, "high": 1}],
        constraints=[{"expr": "t + u", "max": 1.5}],
        design={"type": "sobol", "runs": 16},
        model={"process": "quadratic"},
    ),
}


def _fill_results(study, frame, seed=0):
    generator = np.random.default_rng(seed)
    coded = study.coded(frame)
    sheet = frame.copy()
    for response in study.response_names:
        signal = np.zeros(len(frame))
        for factor in study.factors:
            if factor.is_numeric:
                signal = signal + generator.uniform(-8, 8) * coded[factor.name]
            else:
                level = (frame[factor.name] == factor.levels[0]).to_numpy(dtype=float)
                signal = signal + generator.uniform(-8, 8) * level
        sheet[response] = 50 + signal + generator.normal(0, 1, len(frame))
    return sheet


@pytest.mark.parametrize("name", STUDIES)
def test_whole_workflow_on_an_unusual_study(name, tmp_path):
    study = study_from_config({"name": name, "seed": 3, "responses": RESPONSES, **STUDIES[name]}, tmp_path)
    design = create_design(study)
    quality = evaluate_design(study, design.frame, design.terms)
    assert quality.estimable, quality.warnings
    assert study.is_feasible(design.frame).all()

    sheet = _fill_results(study, design.frame)
    fits, notes = fit_all(study, sheet)
    assert len(fits) == 2 and not notes
    for fit in fits:
        assert "ANALYSIS" in save_fit_outputs(fit, tmp_path / fit.response.name)
    solutions = find_optimum(study, fits)
    assert study.is_feasible(solutions, tolerance=1e-3).all()
    if study.mixture_factors:
        total = solutions[study.mixture_names].sum(axis=1).to_numpy()
        assert total == pytest.approx(np.full(len(total), float(study.mixture_total)), rel=1e-6)
    assert "BEST SETTINGS" in save_optimum_outputs(study, fits, solutions, tmp_path / "optimum")
    assert (tmp_path / "optimum" / "solutions.csv").exists()
