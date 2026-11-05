"""Run the whole workflow from config files, as a user would, in a temporary project folder."""

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from doekit.advisor import OBJECTIVES, recommend_designs
from doekit.designs import create_design
from doekit.expressions import evaluate
from doekit.runsheet import read_runsheet, read_runsheet_bytes
from doekit.study import StudyError, load_study, study_from_config
from doekit.workflow import analyse_step, compare_step, create_design_step, optimise_step, simulate_step

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "configs/examples").glob("*.yaml"))
TEMPLATES = sorted((ROOT / "configs/templates").glob("*.yaml"))
PRIVATE_DATA = ROOT / "archive/private"


@pytest.fixture
def project(tmp_path):
    """A copy of the project's configs and any private data, so tests can't touch real files."""
    (tmp_path / "pyproject.toml").write_text("")
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    if PRIVATE_DATA.exists():
        shutil.copytree(PRIVATE_DATA, tmp_path / "archive/private", ignore=shutil.ignore_patterns("figures"))
    return tmp_path


def test_simulated_example_recommends_settings_that_are_truly_good(project):
    """The test that matters most: the simulation's formulas are known, so check the answer."""
    config = project / "configs/examples/hydrogel_bioink_simulated.yaml"
    create_design_step(config)
    simulate_step(config)
    analyse_step(config)
    solutions = optimise_step(config)
    study = load_study(config)
    best = solutions.iloc[0]
    variables = {name: np.array([best[name]]) for name in study.factor_names}
    for response in study.responses:
        truth = float(np.squeeze(evaluate(study.simulation[response.name]["formula"], variables)))
        assert best[f"{response.name}_pi_low"] <= truth <= best[f"{response.name}_pi_high"], response.name
    assert best["desirability"] > 0.7
    assert (study.results_dir / "optimum" / "report.txt").exists()
    assert (study.results_dir / "viability" / "effects.svg").exists()


def test_concentration_example_with_categorical_factor_and_constraint(project):
    config = project / "configs/examples/hydrogel_concentrations_simulated.yaml"
    create_design_step(config)
    simulate_step(config)
    analyse_step(config)
    best = optimise_step(config).iloc[0]
    assert best["gelma_wt"] + best["alginate_wt"] <= 16 + 1e-3
    assert best["crosslinker"] in {"CaCl2", "BaCl2"}


@pytest.mark.skipif(not PRIVATE_DATA.exists(), reason="the original data isn't in the public repository")
@pytest.mark.parametrize("name", ["mew_fibre_diameter", "jetting_hama_large_valve"])
def test_archived_data_can_be_reanalysed(project, name):
    config = project / f"archive/private/configs/{name}.yaml"
    assert analyse_step(config)
    assert len(optimise_step(config))


def test_existing_data_config_explains_that_there_is_no_design_to_create(project):
    with pytest.raises(StudyError, match="existing data"):
        create_design_step(project / "configs/templates/existing_data.yaml")


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.stem)
def test_every_template_works_from_design_to_optimum(project, path):
    """Copying a template is how people start, so every one has to work as shipped."""
    config = project / "configs/templates" / path.name
    study = load_study(config)
    if study.design.get("type") == "existing":
        return
    create_design_step(config)
    sheet = read_runsheet(study)
    assert sheet[study.response_names].isna().all().all()
    # Fill in made-up results based on the factors, then finish the workflow.
    generator = np.random.default_rng(0)
    coded = study.coded(sheet)
    for response in study.response_names:
        signal = np.zeros(len(sheet))
        for factor in study.factors:
            if factor.is_numeric:
                signal = signal + generator.uniform(-2, 2) * coded[factor.name]
            else:
                signal = signal + (sheet[factor.name] == factor.levels[0]).to_numpy(dtype=float)
        sheet[response] = 50 + signal + generator.normal(0, 0.5, len(sheet))
    sheet.to_csv(study.runsheet_path, index=False)
    assert analyse_step(config)
    assert len(optimise_step(config))


def test_a_template_exists_for_each_family_of_design():
    kinds = {load_study(path).design.get("type") for path in TEMPLATES}
    expected = {
        "fractional_factorial", "plackett_burman", "box_behnken", "central_composite",
        "i_optimal", "simplex_centroid", "latin_hypercube", "existing",
    }  # fmt: skip
    assert expected <= kinds


def test_results_are_protected_from_being_overwritten(project):
    config = project / "configs/examples/hydrogel_bioink_simulated.yaml"
    create_design_step(config)
    study = load_study(config)
    before = study.runsheet_path.read_bytes()
    with pytest.raises(StudyError, match="may contain your results"):
        create_design_step(config)
    simulate_step(config)
    filled = study.runsheet_path.read_bytes()
    assert filled != before
    with pytest.raises(StudyError, match="already has results"):
        simulate_step(config)
    assert study.runsheet_path.read_bytes() == filled


def test_real_studies_cannot_be_filled_with_simulated_results(project):
    config = project / "configs/templates/optimise_box_behnken.yaml"
    create_design_step(config)
    with pytest.raises(StudyError, match="no 'simulation' section"):
        simulate_step(config)


def test_simulated_rows_say_they_are_simulated(project):
    config = project / "configs/examples/hydrogel_bioink_simulated.yaml"
    create_design_step(config)
    simulate_step(config)
    sheet = read_runsheet(load_study(config))
    assert sheet["notes"].str.contains("SIMULATED").all()


def test_compare_designs_skips_ones_that_cannot_estimate_the_model(project):
    config = project / "configs/templates/optimise_box_behnken.yaml"
    table = compare_step(config, ["box_behnken", "central_composite", "i_optimal:runs=24", "plackett_burman"])
    assert set(table["design"]) == {"box_behnken", "central_composite", "i_optimal:runs=24"}


# ------------------------------------------------------------------ the scripts themselves
@pytest.mark.parametrize(
    "script", ["1_create_design", "2_analyse_results", "3_optimise", "recommend_design", "compare_designs", "simulate_results"]
)
def test_scripts_show_help(script):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout


def test_scripts_show_problems_without_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/1_create_design.py"), str(tmp_path / "missing.yaml")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Problem:" in result.stderr
    assert "Traceback" not in result.stderr


def test_design_list_flag():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/1_create_design.py"), "--list"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "box_behnken" in result.stdout and "i_optimal" in result.stdout


# ------------------------------------------------------------------ run sheets from Excel
def _simple_study():
    return study_from_config(
        {
            "name": "t",
            "factors": [{"name": "temp", "low": 0, "high": 10}, {"name": "kind", "levels": ["x", "y"]}],
            "responses": [{"name": "y"}],
        }
    )


@pytest.mark.parametrize(
    "text",
    [
        "run,temp,kind,y,notes\n1,1.5,x,2.5,\n2,3.5,y,4.5,\n",
        "run;temp;kind;y;notes\n1;1,5;x;2,5;\n2;3,5;y;4,5;\n",  # Dutch/European Excel
        "run\ttemp\tkind\ty\tnotes\n1\t1.5\tx\t2.5\t\n2\t3.5\ty\t4.5\t\n",
        "﻿run,temp,kind,y,notes\n1,1.5,x,2.5,\n2,3.5,y,4.5,\n",  # Excel's byte order mark
        "run, temp ,kind,y ,notes\n1,1.5, x ,2.5,\n2,3.5,y,4.5,\n\n,,,,\n",  # stray spaces and blank rows
    ],
    ids=["plain", "european", "tabs", "bom", "messy"],
)
def test_run_sheets_from_different_excel_settings(text):
    sheet = read_runsheet_bytes(_simple_study(), text.encode("utf-8"))
    assert sheet["temp"].tolist() == [1.5, 3.5]
    assert sheet["y"].tolist() == [2.5, 4.5]
    assert sheet["kind"].tolist() == ["x", "y"]


def test_run_sheet_problems_name_the_column_and_row():
    study = _simple_study()
    with pytest.raises(StudyError, match="no column for: temp"):
        read_runsheet_bytes(study, b"run,kind,y\n1,x,1\n")
    with pytest.raises(StudyError, match="no column for the response"):
        read_runsheet_bytes(study, b"run,temp,kind\n1,1,x\n")
    with pytest.raises(StudyError, match="unknown value"):
        read_runsheet_bytes(study, b"run,temp,kind,y\n1,1,z,1\n")
    with pytest.raises(StudyError, match=r"row\(s\) \[2\]"):
        read_runsheet_bytes(study, b"run,temp,kind,y\n1,1,x,1\n2,,y,2\n")
    with pytest.raises(StudyError, match="leave its result blank"):
        read_runsheet_bytes(study, b"run,temp,kind,y\n1,1,x,failed\n2,2,y,2\n")


def test_blank_results_are_allowed():
    sheet = read_runsheet_bytes(_simple_study(), b"run,temp,kind,y\n1,1,x,\n2,2,y,2\n")
    assert sheet["y"].isna().tolist() == [True, False]


def test_blends_must_add_up():
    study = study_from_config(
        {"name": "t", "factors": [{"name": "a", "type": "mixture"}, {"name": "b", "type": "mixture"}], "responses": [{"name": "y"}]}
    )
    with pytest.raises(StudyError, match="add up"):
        read_runsheet_bytes(study, b"a,b,y\n0.5,0.2,1\n")


# ------------------------------------------------------------------ config errors
def test_config_typos_are_caught():
    base = {"name": "t", "factors": [{"name": "a", "low": 0, "high": 1}]}
    with pytest.raises(StudyError, match="unknown setting"):
        study_from_config({**base, "respones": []})
    with pytest.raises(StudyError, match="unknown setting"):
        study_from_config({"name": "t", "factors": [{"name": "a", "low": 0, "high": 1, "hgih": 2}]})
    with pytest.raises(StudyError, match="low must be smaller"):
        study_from_config({"name": "t", "factors": [{"name": "a", "low": 5, "high": 1}]})
    with pytest.raises(StudyError, match="more than once"):
        study_from_config({"name": "t", "factors": [{"name": "a", "low": 0, "high": 1}, {"name": "a", "low": 0, "high": 1}]})
    with pytest.raises(StudyError, match="isn't allowed"):
        study_from_config({"name": "t", "factors": [{"name": "my factor", "low": 0, "high": 1}]})
    with pytest.raises(StudyError, match="needs a target"):
        study_from_config({**base, "responses": [{"name": "y", "goal": "target"}]})
    with pytest.raises(StudyError, match="unknown factors"):
        study_from_config({**base, "constraints": [{"expr": "a + zz", "max": 1}]})


def test_every_shipped_config_is_valid_yaml_with_known_settings():
    for path in EXAMPLES + TEMPLATES:
        load_study(path)
        assert yaml.safe_load(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the advisor
def _factors(count):
    return [{"name": f"x{i + 1}", "low": 0, "high": 1} for i in range(count)]


def _advise(factors, objective="optimise", **kwargs):
    study = study_from_config({"name": "t", "seed": 1, "factors": factors, "responses": [{"name": "y"}]})
    return recommend_designs(study, objective, **kwargs)


def test_advisor_screening_scales_with_factor_count():
    assert _advise(_factors(3), "screen")[0].kind == "full_factorial"
    assert any(item.kind == "fractional_factorial" for item in _advise(_factors(6), "screen"))
    assert any(item.kind == "plackett_burman" for item in _advise(_factors(10), "screen"))


def test_advisor_optimising_chooses_response_surface_designs():
    kinds = [item.kind for item in _advise(_factors(3))]
    assert kinds[0] == "central_composite"
    assert "box_behnken" in kinds and "i_optimal" in kinds
    assert _advise(_factors(3), avoid_extremes=True)[0].kind == "box_behnken"
    assert _advise(_factors(8))[0].kind == "i_optimal"


def test_advisor_sends_mixture_and_process_studies_to_optimal_designs():
    factors = [{"name": n, "type": "mixture"} for n in "abc"] + _factors(2)
    assert _advise(factors)[0].kind == "i_optimal"


def test_advisor_on_mixtures_depends_on_whether_limits_cut_the_simplex():
    free = [{"name": n, "type": "mixture"} for n in "abc"]
    assert _advise(free)[0].kind == "simplex_centroid"
    limited = [
        {"name": "a", "type": "mixture", "low": 0.2, "high": 0.7},
        {"name": "b", "type": "mixture", "low": 0.1, "high": 0.5},
        {"name": "c", "type": "mixture", "low": 0.1, "high": 0.4},
    ]
    kinds = [item.kind for item in _advise(limited)]
    assert kinds[0] == "i_optimal" and "extreme_vertices" in kinds


def test_advisor_handles_categorical_factors_and_constraints():
    categorical = _factors(3) + [{"name": "c", "levels": ["x", "y"]}]
    assert _advise(categorical)[0].kind == "i_optimal"
    study = study_from_config(
        {"name": "t", "seed": 1, "factors": _factors(3), "constraints": [{"expr": "x1 + x2", "max": 1.5}], "responses": [{"name": "y"}]}
    )
    assert recommend_designs(study, "optimise")[0].kind == "i_optimal"


def test_advisor_respects_the_run_budget():
    results = _advise(_factors(4), max_runs=20)
    assert results[0].within_budget
    assert any(item.kind == "i_optimal" and item.runs <= 20 for item in results)
    # Designs over the budget are still shown, but last and marked.
    flags = [item.within_budget for item in results]
    assert flags == sorted(flags, reverse=True)


@pytest.mark.parametrize("objective", list(OBJECTIVES))
@pytest.mark.parametrize(
    "factors",
    [
        _factors(1),
        _factors(2),
        _factors(5),
        _factors(9),
        [{"name": n, "type": "mixture"} for n in "ab"],
        [{"name": n, "type": "mixture"} for n in "abcd"],
        _factors(2) + [{"name": "c", "levels": ["x", "y", "z"]}],
        _factors(2) + [{"name": "g", "levels": [22, 25, 27]}],
    ],
    ids=["1f", "2f", "5f", "9f", "mix2", "mix4", "categorical", "discrete"],
)
def test_advisor_always_has_a_buildable_suggestion(factors, objective):
    results = _advise(factors, objective)
    assert results, "no suggestions"
    for item in results:
        assert item.runs >= item.quality.terms
        assert item.why and item.caution


def test_advisor_refuses_unknown_objective():
    with pytest.raises(StudyError, match="objective"):
        _advise(_factors(3), "guess")


def test_advisor_reports_the_design_it_would_really_create():
    study = study_from_config({"name": "t", "seed": 1, "factors": _factors(3), "responses": [{"name": "y"}]})
    for item in recommend_designs(study, "optimise"):
        trial = replace(study, design=dict(item.design), model=item.model if item.model is not None else study.model)
        assert len(create_design(trial).frame) == item.runs
