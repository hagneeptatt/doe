"""Click through the web app without a browser, using Streamlit's own test tool."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from doekit.designs import create_design
from doekit.study import study_from_config, without_server_paths

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app/streamlit_app.py")
WORKED_EXAMPLE = "Worked example: hydrogel blend and printing settings (simulated results)"

DEFINE = "1. Define the study"
DESIGN = "2. Choose a design"
RESULTS = "3. Enter results"
ANALYSE = "4. Analyse"
OPTIMISE = "5. Find best settings"
MANUAL = "Manual"


@pytest.fixture
def app():
    test = AppTest.from_file(APP, default_timeout=180)
    test.run()
    assert not test.exception
    return test


def no_problems(test):
    assert not test.exception
    unexpected = [e.value for e in test.error]
    assert not unexpected, unexpected


def go(test, page):
    test.session_state["page"] = page
    test.run()
    no_problems(test)


def load_example(test, title=WORKED_EXAMPLE):
    test.selectbox(key="example_choice").set_value(title)
    test.button(key="load_example").click()
    test.run()
    no_problems(test)
    assert any("Study is ready" in s.value for s in test.success)


def make_design(test, objective="optimise"):
    go(test, DESIGN)
    assert test.button(key="suggest").disabled  # the objective has to be a deliberate choice
    test.radio(key="objective").set_value(objective)
    test.run()
    test.button(key="suggest").click()
    test.run()
    no_problems(test)
    test.button(key="use_0").click()
    test.run()
    no_problems(test)
    assert test.session_state["sheet"] is not None


def simulate(test):
    go(test, RESULTS)
    test.button(key="simulate").click()
    test.run()
    no_problems(test)


def test_app_opens_on_the_first_step_with_guidance(app):
    assert app.header[0].value == DEFINE
    assert any("Add at least one factor" in i.value for i in app.info)


def test_later_steps_explain_what_to_do_first(app):
    for page in (DESIGN, RESULTS, ANALYSE, OPTIMISE):
        go(app, page)
        assert any("Finish defining the study" in i.value for i in app.info), page


def test_full_journey_through_the_worked_example(app):
    load_example(app)
    make_design(app)
    assert any(m.label == "Runs" for m in app.metric)

    simulate(app)
    sheet = app.session_state["sheet"]
    assert sheet["viability"].notna().all()
    assert any("simulated" in w.value.lower() for w in app.warning)

    go(app, ANALYSE)
    assert any(m.label == "Predicted R²" for m in app.metric)
    assert app.success or app.warning  # the verdict is shown

    views = ["What matters", "Effect of each factor", "Interactions", "Surfaces", "Model checks", "Tables"]
    for view in views:
        app.radio(key="analysis_view").set_value(view)
        app.run()
        no_problems(app)

    go(app, OPTIMISE)
    app.button(key="find_best").click()
    app.run()
    no_problems(app)
    assert any(m.label == "Overall desirability" for m in app.metric)
    solutions = app.session_state["solutions"]
    assert solutions.iloc[0]["desirability"] > 0.5
    assert solutions.iloc[0][["gelma", "alginate", "nanocellulose"]].sum() == pytest.approx(1)

    app.button(key="prepare_zip").click()
    app.run()
    no_problems(app)
    assert app.session_state["results_zip"]


def test_study_settings_survive_moving_between_steps(app):
    """Regression: these used to reset when their page went off screen, silently changing the study."""
    load_example(app)
    assert app.session_state["study_seed"] == 42
    app.session_state["mixture_total"] = 1.2
    for page in (DESIGN, RESULTS, ANALYSE, OPTIMISE, MANUAL, DEFINE):
        go(app, page)
        assert app.session_state["study_name"] == "hydrogel_bioink_simulated", page
        assert app.session_state["study_seed"] == 42, page
        assert app.session_state["mixture_total"] == 1.2, page


def test_the_app_gives_exactly_the_same_design_as_the_scripts(app):
    load_example(app)
    make_design(app)
    config = yaml.safe_load((ROOT / "configs/examples/hydrogel_bioink_simulated.yaml").read_text(encoding="utf-8"))
    config["design"] = app.session_state["design_settings"]
    study = study_from_config(config)
    expected = create_design(study).frame[study.factor_names]
    sheet = app.session_state["sheet"][study.factor_names]
    assert sheet.to_numpy().tolist() == expected.to_numpy().tolist()


def test_all_examples_and_templates_load(app):
    titles = list(app.selectbox(key="example_choice").options)
    assert len(titles) >= 10
    for title in titles:
        load_example(app, title)


def test_the_data_in_the_archive_is_not_offered(app):
    options = " ".join(app.selectbox(key="example_choice").options).lower()
    assert "mew" not in options and "jetting" not in options


def test_results_are_not_replaced_without_confirmation(app):
    load_example(app)
    make_design(app)
    simulate(app)
    go(app, DESIGN)
    assert any("replaces them" in w.value for w in app.warning)
    app.radio(key="objective").set_value("optimise")
    app.run()
    app.button(key="suggest").click()
    app.run()
    assert app.button(key="use_0").disabled
    assert app.button(key="create_manual").disabled

    go(app, DEFINE)
    assert app.button(key="load_example").disabled


def test_a_real_study_cannot_be_given_simulated_results(app):
    load_example(app, "Template: optimise 3 to 5 numeric factors, avoiding extreme corners")
    make_design(app)
    go(app, RESULTS)
    assert not [b for b in app.button if b.key == "simulate"]


def test_uploaded_config_cannot_read_files_from_the_server():
    """An uploaded config's data_file setting could otherwise expose files on a hosted server."""
    config = {
        "name": "attack",
        "data_file": "archive/original_analysis/mew_doe_results.csv",
        "factors": [{"name": "pres", "low": 0.1, "high": 0.5}],
    }
    cleaned = without_server_paths(config)
    assert "data_file" not in cleaned
    assert study_from_config(cleaned, ROOT).data_file is None
    assert "data_file" in config  # the original is left untouched
    # The app has to use this on everything it loads.
    assert "config = without_server_paths(config)" in Path(APP).read_text(encoding="utf-8")


def test_bad_factor_settings_show_a_clear_message(app):
    app.session_state["factors_current"] = pd.DataFrame(
        [{"name": "a", "type": "continuous", "low": 5.0, "high": 1.0, "levels": "", "units": ""}]
    )
    app.session_state["factors_base"] = app.session_state["factors_current"]
    app.session_state["version"] = 100
    app.run()
    assert any("low must be smaller" in e.value for e in app.error)
    assert not app.exception


def test_partial_results_can_be_analysed_and_are_reported(app):
    load_example(app)
    make_design(app)
    simulate(app)
    sheet = app.session_state["sheet"].copy()
    sheet.loc[sheet.index[:2], "viability"] = np.nan
    app.session_state["sheet"] = sheet
    go(app, ANALYSE)
    app.selectbox(key="analysis_response").set_value("viability")
    app.run()
    no_problems(app)
    assert any("left out" in i.value for i in app.info)


def test_changing_factors_after_the_design_warns(app):
    load_example(app)
    make_design(app)
    factors = app.session_state["factors_current"].copy()
    factors.loc[factors["name"] == "pressure", "high"] = 150.0
    app.session_state["factors_current"] = factors
    go(app, RESULTS)
    assert any("factors have changed" in w.value for w in app.warning)


def test_every_link_within_the_manual_leads_somewhere():
    text = (ROOT / "docs/manual.md").read_text(encoding="utf-8")
    headings = re.findall(r"^#{1,6} (.+)$", text, flags=re.MULTILINE)
    anchors = {re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-") for heading in headings}
    links = re.findall(r"\]\(#([^)]+)\)", text)
    assert len(links) > 15
    assert not [link for link in links if link not in anchors]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"docs/manual\.md#([^)]+)\)", readme):
        assert link in anchors, link


def test_every_method_table_source_is_in_the_references():
    """Each short citation in the methods section has to have a full entry."""
    text = (ROOT / "docs/manual.md").read_text(encoding="utf-8")
    methods = text[text.index("## 11. Methods behind each step") : text.index("## 12. How doekit was checked")]
    references = text[text.index("## 13. References") :]
    citations = re.findall(r"([A-Z][\w'é]+)(?: & [A-Z][\w'é]+| et al\.)? \((\d{4})\)", methods)
    assert len(citations) > 15
    for author, year in citations:
        assert re.search(rf"^- {author},.*\({year}\)", references, flags=re.MULTILINE), f"{author} ({year})"


def test_manual_page_renders(app):
    go(app, MANUAL)
    text = " ".join(m.value for m in app.markdown)
    assert "Choosing a design" in text
