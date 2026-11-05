"""Check that work survives uploads, refreshes and crashes, and that saves stay private."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.runtime.context import ContextProxy
from streamlit.testing.v1 import AppTest

from doekit.saved_study import (
    FILE_NAME,
    MAX_HISTORY,
    build_document,
    dump_document,
    is_saved_study,
    keep_earlier_version,
    list_saved_studies,
    new_save_path,
    read_document,
    read_table,
    save_document,
    write_atomically,
)

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app/streamlit_app.py")
WORKED_EXAMPLE = "Worked example: hydrogel blend and printing settings (simulated results)"
TEMPLATE = "Template: optimise 3 to 5 numeric factors, avoiding extreme corners"

DEFINE = "1. Define the study"
DESIGN = "2. Choose a design"
RESULTS = "3. Enter results"
ANALYSE = "4. Analyse"
OPTIMISE = "5. Find best settings"


def open_app():
    test = AppTest.from_file(APP, default_timeout=180)
    test.run()
    assert not test.exception
    return test


def no_problems(test):
    assert not test.exception
    assert not [e.value for e in test.error]


def go(test, page):
    test.session_state["page"] = page
    test.run()
    no_problems(test)


def load(test, title=WORKED_EXAMPLE):
    test.selectbox(key="example_choice").set_value(title)
    test.button(key="load_example").click()
    test.run()
    no_problems(test)


def make_design(test):
    go(test, DESIGN)
    test.radio(key="objective").set_value("optimise")
    test.run()
    test.button(key="suggest").click()
    test.run()
    test.button(key="use_0").click()
    test.run()
    no_problems(test)


def save_files():
    folder = Path(os.environ["DOEKIT_SAVE_DIR"])
    return sorted(folder.glob(f"*/{FILE_NAME}")) if folder.exists() else []


def filled_csv(test, separator=",", decimal="."):
    """Fill in the run sheet as someone would in Excel. Notes stay empty."""
    sheet = test.session_state["sheet"].copy()
    generator = np.random.default_rng(0)
    for name, centre, spread in [("printability", 1, 0.1), ("filament_width", 400, 40), ("modulus", 15, 3), ("viability", 85, 4)]:
        sheet[name] = np.round(generator.normal(centre, spread, len(sheet)), 3)
    return sheet, sheet.to_csv(index=False, sep=separator, decimal=decimal).encode("utf-8")


def upload_sheet(test, raw, name="results.csv"):
    go(test, RESULTS)
    test.file_uploader(key="sheet_upload").set_value((name, raw, "text/csv"))
    test.run()


# ------------------------------------------------------------------ the reported crash
@pytest.mark.parametrize("separator, decimal", [(",", "."), (";", ",")], ids=["plain", "european"])
def test_uploading_a_filled_in_run_sheet_with_empty_notes_works(separator, decimal):
    """Regression: an empty notes column read as numbers, which crashed the results table."""
    test = open_app()
    load(test)
    make_design(test)
    expected, raw = filled_csv(test, separator, decimal)
    upload_sheet(test, raw)
    no_problems(test)
    sheet = test.session_state["sheet"]
    assert sheet["viability"].tolist() == expected["viability"].tolist()
    assert (sheet["notes"] == "").all()
    assert not sheet["notes"].isna().any()
    # The later steps have to work with the uploaded results too.
    go(test, ANALYSE)
    assert any(m.label == "Predicted R²" for m in test.metric)
    go(test, OPTIMISE)
    test.button(key="find_best").click()
    test.run()
    no_problems(test)
    assert test.session_state["solutions"] is not None


def test_notes_in_an_uploaded_sheet_are_kept():
    test = open_app()
    load(test)
    make_design(test)
    sheet, _ = filled_csv(test)
    sheet["notes"] = ["nozzle clogged", 42] + [""] * (len(sheet) - 2)
    upload_sheet(test, sheet.to_csv(index=False).encode("utf-8"))
    no_problems(test)
    notes = test.session_state["sheet"]["notes"].tolist()
    assert notes[0] == "nozzle clogged" and notes[1] == "42" and notes[2] == ""


def test_a_bad_upload_explains_itself_and_keeps_the_existing_sheet():
    test = open_app()
    load(test)
    make_design(test)
    before = test.session_state["sheet"].copy()
    upload_sheet(test, b"wrong,columns\n1,2\n")
    assert not test.exception
    assert any("no column for" in e.value for e in test.error)
    pd.testing.assert_frame_equal(test.session_state["sheet"], before)


def test_an_upload_without_a_notes_column_works():
    test = open_app()
    load(test)
    make_design(test)
    sheet, _ = filled_csv(test)
    upload_sheet(test, sheet.drop(columns=["notes"]).to_csv(index=False).encode("utf-8"))
    no_problems(test)
    assert "notes" in test.session_state["sheet"]


# ------------------------------------------------------------------ carrying on after a refresh
def test_refreshing_the_page_loses_nothing():
    """A refresh starts a new session, so everything has to come back from the save."""
    test = open_app()
    load(test)
    test.session_state["study_seed"] = 77
    test.run()
    make_design(test)
    test.session_state["max_runs"] = 40
    test.session_state["avoid_extremes"] = True
    test.run()
    go(test, RESULTS)
    test.button(key="simulate").click()
    test.run()
    go(test, OPTIMISE)
    test.button(key="find_best").click()
    test.run()
    no_problems(test)
    go(test, DEFINE)

    state = test.session_state
    sheet = state["sheet"].copy()
    design_settings, model_settings = dict(state["design_settings"]), dict(state["model_settings"])
    kind, notes, signature = state["design_kind"], list(state["design_notes"]), state["design_signature"]
    solutions = state["solutions"].copy()
    factors = state["factors_current"].copy()
    responses = state["responses_current"].copy()

    refreshed = open_app()
    assert refreshed.session_state["sheet"] is None
    assert any("hydrogel_bioink_simulated" in str(option) for option in refreshed.selectbox(key="saved_choice").options)
    refreshed.button(key="resume").click()
    refreshed.run()
    no_problems(refreshed)

    state = refreshed.session_state
    assert state["page"] == RESULTS
    assert any("Carrying on" in s.value for s in refreshed.success)
    assert state["study_name"] == "hydrogel_bioink_simulated"
    assert state["study_seed"] == 77
    assert state["mixture_total"] == 1.0
    assert state["objective"] == "optimise"
    assert state["max_runs"] == 40
    assert state["avoid_extremes"] is True
    assert state["design_settings"] == design_settings
    assert state["model_settings"] == model_settings
    assert state["design_kind"] == kind and state["design_notes"] == notes and state["design_signature"] == signature
    pd.testing.assert_frame_equal(state["factors_current"], factors, check_dtype=False)
    pd.testing.assert_frame_equal(state["responses_current"], responses, check_dtype=False)
    assert state["sheet"].to_csv(index=False) == sheet.to_csv(index=False)

    # The best settings come back without having to search again.
    go(refreshed, OPTIMISE)
    assert any(m.label == "Overall desirability" for m in refreshed.metric)
    assert refreshed.session_state["solutions"].iloc[0]["desirability"] == pytest.approx(solutions.iloc[0]["desirability"])

    # It carries on in the same save, rather than starting another.
    assert len(save_files()) == 1


def test_a_half_finished_study_is_saved_and_comes_back():
    """Tables are saved as they're filled in, even before they make a valid study."""
    test = open_app()
    half_done = pd.DataFrame([{"name": "temp", "type": "continuous", "low": 40.0, "high": np.nan, "levels": "", "units": "C"}])
    test.session_state["factors_current"] = half_done
    test.session_state["factors_base"] = half_done
    test.run()
    assert not test.exception
    assert len(save_files()) == 1

    refreshed = open_app()
    refreshed.button(key="resume").click()
    refreshed.run()
    assert not refreshed.exception
    factors = refreshed.session_state["factors_current"]
    assert factors.loc[0, "name"] == "temp" and factors.loc[0, "low"] == 40.0 and np.isnan(factors.loc[0, "high"])


def test_a_crash_on_a_page_does_not_lose_the_latest_changes(monkeypatch):
    """The study is saved even when a page fails part-way through."""
    test = open_app()
    load(test)
    make_design(test)
    test.session_state["study_seed"] = 99

    def crash(*_args, **_kwargs):
        raise RuntimeError("something unexpected")

    with monkeypatch.context() as patch:
        patch.setattr("doekit.evaluate.evaluate_design", crash)
        test.session_state["page"] = DESIGN
        test.run()
    assert not test.exception
    assert any("your study has been kept" in e.value for e in test.error)
    assert read_document(save_files()[0])["config"]["seed"] == 99

    # The sidebar still works, so the person isn't stuck.
    go(test, RESULTS)
    assert test.session_state["sheet"] is not None


# ------------------------------------------------------------------ not losing or mixing up studies
def test_just_opening_an_example_saves_nothing():
    test = open_app()
    load(test)
    go(test, DESIGN)
    go(test, DEFINE)
    assert save_files() == []


def test_two_studies_with_the_same_name_keep_separate_saves():
    first = open_app()
    load(first, TEMPLATE)
    make_design(first)
    second = open_app()
    load(second, TEMPLATE)
    make_design(second)
    assert first.session_state["study_name"] == second.session_state["study_name"]
    assert len(save_files()) == 2


def test_replacing_results_keeps_the_earlier_version():
    test = open_app()
    load(test)
    make_design(test)
    go(test, RESULTS)
    test.button(key="simulate").click()
    test.run()
    earlier = test.session_state["sheet"]["viability"].tolist()
    go(test, DESIGN)  # the save now holds the simulated results
    test.checkbox(key="confirm_replace").check()
    test.run()
    test.button(key="suggest").click()
    test.run()
    test.button(key="use_1").click()
    test.run()
    no_problems(test)
    assert test.session_state["sheet"]["viability"].isna().all()

    history = list((save_files()[0].parent / "history").glob("*.yaml"))
    assert len(history) == 1
    kept = read_table(read_document(history[0])["runsheet_csv"])
    assert kept["viability"].tolist() == pytest.approx(earlier)


# ------------------------------------------------------------------ privacy on a shared server
def test_autosave_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("DOEKIT_AUTOSAVE", "off")
    test = open_app()
    load(test)
    make_design(test)
    assert save_files() == []
    assert not [element for element in test.selectbox if element.key == "saved_choice"]
    assert any("isn't saved on this server" in caption.value for caption in test.sidebar.caption)


def test_visitors_from_other_computers_are_not_saved_and_see_no_saved_studies(monkeypatch):
    owner = open_app()
    load(owner)
    make_design(owner)
    assert len(save_files()) == 1

    monkeypatch.setattr(ContextProxy, "ip_address", property(lambda self: "10.1.2.3"))
    visitor = open_app()
    assert not [element for element in visitor.selectbox if element.key == "saved_choice"]
    load(visitor, TEMPLATE)
    make_design(visitor)
    assert len(save_files()) == 1
    assert any("isn't saved on this server" in caption.value for caption in visitor.sidebar.caption)


def test_a_downloaded_study_file_can_be_loaded_on_another_computer(monkeypatch):
    """This is how people on a hosted copy keep their work."""
    test = open_app()
    load(test)
    make_design(test)
    go(test, RESULTS)
    test.button(key="simulate").click()
    test.run()
    go(test, DEFINE)
    raw = save_files()[0].read_bytes()
    original = test.session_state["sheet"].to_csv(index=False)

    monkeypatch.setenv("DOEKIT_AUTOSAVE", "off")
    other = open_app()
    other.file_uploader(key="config_upload").set_value(("my_study.doekit.yaml", raw, "text/yaml"))
    other.run()
    other.button(key="load_upload").click()
    other.run()
    no_problems(other)
    assert other.session_state["study_name"] == "hydrogel_bioink_simulated"
    assert other.session_state["sheet"].to_csv(index=False) == original
    assert other.session_state["design_kind"] == "i_optimal"


def test_an_uploaded_study_file_cannot_read_files_from_the_server():
    test = open_app()
    document = build_document(
        {
            "name": "attack",
            "data_file": "archive/private/mew_doe_results.csv",
            "factors": [{"name": "a", "low": 0.0, "high": 1.0}],
            "responses": [],
        },
        None,
        None,
        {},
    )
    test.file_uploader(key="config_upload").set_value(("x.yaml", dump_document(document), "text/yaml"))
    test.run()
    test.button(key="load_upload").click()
    test.run()
    no_problems(test)
    assert "data_file" not in test.session_state["factors_current"].columns
    assert test.session_state["study_name"] == "attack"
    saved = read_document(save_files()[0])
    assert "data_file" not in saved["config"]


@pytest.mark.parametrize(
    "raw",
    [
        b"doekit_saved_study: 1\nconfig: not a dictionary\n",
        b"doekit_saved_study: 1\nconfig: {factors: [1, 2]}\n",
        b"doekit_saved_study: 1\nconfig: {name: x}\nrunsheet_csv: 5\n",
        b": : : not yaml [",
        "��".encode("utf-16"),
    ],
    ids=["config-text", "bad-factors", "bad-sheet", "not-yaml", "not-text"],
)
def test_a_damaged_study_file_explains_itself_and_changes_nothing(raw):
    test = open_app()
    load(test)
    before = test.session_state["factors_current"].copy()
    test.file_uploader(key="config_upload").set_value(("damaged.yaml", raw, "text/yaml"))
    test.run()
    test.button(key="load_upload").click()
    test.run()
    assert not test.exception
    assert any("Couldn't load that file" in e.value for e in test.error)
    pd.testing.assert_frame_equal(test.session_state["factors_current"], before)


def test_a_failure_to_save_warns_but_does_not_stop_the_app(monkeypatch, tmp_path):
    blocked = tmp_path / "not_a_folder"
    blocked.write_text("a file is in the way")
    monkeypatch.setenv("DOEKIT_SAVE_DIR", str(blocked))
    test = open_app()
    load(test)
    make_design(test)
    assert not test.exception
    assert any("Couldn't save automatically" in w.value for w in test.sidebar.warning)
    assert test.session_state["sheet"] is not None


# ------------------------------------------------------------------ the save file itself
def _document(sheet=None, name="study"):
    return build_document({"name": name, "factors": [{"name": "a", "low": 0.0, "high": 1.0}], "responses": [{"name": "y"}]}, sheet, None, {})


def test_numbers_come_back_exactly():
    sheet = pd.DataFrame({"a": [0.1 + 0.2, 1 / 3, 1e-17], "y": [np.pi, np.nan, 2.0], "notes": ["", "x, with a comma", 'and "quotes"']})
    back = read_table(yaml.safe_load(dump_document(_document(sheet)))["runsheet_csv"])
    assert back["a"].tolist() == sheet["a"].tolist()
    assert back["y"].iloc[0] == sheet["y"].iloc[0] and np.isnan(back["y"].iloc[1])
    assert back["notes"].iloc[1] == "x, with a comma" and back["notes"].iloc[2] == 'and "quotes"'


def test_a_config_is_not_mistaken_for_a_saved_study():
    assert is_saved_study(_document())
    assert not is_saved_study({"name": "x", "factors": []})
    assert not is_saved_study("text")
    assert not is_saved_study(None)


def test_a_failed_write_leaves_the_previous_save_intact(tmp_path, monkeypatch):
    path = tmp_path / "study" / FILE_NAME
    write_atomically(path, b"the good version")

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        write_atomically(path, b"a broken version")
    assert path.read_bytes() == b"the good version"
    assert [item.name for item in path.parent.iterdir()] == [FILE_NAME]


def test_save_locations_are_unique_and_stay_inside_the_folder(tmp_path):
    first, second = new_save_path(tmp_path, "same"), new_save_path(tmp_path, "same")
    assert first != second
    for name in ("../../../etc/passwd", "/absolute", "a b/c", "", "..", "x" * 300):
        path = new_save_path(tmp_path, name).resolve()
        assert path.is_relative_to(tmp_path.resolve())
        assert path.parent.parent == tmp_path.resolve()
        assert len(path.parent.name) < 60


def test_listing_shows_newest_first_with_progress_and_skips_damaged_files(tmp_path):
    older = tmp_path / "older" / FILE_NAME
    sheet = pd.DataFrame({"a": [0.0, 0.5, 1.0], "y": [1.0, np.nan, 3.0]})
    save_document(older, _document(sheet, "older_study"))
    os.utime(older, (1_000_000_000, 1_000_000_000))
    save_document(tmp_path / "newer" / FILE_NAME, _document(None, "newer_study"))
    (tmp_path / "damaged").mkdir()
    (tmp_path / "damaged" / FILE_NAME).write_text(": : [not yaml")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / FILE_NAME).write_text("name: just a config\n")

    studies = list_saved_studies(tmp_path)
    assert [study.name for study in studies] == ["newer_study", "older_study"]
    assert "no run sheet yet" in studies[0].label
    assert "2 of 3 runs complete" in studies[1].label
    assert list_saved_studies(tmp_path / "missing") == []


def test_history_keeps_only_the_most_recent_versions(tmp_path):
    path = tmp_path / "study" / FILE_NAME
    assert keep_earlier_version(path) is None
    for number in range(MAX_HISTORY + 5):
        write_atomically(path, f"version {number}".encode())
        keep_earlier_version(path)
    kept = sorted((path.parent / "history").glob("*.yaml"))
    assert len(kept) == MAX_HISTORY
    assert kept[-1].read_text() == f"version {MAX_HISTORY + 4}"
    assert kept[0].read_text() == "version 5"
