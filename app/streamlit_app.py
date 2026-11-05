"""A point-and-click interface for doekit.

Start it with:   streamlit run app/streamlit_app.py

Every calculation is done by the doekit package, the same code the scripts use. This
file only handles the screens. Nothing is saved on the server: a study lives in the
browser session until it's downloaded, so several people can use one copy of the app.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from doekit import plots, report
from doekit.advisor import OBJECTIVES, recommend_designs
from doekit.analysis import (
    ResponseFit,
    anova_table,
    coefficient_table,
    fit_metrics,
    fit_response,
    interpret_fit,
    lenth_analysis,
    run_diagnostics,
)
from doekit.designs import DESIGNS, Design, create_design
from doekit.evaluate import evaluate_design
from doekit.model import CROSSING_ORDERS, MIXTURE_ORDERS, PROCESS_ORDERS, model_terms
from doekit.optimise import desirability_limits, find_optimum
from doekit.runsheet import build_runsheet, clean_runsheet, read_runsheet_bytes
from doekit.simulate import SIMULATED_NOTE, simulate_responses
from doekit.study import FACTOR_KINDS, GOALS, TRANSFORMS, Study, StudyError, study_from_config, without_server_paths
from doekit.workflow import save_fit_outputs, save_optimum_outputs

ROOT = Path(__file__).resolve().parents[1]
ss = st.session_state

FACTOR_COLUMNS = ["name", "type", "low", "high", "levels", "units"]
RESPONSE_COLUMNS = ["name", "units", "goal", "low", "high", "target", "importance", "transform"]
CONSTRAINT_COLUMNS = ["expr", "min", "max"]

PAGE_DEFINE = "1. Define the study"
PAGE_DESIGN = "2. Choose a design"
PAGE_RESULTS = "3. Enter results"
PAGE_ANALYSE = "4. Analyse"
PAGE_OPTIMISE = "5. Find best settings"
PAGE_MANUAL = "Manual"
PAGES = [PAGE_DEFINE, PAGE_DESIGN, PAGE_RESULTS, PAGE_ANALYSE, PAGE_OPTIMISE, PAGE_MANUAL]

# Only simulated examples and blank templates are offered, so a hosted copy can't show
# real data to its visitors.
EXAMPLE_TITLES = {
    "examples/hydrogel_bioink_simulated": "Worked example: hydrogel blend and printing settings (simulated results)",
    "examples/hydrogel_concentrations_simulated": "Worked example: hydrogel concentrations and crosslinker (simulated results)",
    "templates/mixture_process_i_optimal": "Template: optimise a blend and its process settings",
    "templates/optimise_i_optimal_mixed_factors": "Template: optimise with categorical factors, constraints or a fixed budget",
    "templates/optimise_box_behnken": "Template: optimise 3 to 5 numeric factors, avoiding extreme corners",
    "templates/optimise_central_composite": "Template: optimise 2 to 5 numeric factors",
    "templates/mixture_simplex_centroid": "Template: optimise a blend only",
    "templates/screening_fractional_factorial": "Template: screen 4 to 11 factors",
    "templates/screening_plackett_burman": "Template: screen many factors with very few runs",
    "templates/explore_latin_hypercube": "Template: explore unknown behaviour, or collect machine learning data",
    "templates/existing_data": "Template: analyse an experiment that's already been done",
}


# ---------------------------------------------------------------------- tables and config
def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return ""
    return str(value).strip()


def _number(value) -> float | None:
    if value is None or value is pd.NA or value == "":
        return None
    number = float(value)
    return None if np.isnan(number) else number


def _table(rows: list[dict], columns: list[str], numeric: tuple[str, ...]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column in numeric:
            table[column] = pd.to_numeric(table[column], errors="coerce").astype(float)
        else:
            table[column] = table[column].map(_text).astype(object)
    return table


def factors_to_table(factors: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in factors:
        levels = list(entry.get("levels") or [])
        kind = entry.get("type")
        if kind is None:
            numeric = all(isinstance(level, (int, float)) and not isinstance(level, bool) for level in levels)
            kind = "continuous" if not levels else "discrete" if numeric else "categorical"
        rows.append({**entry, "type": kind, "levels": ", ".join(str(level) for level in levels)})
    return _table(rows, FACTOR_COLUMNS, ("low", "high"))


def table_to_factors(table: pd.DataFrame) -> list[dict]:
    factors = []
    for row in table.to_dict("records"):
        name = _text(row["name"])
        if not name:
            continue
        kind = _text(row["type"]) or "continuous"
        entry: dict = {"name": name, "type": kind}
        if _text(row["units"]):
            entry["units"] = _text(row["units"])
        if kind in ("categorical", "discrete"):
            entry["levels"] = [piece.strip() for piece in _text(row["levels"]).split(",") if piece.strip()]
        else:
            for key in ("low", "high"):
                if _number(row[key]) is not None:
                    entry[key] = _number(row[key])
        factors.append(entry)
    return factors


def responses_to_table(responses: list[dict]) -> pd.DataFrame:
    rows = [{"goal": "none", "importance": 1.0, "transform": "none", **entry} for entry in responses]
    return _table(rows, RESPONSE_COLUMNS, ("low", "high", "target", "importance"))


def table_to_responses(table: pd.DataFrame) -> list[dict]:
    responses = []
    for row in table.to_dict("records"):
        name = _text(row["name"])
        if not name:
            continue
        entry: dict = {"name": name, "goal": _text(row["goal"]) or "none", "transform": _text(row["transform"]) or "none"}
        if _text(row["units"]):
            entry["units"] = _text(row["units"])
        for key in ("low", "high", "target", "importance"):
            if _number(row[key]) is not None:
                entry[key] = _number(row[key])
        responses.append(entry)
    return responses


def table_to_constraints(table: pd.DataFrame) -> list[dict]:
    constraints = []
    for row in table.to_dict("records"):
        expression = _text(row["expr"])
        if not expression:
            continue
        entry: dict = {"expr": expression}
        for key in ("min", "max"):
            if _number(row[key]) is not None:
                entry[key] = _number(row[key])
        constraints.append(entry)
    return constraints


def current_config() -> dict:
    config: dict = {
        "name": _text(ss.study_name) or "my_study",
        "description": ss.study_description,
        "seed": int(ss.study_seed),
    }
    factors = table_to_factors(ss.factors_current)
    if any(entry["type"] == "mixture" for entry in factors):
        config["mixture_total"] = float(ss.mixture_total)
    config["factors"] = factors
    constraints = table_to_constraints(ss.constraints_current)
    if constraints:
        config["constraints"] = constraints
    config["responses"] = table_to_responses(ss.responses_current)
    if ss.design_settings:
        config["design"] = dict(ss.design_settings)
    if ss.model_settings:
        config["model"] = dict(ss.model_settings)
    if ss.simulation:
        config["simulation"] = dict(ss.simulation)
    return config


def current_study() -> tuple[Study | None, str | None]:
    try:
        return study_from_config(current_config(), ROOT, "This study"), None
    except (StudyError, ValueError, TypeError) as error:
        return None, str(error)


def _hash(value) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _sheet_hash() -> str:
    if ss.sheet is None:
        return "none"
    return _hash(ss.sheet.astype(str).to_dict("list"))


def _factor_signature(config: dict) -> str:
    return _hash([config.get("factors"), config.get("mixture_total"), config.get("constraints")])


# ---------------------------------------------------------------------- session state
# Streamlit forgets a widget's value once its page is off screen. For these, that would
# silently change the study (its seed or mixture total, for example) on moving between steps.
PERSISTENT_WIDGETS = [
    "study_name",
    "study_description",
    "study_seed",
    "mixture_total",
    "objective",
    "max_runs",
    "signal_to_noise",
    "avoid_extremes",
    "example_choice",
    "design_view",
    "analysis_response",
    "analysis_view",
    "manual_kind",
]


def init_state() -> None:
    for key in PERSISTENT_WIDGETS:
        if key in ss:
            ss[key] = ss[key]  # assigning it marks it as ours, so Streamlit keeps it
    defaults = {
        "study_name": "my_study",
        "study_description": "",
        "study_seed": 1,
        "mixture_total": 1.0,
        "signal_to_noise": 2.0,
        "design_settings": {},
        "model_settings": {},
        "simulation": {},
        "sheet": None,
        "sheet_base": None,
        "design_notes": [],
        "design_signature": None,
        "design_kind": None,
        "recommendations": None,
        "solutions": None,
        "solutions_key": None,
        "fit_cache": {},
        "figure_cache": {},
        "results_zip": None,
        "version": 0,
        "last_page": None,
        "config_loaded": False,
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value
    if "factors_current" not in ss:
        ss.factors_current = factors_to_table([])
        ss.responses_current = responses_to_table([])
        ss.constraints_current = _table([], CONSTRAINT_COLUMNS, ("min", "max"))
    if "factors_base" not in ss:
        refresh_editors()


def refresh_editors() -> None:
    """Point the table editors at the latest data.

    An editor's starting data has to stay the same while it's on screen, or edits get
    lost. So it only changes on moving between pages, or when new data is loaded.
    """
    ss.factors_base = ss.factors_current.copy()
    ss.responses_base = ss.responses_current.copy()
    ss.constraints_base = ss.constraints_current.copy()
    ss.sheet_base = None if ss.sheet is None else ss.sheet.copy()
    ss.version += 1


def has_results() -> bool:
    if ss.sheet is None:
        return False
    names = [name for name in table_to_response_names() if name in ss.sheet]
    return bool(names) and bool(ss.sheet[names].notna().any().any())


def table_to_response_names() -> list[str]:
    return [entry["name"] for entry in table_to_responses(ss.responses_current)]


def set_sheet(sheet: pd.DataFrame | None) -> None:
    ss.sheet = sheet
    ss.solutions = None
    ss.results_zip = None
    refresh_editors()


def load_config(config: dict, source: str) -> None:
    config = without_server_paths(config)
    study_from_config(config, ROOT, source)  # check it first, so a bad file changes nothing
    ss.study_name = config["name"]
    ss.study_description = str(config.get("description", "")).strip()
    ss.study_seed = int(config.get("seed") or 1)
    ss.mixture_total = float(config.get("mixture_total", 1.0))
    ss.factors_current = factors_to_table(config["factors"])
    ss.responses_current = responses_to_table(config.get("responses") or [])
    ss.constraints_current = _table(config.get("constraints") or [], CONSTRAINT_COLUMNS, ("min", "max"))
    design = dict(config.get("design") or {})
    ss.design_settings = design
    model = config.get("model") or {}
    ss.model_settings = {"process": model} if isinstance(model, str) else dict(model)
    ss.simulation = dict(config.get("simulation") or {})
    ss.pop("manual_kind", None)  # start the design dropdown from this study's own design
    ss.recommendations = None
    ss.design_notes = []
    ss.design_signature = None
    ss.design_kind = None
    ss.config_loaded = True
    ss.fit_cache = {}
    ss.figure_cache = {}
    set_sheet(None)


def go_to(page: str) -> None:
    ss.page = page


def next_button(page: str, label: str | None = None) -> None:
    st.divider()
    st.button(label or f"Next: {page} →", key=f"next_{page}", type="primary", on_click=go_to, args=(page,))


def show_figure(key: str, build, empty_message: str = "Nothing to show for this study.") -> None:
    """Draw a figure, reusing it if it's already been drawn this session."""
    cache = ss.figure_cache
    if key not in cache:
        figure = build()
        if figure is None:
            st.info(empty_message)
            return
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        cache[key] = (buffer.getvalue(), int(figure.get_size_inches()[0] * 100))
        if len(cache) > 80:
            cache.pop(next(iter(cache)))
    image, width = cache[key]
    st.image(image, width=width)


def show_table(frame: pd.DataFrame, digits: int = 4) -> None:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda v: float(f"{v:.{digits}g}") if np.isfinite(v) else np.nan)
    st.dataframe(shown, hide_index=True, width="stretch")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def yaml_bytes(config: dict) -> bytes:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8")


def simulated_warning() -> None:
    if ss.sheet is not None and "notes" in ss.sheet and ss.sheet["notes"].astype(str).str.contains("SIMULATED").any():
        st.warning("**These results are simulated.** They're for practice and are not real measurements.")


def require_study() -> Study | None:
    study, error = current_study()
    if study is None:
        st.info("Finish defining the study first.")
        if error and table_to_factors(ss.factors_current):
            st.error(error)
        st.button(f"Go to {PAGE_DEFINE}", key="go_define", on_click=go_to, args=(PAGE_DEFINE,))
    return study


def design_changed_warning(study: Study) -> None:
    if ss.sheet is not None and ss.design_signature and ss.design_signature != _factor_signature(current_config()):
        st.warning(
            "The factors have changed since this run sheet was made, so the two may no longer match. "
            "Create a new design, or undo the change."
        )


# ---------------------------------------------------------------------- page 1: define
def example_configs() -> dict[str, Path]:
    return {title: ROOT / "configs" / f"{stem}.yaml" for stem, title in EXAMPLE_TITLES.items() if (ROOT / "configs" / f"{stem}.yaml").exists()}


def page_define() -> None:
    st.header(PAGE_DEFINE)
    st.write(
        "List what you'll change (**factors**) and what you'll measure (**responses**). "
        "New to DOE? Load the first worked example below to see a complete study."
    )

    with st.expander("Start from an example, a template or a saved study", expanded=not ss.config_loaded):
        confirm = True
        if has_results():
            confirm = st.checkbox("Replace my current study and its results", key="confirm_load")
        left, right = st.columns(2)
        with left:
            examples = example_configs()
            choice = st.selectbox("Example or template", list(examples), key="example_choice")
            if st.button("Load example", key="load_example", disabled=not confirm):
                load_config(yaml.safe_load(examples[choice].read_text(encoding="utf-8")), "This example")
                st.rerun()
        with right:
            uploaded = st.file_uploader("Saved study (config file)", type=["yaml", "yml"], key="config_upload")
            if st.button("Load saved study", key="load_upload", disabled=(uploaded is None or not confirm)):
                try:
                    load_config(yaml.safe_load(uploaded.getvalue().decode("utf-8")), "This file")
                    st.rerun()
                except (StudyError, yaml.YAMLError, UnicodeDecodeError, AttributeError, TypeError) as error:
                    st.error(f"Couldn't load that file. {error}")

    left, right = st.columns([3, 1])
    with left:
        st.text_input("Study name", key="study_name", help="Letters, numbers and underscores, starting with a letter.")
    with right:
        st.number_input(
            "Random seed", key="study_seed", step=1, min_value=0,
            help="The same seed recreates exactly the same design, and is kept in the config file.",
        )  # fmt: skip
    st.text_input("Description (optional)", key="study_description")

    st.subheader("Factors: what you'll change")
    st.caption(
        "**continuous**: any value between low and high. **discrete**: numbers, but only some values exist "
        "(type them under levels, like 22, 25, 27). **categorical**: named options (like CaCl2, BaCl2). "
        "**mixture**: a share of a blend, where all shares add up to a fixed total."
    )
    ss.factors_current = st.data_editor(
        ss.factors_base,
        key=f"factors_editor_{ss.version}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Name", help="For example nozzle_temp"),
            "type": st.column_config.SelectboxColumn("Type", options=list(FACTOR_KINDS), default="continuous"),
            "low": st.column_config.NumberColumn("Low", format="%g"),
            "high": st.column_config.NumberColumn("High", format="%g"),
            "levels": st.column_config.TextColumn("Levels", help="For discrete and categorical factors, separated by commas"),
            "units": st.column_config.TextColumn("Units"),
        },
    )
    if any(entry["type"] == "mixture" for entry in table_to_factors(ss.factors_current)):
        st.number_input(
            "Mixture components add up to", key="mixture_total", min_value=0.0001, format="%g",
            help="For example 1 for fractions, or 100 for percentages.",
        )  # fmt: skip
        st.caption(
            "Not sure whether your composition is a mixture? See 'Is composition a mixture?' in the Manual. "
            "If each ingredient is set independently and water makes up the rest, continuous factors are usually simpler."
        )

    st.subheader("Responses: what you'll measure")
    st.caption(
        "**Goal** says what a good result looks like. **Low** and **high** are limits: for maximise, low is the worst "
        "acceptable value and high is as good as it gets. For target, give the target as well as the limits on either "
        "side. **Importance** is how much it counts relative to the others. **Transform** is optional: use log for "
        "responses spanning a wide range, such as viscosity."
    )
    ss.responses_current = st.data_editor(
        ss.responses_base,
        key=f"responses_editor_{ss.version}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "units": st.column_config.TextColumn("Units"),
            "goal": st.column_config.SelectboxColumn("Goal", options=list(GOALS), default="none"),
            "low": st.column_config.NumberColumn("Low", format="%g"),
            "high": st.column_config.NumberColumn("High", format="%g"),
            "target": st.column_config.NumberColumn("Target", format="%g"),
            "importance": st.column_config.NumberColumn("Importance", min_value=0.1, default=1.0, format="%g"),
            "transform": st.column_config.SelectboxColumn("Transform", options=list(TRANSFORMS), default="none"),
        },
    )

    with st.expander("Optional: rule out impossible combinations (constraints)"):
        st.caption("For example `gelma_wt + alginate_wt` with a max of 16, if higher total polymer can't be extruded.")
        ss.constraints_current = st.data_editor(
            ss.constraints_base,
            key=f"constraints_editor_{ss.version}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "expr": st.column_config.TextColumn("Expression"),
                "min": st.column_config.NumberColumn("Min", format="%g"),
                "max": st.column_config.NumberColumn("Max", format="%g"),
            },
        )

    study, error = current_study()
    if study is None:
        if table_to_factors(ss.factors_current):
            st.error(error)
        else:
            st.info("Add at least one factor to continue, or load an example above.")
        return
    mixture = len(study.mixture_factors)
    process = len(study.process_factors)
    parts = [f"{mixture} mixture components"] if mixture else []
    parts += [f"{process} other factors"] if process else []
    parts += [f"{len(study.responses)} responses"]
    st.success("Study is ready: " + ", ".join(parts) + ".")
    design_changed_warning(study)
    next_button(PAGE_DESIGN)


# ---------------------------------------------------------------------- page 2: design
def _model_panel(study: Study) -> None:
    settings = dict(ss.model_settings)
    if settings.get("terms"):
        st.info("This study uses a hand-written list of model terms from its config file.")
        return
    automatic = "automatic"
    columns = st.columns(3)
    if study.process_factors:
        options = [automatic, *PROCESS_ORDERS]
        current = settings.get("process", automatic)
        with columns[0]:
            chosen = st.selectbox(
                "Process model", options, index=options.index(current), key=f"model_process_{ss.version}",
                help="linear: straight lines. interaction: effects can depend on each other. "
                "quadratic: also curves, which finding an optimum needs.",
            )  # fmt: skip
        settings.pop("process", None)
        if chosen != automatic:
            settings["process"] = chosen
    if study.mixture_factors:
        options = [automatic, *MIXTURE_ORDERS]
        current = settings.get("mixture", automatic)
        with columns[1]:
            chosen = st.selectbox(
                "Mixture model", options, index=options.index(current), key=f"model_mixture_{ss.version}",
                help="linear: simple averaging. quadratic: pairs of components can have synergy. "
                "special_cubic: three-way synergy too.",
            )  # fmt: skip
        settings.pop("mixture", None)
        if chosen != automatic:
            settings["mixture"] = chosen
    if study.mixture_factors and study.process_factors:
        options = list(CROSSING_ORDERS)
        current = settings.get("mixture_process", "linear")
        with columns[2]:
            chosen = st.selectbox(
                "Blend x process", options, index=options.index(current), key=f"model_crossing_{ss.version}",
                help="linear: the effect of a process setting can depend on the blend. none: it can't.",
            )  # fmt: skip
        settings["mixture_process"] = chosen
    ss.model_settings = settings


def _create(study: Study, design_settings: dict, model: dict | None) -> None:
    """Create a design and make it the study's run sheet."""
    ss.design_settings = dict(design_settings)
    if model is not None:
        keep = {key: ss.model_settings[key] for key in ("reduce", "alpha") if key in ss.model_settings}
        ss.model_settings = {**dict(model), **keep}
    updated, error = current_study()
    if updated is None:
        raise StudyError(error)
    design = create_design(updated)
    ss.design_notes = design.notes
    ss.design_kind = design.kind
    ss.design_signature = _factor_signature(current_config())
    ss.fit_cache = {}
    ss.figure_cache = {}
    set_sheet(build_runsheet(updated, design))


def _your_design(study: Study) -> None:
    sheet = ss.sheet
    if sheet is None or not ss.design_kind:
        return
    st.subheader("Your design")
    st.success("Design created. Check it below, then download the run sheet.")
    frame = sheet.copy()
    design_columns = [name for name in ("run", "block", "std_order", *study.factor_names) if name in frame]
    quality = evaluate_design(study, frame[design_columns])
    entry = DESIGNS[ss.design_kind]
    st.write(f"**{entry.title}**: {entry.summary}")
    columns = st.columns(4)
    columns[0].metric("Runs", quality.runs)
    columns[1].metric("Model terms", quality.terms)
    columns[2].metric("Spare runs to check the model", quality.residual_df)
    columns[3].metric("Replicates to measure noise", quality.pure_error_df)
    for note in ss.design_notes:
        st.info(note)
    for warning in quality.warnings:
        st.warning(warning)

    view = st.radio("Show", ["Run sheet", "Where the runs sit", "How entangled the terms are", "Precision of each term"], horizontal=True, key="design_view")
    key = _hash([current_config(), sheet.index.size])
    if view == "Run sheet":
        st.caption("Do the runs in the order of the 'run' column. The order is random on purpose, so that drift over time can't be mistaken for a factor effect.")
        st.dataframe(frame[design_columns], hide_index=True, width="stretch")
        st.caption("The downloaded run sheet also has an empty column for each response, and one for notes.")
    elif view == "Where the runs sit":
        if study.mixture_factors:
            show_figure(f"blends_{key}", lambda: plots.plot_design_ternary(study, frame), "Blend plots need at least 3 components.")
        if len(study.process_factors) >= 2:
            show_figure(f"points_{key}", lambda: plots.plot_design_points(study, frame))
    elif view == "How entangled the terms are":
        st.caption("Pale off-diagonal squares mean the effects can be told apart. Dark ones mean they can't.")
        if quality.estimable:
            terms = model_terms(study, frame)
            show_figure(f"correlation_{key}", lambda: plots.plot_term_correlation(study, frame, terms))
    elif quality.estimable:
        st.caption(
            f"Standard error is in multiples of the noise. Power is the chance of detecting a low-to-high change of "
            f"{quality.signal_to_noise:g} times the noise. Aim for at least 0.8 on the terms you care about. "
            "Mixture terms don't have a power figure, because their effect differs from blend to blend."
        )
        show_table(quality.term_table)

    columns = st.columns(3)
    columns[0].download_button("Download run sheet (CSV)", csv_bytes(sheet), f"{study.name}_runsheet.csv", "text/csv", key="download_sheet", type="primary")
    columns[1].download_button("Download study config", yaml_bytes(current_config()), f"{study.name}.yaml", "text/yaml", key="download_config")
    design = Design(frame[design_columns], model_terms(study, frame), ss.design_kind, int(ss.study_seed), ss.design_notes)
    text = report.design_report(study, design, quality)
    columns[2].download_button("Download design report", text.encode("utf-8"), f"{study.name}_design_report.txt", "text/plain", key="download_report")
    st.caption("Keep the run sheet and the config: they're all you need to carry on later, here or with the scripts.")
    next_button(PAGE_RESULTS)


def page_design() -> None:
    st.header(PAGE_DESIGN)
    study = require_study()
    if study is None:
        return
    design_changed_warning(study)
    if ss.sheet is not None and ss.design_kind:
        _your_design(study)
        st.divider()
        st.subheader("Choose a different design")
    else:
        st.write(
            "Say what you want to learn and what you can afford, and you'll get suggestions with real run counts. "
            "The Manual explains the reasoning."
        )
    _design_chooser(study)


def _design_chooser(study: Study) -> None:
    objective = st.radio(
        "What do you want to do?", list(OBJECTIVES), format_func=OBJECTIVES.get, index=None, key="objective",
        help="This is the main decision in planning an experiment. See 'Choosing a design' in the Manual.",
    )  # fmt: skip
    left, middle, right = st.columns(3)
    with left:
        budget = st.number_input("Most runs you can afford (0 = no limit)", min_value=0, step=1, key="max_runs")
    with middle:
        signal = st.number_input(
            "Smallest effect worth detecting (x noise)", min_value=0.5, step=0.5, format="%g", key="signal_to_noise",
            help="As a multiple of the run-to-run scatter. A change of twice the noise is a common starting point.",
        )  # fmt: skip
    with right:
        avoid = st.checkbox(
            "Some extreme combinations would fail", key="avoid_extremes", disabled=objective != "optimise",
            help="For example, high pressure with high speed and low temperature all at once.",
        )  # fmt: skip

    if st.button("Suggest designs", key="suggest", type="primary", disabled=objective is None):
        with st.spinner("Building and evaluating each option..."):
            ss.recommendations = recommend_designs(study, objective, budget or None, avoid, signal)
            ss.recommendations_for = _factor_signature(current_config())

    confirm = True
    if has_results():
        st.warning("You already have results in your run sheet. Creating a new design replaces them.")
        confirm = st.checkbox("Replace my current run sheet and its results", key="confirm_replace")

    results = ss.recommendations
    if results is not None and ss.get("recommendations_for") == _factor_signature(current_config()):
        if not results:
            st.warning("No built-in design fits this study. Check its factors and constraints.")
        else:
            st.markdown("**Suggestions**")
            show_table(pd.DataFrame([{"design": item.title, "within budget": item.within_budget, **item.quality.summary()} for item in results]))
            st.caption(
                "**Spare runs** check the model, and **replicates** measure noise. **Prediction error** is in multiples "
                "of the noise, so lower is better. **Lowest power** is the weakest chance of detecting your smallest "
                "effect: aim for at least 0.8."
            )
            with st.expander("Compare prediction error across the region"):
                st.caption("Lower is better. A flatter line means predictions are similarly reliable everywhere.")
                key = _hash([current_config(), [item.design for item in results]])
                show_figure(f"fds_{key}", lambda: plots.plot_fraction_of_design_space({item.title: item.quality for item in results}))
            for index, item in enumerate(results):
                with st.container(border=True):
                    tag = "Suggested: " if index == 0 and item.within_budget else ""
                    over = "" if item.within_budget else " (over your budget)"
                    st.markdown(f"**{tag}{item.title}: {item.runs} runs{over}**")
                    st.write(item.why)
                    st.caption(f"Keep in mind: {item.caution}")
                    if st.button("Use this design", key=f"use_{index}", disabled=not confirm):
                        _create(study, item.design, item.model if item.model is not None else None)
                        st.rerun()

    with st.expander("Or choose a design yourself, and adjust the model"):
        kinds = list(DESIGNS)
        current = ss.design_settings.get("type")
        if "manual_kind" not in ss:
            ss.manual_kind = current if current in kinds else kinds[0]
        kind = st.selectbox(
            "Design type", kinds,
            format_func=lambda name: f"{DESIGNS[name].title} ({DESIGNS[name].purpose})", key="manual_kind",
        )  # fmt: skip
        st.caption(DESIGNS[kind].summary)
        available = sorted(DESIGNS[kind].options | {"repeats", "replicate_runs", "blocks", "randomise"})
        start = {key: value for key, value in ss.design_settings.items() if key != "type"} if current == kind else {}
        text = st.text_area(
            "Design settings (optional, one per line)",
            value=yaml.safe_dump(start, sort_keys=False) if start else "",
            key=f"manual_settings_{kind}_{ss.version}",
            placeholder="blocks: 2",
        )
        st.caption("Available settings: " + ", ".join(available))
        _model_panel(study)
        if st.button("Create this design", key="create_manual", disabled=not confirm):
            try:
                extra = yaml.safe_load(text) if text.strip() else {}
                if extra and not isinstance(extra, dict):
                    raise StudyError("Write settings one per line, like blocks: 2")
                _create(study, {"type": kind, **(extra or {})}, None)
                st.rerun()
            except yaml.YAMLError as error:
                st.error(f"Couldn't read those settings. {error}")


# ---------------------------------------------------------------------- page 3: results
def page_results() -> None:
    st.header(PAGE_RESULTS)
    study = require_study()
    if study is None:
        return
    design_changed_warning(study)
    st.write(
        "Type results straight into the table, or fill in the downloaded run sheet in Excel and upload it. "
        "If a run failed, leave its result blank and say why in the notes."
    )

    uploaded = st.file_uploader(
        "Upload a filled-in run sheet, or data from an experiment you've already done (CSV)",
        type=["csv", "txt"], key="sheet_upload",
    )  # fmt: skip
    if uploaded is not None:
        signature = _hash([uploaded.name, hashlib.sha1(uploaded.getvalue()).hexdigest()])
        if ss.get("sheet_upload_signature") != signature:
            try:
                sheet = read_runsheet_bytes(study, uploaded.getvalue(), f"'{uploaded.name}'")
                if "notes" not in sheet:
                    sheet["notes"] = ""
                ss.sheet_upload_signature = signature
                ss.design_signature = _factor_signature(current_config())
                ss.fit_cache = {}
                ss.figure_cache = {}
                set_sheet(sheet)
                st.rerun()
            except StudyError as error:
                st.error(str(error))

    if ss.sheet is None:
        st.info("There's no run sheet yet. Create a design in step 2, or upload one above.")
        st.button(f"Go to {PAGE_DESIGN}", key="go_design", on_click=go_to, args=(PAGE_DESIGN,))
        return

    for name in study.response_names:
        if name not in ss.sheet:
            ss.sheet[name] = np.nan
            refresh_editors()
    if "notes" not in ss.sheet:
        ss.sheet["notes"] = ""
        refresh_editors()

    simulated_warning()
    fixed = [column for column in ss.sheet.columns if column not in (*study.response_names, "notes")]
    edited = st.data_editor(
        ss.sheet_base,
        key=f"sheet_editor_{ss.version}",
        hide_index=True,
        width="stretch",
        disabled=fixed,
        column_config={
            **{name: st.column_config.NumberColumn(study.response(name).label, format="%g") for name in study.response_names},
            "notes": st.column_config.TextColumn("Notes"),
        },
    )
    ss.sheet = edited

    total = len(edited)
    for name in study.response_names:
        done = int(edited[name].notna().sum())
        st.progress(done / total if total else 0.0, text=f"{study.response(name).label}: {done} of {total} runs")

    columns = st.columns(2)
    columns[0].download_button("Download run sheet (CSV)", csv_bytes(edited), f"{study.name}_runsheet.csv", "text/csv", key="download_sheet_results")
    if ss.simulation:
        replace_ok = True
        if has_results() and not edited["notes"].astype(str).str.contains("SIMULATED").all():
            replace_ok = columns[1].checkbox("Replace my results with simulated ones", key="confirm_simulate")
        if columns[1].button("Fill with simulated results (practice only)", key="simulate", disabled=not replace_ok):
            set_sheet(simulate_responses(study, edited, int(ss.study_seed)))
            st.rerun()
        st.caption(f"This study has built-in made-up formulas for practice. Simulated rows are marked '{SIMULATED_NOTE}'.")
    if has_results():
        next_button(PAGE_ANALYSE)


# ---------------------------------------------------------------------- page 4: analyse
def checked_sheet(study: Study) -> pd.DataFrame | None:
    if ss.sheet is None or not has_results():
        st.info("There are no results yet. Enter some in step 3.")
        st.button(f"Go to {PAGE_RESULTS}", key="go_results", on_click=go_to, args=(PAGE_RESULTS,))
        return None
    return clean_runsheet(study, ss.sheet)


def fit_for(study: Study, sheet: pd.DataFrame, name: str) -> ResponseFit:
    key = _hash([current_config(), _sheet_hash(), name])
    cache = ss.fit_cache
    if key not in cache:
        cache[key] = fit_response(study, sheet, study.response(name))
        if len(cache) > 40:
            cache.pop(next(iter(cache)))
    return cache[key]


STATUS_DISPLAY = {"OK": st.success, "CHECK": st.warning, "INFO": st.info}


def _analysis_controls(study: Study) -> None:
    with st.expander("Change the model"):
        _model_panel(study)
        left, right = st.columns(2)
        with left:
            reduce_model = st.checkbox(
                "Drop terms the data doesn't support", value=bool(ss.model_settings.get("reduce", False)),
                key=f"reduce_{ss.version}",
                help="Repeatedly removes the least significant term, keeping the model hierarchical.",
            )  # fmt: skip
        with right:
            alpha = st.number_input(
                "Keep terms with p below", min_value=0.01, max_value=0.5, step=0.01,
                value=float(ss.model_settings.get("alpha", 0.10)), key=f"alpha_{ss.version}", disabled=not reduce_model,
            )  # fmt: skip
        settings = dict(ss.model_settings)
        settings.pop("reduce", None)
        settings.pop("alpha", None)
        if reduce_model:
            settings["reduce"] = True
            settings["alpha"] = float(alpha)
        ss.model_settings = settings


def _set_transform(name: str, widget_key: str) -> None:
    table = ss.responses_current.copy()
    table.loc[table["name"] == name, "transform"] = ss[widget_key]
    ss.responses_current = table
    refresh_editors()


def page_analyse() -> None:
    st.header(PAGE_ANALYSE)
    study = require_study()
    if study is None:
        return
    design_changed_warning(study)
    sheet = checked_sheet(study)
    if sheet is None:
        return
    simulated_warning()

    available = [name for name in study.response_names if sheet[name].notna().any()]
    left, right = st.columns([3, 1])
    with left:
        name = st.selectbox("Response", available, format_func=lambda n: study.response(n).label, key="analysis_response")
    with right:
        transform_key = f"transform_{name}_{ss.version}"
        st.selectbox(
            "Transform", list(TRANSFORMS), index=list(TRANSFORMS).index(study.response(name).transform), key=transform_key,
            on_change=_set_transform, args=(name, transform_key),
            help="Try log if the scatter grows with the size of the response.",
        )  # fmt: skip
    _analysis_controls(study)
    study, _ = current_study()

    fit = fit_for(study, sheet, name)
    metrics = fit_metrics(fit)
    fit_key = _hash([current_config(), _sheet_hash(), name])

    columns = st.columns(4)
    columns[0].metric("R²", f"{metrics['r2']:.3f}", help="The share of the variation the model explains.")
    columns[1].metric("Adjusted R²", f"{metrics['adj_r2']:.3f}", help="The same, with a penalty for extra terms.")
    columns[2].metric("Predicted R²", f"{metrics['pred_r2']:.3f}", help="How well the model predicts runs it hasn't seen. The most honest of the three.")
    columns[3].metric("Runs used", f"{metrics['runs']} of {fit.total_runs}")

    st.subheader("Verdict")
    for check in interpret_fit(fit):
        STATUS_DISPLAY[check.status](f"**{check.topic}.** {check.message}")
    if fit.removed_terms:
        st.caption("Terms removed by model reduction: " + ", ".join(fit.removed_terms))

    st.subheader("Details")
    views = ["What matters", "Effect of each factor", "Interactions", "Surfaces", "Model checks", "Tables"]
    view = st.radio("Show", views, horizontal=True, key="analysis_view", label_visibility="collapsed")
    if view == "What matters":
        st.caption(
            "Longer bars stand out more clearly from the noise. With no spare runs, this becomes a half-normal plot, "
            "where effects far from the line are real."
        )
        show_figure(f"effects_{fit_key}", lambda: plots.plot_effects(fit))
    elif view == "Effect of each factor":
        st.caption(
            "How the response changes with one factor, with the others held at the centre of the region. "
            "For a blend, each line changes one component while the others keep their proportions to each other."
        )
        show_figure(f"main_{fit_key}", lambda: plots.plot_main_effects(fit))
    elif view == "Interactions":
        st.caption("If the lines aren't parallel, one factor's effect depends on the other.")
        show_figure(f"interactions_{fit_key}", lambda: plots.plot_interactions(fit), "The model has no interactions between process factors.")
    elif view == "Surfaces":
        _surfaces(study, fit, fit_key, None)
    elif view == "Model checks":
        st.caption("These plots show whether the model's assumptions hold. Look for patterns: there shouldn't be any.")
        show_figure(f"checks_{fit_key}", lambda: plots.plot_diagnostics(fit), "Needs at least 2 spare runs.")
    else:
        _analysis_tables(fit)

    st.download_button("Download the report for this response", report.fit_report(fit).encode("utf-8"), f"{study.name}_{name}_report.txt", "text/plain", key="download_fit_report")
    if any(response.goal != "none" for response in study.responses):
        next_button(PAGE_OPTIMISE)


def _surfaces(study: Study, fit: ResponseFit, fit_key: str, best: dict | None) -> None:
    pairs = plots.surface_pairs(fit, max_factors=6)
    triples = plots.ternary_triples(study)
    options = [f"{a} and {b}" for a, b in pairs] + [" + ".join(triple) for triple in triples]
    if not options:
        st.info("Surfaces need at least two numeric factors in the model, or three mixture components.")
        return
    suffix = "best" if best else "centre"
    choice = st.selectbox("Plot over", options, key=f"surface_choice_{fit.response.name}_{suffix}")
    hold_note = "the best settings" if best else "the centre of the region"
    st.caption(f"Factors not shown are held at {hold_note}, and the title lists their values.")
    key = f"surface_{fit_key}_{suffix}_{choice}"
    if " and " in choice:
        a, b = choice.split(" and ")
        show_figure(key, lambda: plots.plot_surface(fit, a, b, best, best))
    else:
        show_figure(key, lambda: plots.plot_ternary(fit, choice.split(" + "), best, best))


def _analysis_tables(fit: ResponseFit) -> None:
    coefficients = coefficient_table(fit).drop(columns=["group"])
    st.markdown("**Coefficients**, in coded units so their sizes can be compared directly")
    show_table(coefficients)
    if fit.residual_df <= 0:
        lenth = lenth_analysis(coefficient_table(fit))
        if lenth:
            st.write(f"Lenth's margin of error: {lenth['margin_of_error']:.4g}. Standing out: {', '.join(lenth['active']) or 'none'}.")
    else:
        st.markdown("**Analysis of variance**")
        show_table(anova_table(fit))
    st.markdown("**Run by run**")
    show_table(run_diagnostics(fit))


# ---------------------------------------------------------------------- page 5: optimise
def _what_if(study: Study, fits: list[ResponseFit], best: pd.Series) -> None:
    with st.expander("Try your own settings"):
        st.caption("See what the models predict for settings you have in mind.")
        values: dict = {}
        columns = st.columns(min(4, max(len(study.factors), 1)))
        components = study.mixture_names
        for index, factor in enumerate(study.factors):
            column = columns[index % len(columns)]
            key = f"whatif_{factor.name}"
            if factor.kind == "categorical":
                values[factor.name] = column.selectbox(factor.label, factor.levels, index=list(factor.levels).index(best[factor.name]), key=key)
            elif factor.kind == "discrete":
                levels = list(factor.levels)
                start = levels.index(float(factor.nearest_level(best[factor.name])))
                values[factor.name] = column.selectbox(factor.label, levels, index=start, key=key)
            elif factor.kind == "mixture" and factor.name == components[-1]:
                remainder = float(study.mixture_total) - sum(values[name] for name in components[:-1])
                values[factor.name] = remainder
                column.metric(factor.label, f"{remainder:.4g}", help="Whatever remains of the total.")
            else:
                low, high = float(factor.low), float(factor.high)
                values[factor.name] = column.slider(factor.label, low, high, float(np.clip(best[factor.name], low, high)), step=(high - low) / 200, key=key)
        frame = pd.DataFrame([values])[study.factor_names]
        if not study.is_feasible(frame).all():
            st.warning("These settings break a limit or constraint, so treat the predictions with caution.")
        rows = []
        for fit in fits:
            prediction = fit.predict(frame).iloc[0]
            rows.append(
                {
                    "response": fit.response.label,
                    "predicted": prediction["predicted"],
                    "a new run would fall from": prediction["pi_low"],
                    "to": prediction["pi_high"],
                }
            )
        show_table(pd.DataFrame(rows))


def _results_zip(study: Study, sheet: pd.DataFrame, fits: list[ResponseFit], solutions: pd.DataFrame | None) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / f"{study.name}.yaml").write_bytes(yaml_bytes(current_config()))
        (root / "runsheet.csv").write_bytes(csv_bytes(sheet))
        for fit in fits:
            save_fit_outputs(fit, root / fit.response.name)
        if solutions is not None:
            save_optimum_outputs(study, fits, solutions, root / "optimum")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    return buffer.getvalue()


def page_optimise() -> None:
    st.header(PAGE_OPTIMISE)
    study = require_study()
    if study is None:
        return
    design_changed_warning(study)
    sheet = checked_sheet(study)
    if sheet is None:
        return
    simulated_warning()

    available = [name for name in study.response_names if sheet[name].notna().any()]
    fits = [fit_for(study, sheet, name) for name in available]
    active = [fit for fit in fits if fit.response.goal != "none"]
    if not active:
        st.info("Give at least one response a goal in step 1 to find its best settings.")
        st.button(f"Go to {PAGE_DEFINE}", key="go_define_goals", on_click=go_to, args=(PAGE_DEFINE,))
        return

    st.write(
        "This looks for the settings that best meet every goal at once. Each predicted response is scored from "
        "0 (unacceptable) to 1 (ideal), and one unacceptable response makes the whole setting unacceptable."
    )
    rows = []
    for fit in active:
        limits = desirability_limits(fit)
        response = fit.response
        rows.append(
            {
                "response": response.label,
                "goal": response.goal,
                "low limit": limits.low,
                "target": "" if limits.target is None else f"{limits.target:g}",
                "high limit": limits.high,
                "importance": response.importance,
            }
        )
    show_table(pd.DataFrame(rows))
    if any(fit.response.low is None or fit.response.high is None for fit in active):
        st.caption("Limits you didn't set are filled in from the lowest and highest values you measured.")

    key = _hash([current_config(), _sheet_hash()])
    if st.button("Find best settings", key="find_best", type="primary"):
        with st.spinner("Searching..."):
            ss.solutions = find_optimum(study, fits, seed=int(ss.study_seed))
            ss.solutions_key = key
    if ss.solutions is None or ss.solutions_key != key:
        return

    solutions = ss.solutions
    best = solutions.iloc[0]
    st.subheader("Best settings found")
    columns = st.columns(min(4, len(study.factors)))
    for index, factor in enumerate(study.factors):
        value = best[factor.name]
        columns[index % len(columns)].metric(factor.label, value if isinstance(value, str) else f"{float(value):.4g}")

    st.subheader("What the models predict")
    st.metric("Overall desirability", f"{best['desirability']:.3f}", help="0 is unacceptable, 1 is ideal.")
    rows = []
    for fit in active:
        name = fit.response.name
        rows.append(
            {
                "response": fit.response.label,
                "predicted": best[f"{name}_predicted"],
                "a new run would fall from": best[f"{name}_pi_low"],
                "to": best[f"{name}_pi_high"],
                "desirability": best[f"{name}_desirability"],
            }
        )
    show_table(pd.DataFrame(rows))
    if best["desirability"] <= 0:
        st.warning("No settings meet all the limits at once. These are the closest found. Relax a limit, widen the factor ranges, or accept a trade-off.")
    if not bool(best["well_supported"]):
        st.warning("These settings are far from the runs you did, so this prediction is an extrapolation.")
    st.info(
        "**Before you rely on this:** it's a prediction, not a measurement. Prepare these settings and do at least 3 "
        "confirmation runs. If the results fall inside the ranges above, the model is confirmed. If they don't, something "
        "is missing from the model, so focus the next experiment around this region."
    )

    if len(solutions) > 1:
        with st.expander("Other good options"):
            shown = ["desirability", *study.factor_names, *[f"{fit.response.name}_predicted" for fit in active]]
            show_table(solutions[shown])
    _what_if(study, active, best)

    st.subheader("The sweet spot")
    best_values = {name: best[name] for name in study.factor_names}
    limits = [desirability_limits(fit) for fit in active]
    pairs = plots.surface_pairs(active[0], max_factors=5)
    triples = plots.ternary_triples(study)
    options = [f"{a} and {b}" for a, b in pairs] + [" + ".join(triple) for triple in triples]
    if options:
        st.caption("Dark areas meet your goals best. A wide dark area means the process tolerates small errors in those settings.")
        choice = st.selectbox("Plot over", options, key="sweet_choice")
        figure_key = f"sweet_{key}_{choice}"
        if " and " in choice:
            a, b = choice.split(" and ")
            show_figure(figure_key, lambda: plots.plot_desirability_map(active, limits, a, b, best_values, best_values))
        else:
            show_figure(figure_key, lambda: plots.plot_desirability_ternary(active, limits, choice.split(" + "), best_values, best_values))

    st.subheader("Each response around the best settings")
    name = st.selectbox("Response", [fit.response.name for fit in active], format_func=lambda n: study.response(n).label, key="best_response")
    fit = next(item for item in active if item.response.name == name)
    _surfaces(study, fit, _hash([key, name]), best_values)

    st.subheader("Save your work")
    columns = st.columns(3)
    columns[0].download_button("Download best settings (CSV)", csv_bytes(solutions), f"{study.name}_best_settings.csv", "text/csv", key="download_solutions")
    columns[1].download_button("Download summary", report.optimum_report(study, active, solutions).encode("utf-8"), f"{study.name}_best_settings.txt", "text/plain", key="download_optimum_report")
    if columns[2].button("Prepare all reports and figures", key="prepare_zip"):
        with st.spinner("Drawing every figure as an editable SVG..."):
            ss.results_zip = _results_zip(study, sheet, fits, solutions)
    if ss.results_zip:
        columns[2].download_button("Download everything (ZIP)", ss.results_zip, f"{study.name}_results.zip", "application/zip", key="download_zip", type="primary")


# ---------------------------------------------------------------------- manual
def page_manual() -> None:
    path = ROOT / "docs" / "manual.md"
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.info("The manual isn't included with this copy of the app.")


# ---------------------------------------------------------------------- layout
def _progress() -> list[tuple[bool, str]]:
    """Summarise what's been done so far, for the sidebar."""
    return [
        (current_study()[0] is not None, "Study defined"),
        (ss.sheet is not None, "Design created"),
        (has_results(), "Results entered"),
        (ss.solutions is not None, "Best settings found"),
    ]


def sidebar() -> str:
    with st.sidebar:
        st.title("doekit")
        st.caption("Design, analyse and optimise experiments")
        # These labels have to stay fixed: Streamlit tells widgets apart partly by their
        # options, so changing them would send people back to the first step.
        page = st.radio("Steps", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        st.caption("  \n".join(f"{'✓' if done else '○'} {label}" for done, label in _progress()))
        if ss.sheet is not None:
            total = len(ss.sheet)
            names = [name for name in table_to_response_names() if name in ss.sheet]
            complete = int(ss.sheet[names].notna().all(axis=1).sum()) if names else 0
            st.caption(f"**{_text(ss.study_name)}**: {complete} of {total} runs complete")
        st.caption(
            "Your work isn't saved here, and closing this page loses it. Download your run sheet and config "
            "to keep them, and load them again to carry on."
        )
    return page


def main() -> None:
    st.set_page_config(page_title="doekit: design of experiments", page_icon="🧪", layout="wide")
    init_state()
    page = sidebar()
    if page != ss.last_page:
        refresh_editors()
        ss.last_page = page
    screens = {
        PAGE_DEFINE: page_define,
        PAGE_DESIGN: page_design,
        PAGE_RESULTS: page_results,
        PAGE_ANALYSE: page_analyse,
        PAGE_OPTIMISE: page_optimise,
        PAGE_MANUAL: page_manual,
    }
    try:
        screens[page]()
    except StudyError as error:
        st.error(str(error))
    except Exception:  # keep the app usable and give people something to report
        st.error("Something unexpected went wrong. The details below will help whoever looks after this tool.")
        st.code(traceback.format_exc())


main()
