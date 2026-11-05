"""The steps of a DOE study, each reading from and writing to the project's folders.

The scripts in scripts/ are thin wrappers around these functions.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from . import plots, report
from .advisor import recommend_designs
from .analysis import ResponseFit, anova_table, coefficient_table, fit_all, run_diagnostics
from .designs import DESIGNS, create_design
from .evaluate import evaluate_design
from .model import model_terms
from .optimise import desirability_limits, find_optimum
from .runsheet import read_runsheet, write_runsheet
from .simulate import simulate_responses
from .study import Study, StudyError, load_study


def run_script(function, *args, **kwargs):
    """Run a step and show problems as a clear message rather than a traceback."""
    try:
        return function(*args, **kwargs)
    except StudyError as error:
        print(f"\nProblem: {error}\n", file=sys.stderr)
        raise SystemExit(1) from None


def _save(figure, folder: Path, name: str, study: Study) -> Path | None:
    if figure is None:
        return None
    return plots.save_figure(figure, folder / f"{name}.{study.figure_format}")


def create_design_step(config, force: bool = False) -> Path:
    """Generate the design, write the run sheet, and report on the design's quality."""
    study = load_study(config)
    design = create_design(study)
    quality = evaluate_design(study, design.frame, design.terms)
    path = write_runsheet(study, design, force=force)

    folder = study.results_dir / "design"
    folder.mkdir(parents=True, exist_ok=True)
    text = report.design_report(study, design, quality)
    (folder / "design_report.txt").write_text(text, encoding="utf-8")
    if quality.estimable:
        quality.term_table.to_csv(folder / "term_precision.csv", index=False)
        _save(plots.plot_term_correlation(study, design.frame, design.terms), folder, "term_correlation", study)
        _save(plots.plot_fraction_of_design_space({DESIGNS[design.kind].title: quality}), folder, "prediction_error", study)
    _save(plots.plot_design_points(study, design.frame), folder, "design_points", study)
    _save(plots.plot_design_ternary(study, design.frame), folder, "design_blends", study)

    print(text)
    print(f"Run sheet:  {path}")
    print(f"Report:     {folder}")
    return path


def simulate_step(config, force: bool = False) -> Path:
    """Fill the run sheet with made-up results, for practising the workflow."""
    study = load_study(config)
    sheet = read_runsheet(study)
    if sheet[study.response_names].notna().any().any() and not force:
        raise StudyError(
            f"The run sheet at {study.runsheet_path} already has results, so they were not replaced. "
            "Use --force if you're sure they can be overwritten."
        )
    simulated = simulate_responses(study, sheet)
    simulated.to_csv(study.runsheet_path, index=False)
    print(f"Simulated results written to {study.runsheet_path}")
    print("These are made up, for practice. They are not real measurements.")
    return study.runsheet_path


def save_fit_outputs(fit: ResponseFit, folder: Path) -> str:
    """Save one response's report, tables and figures to a folder."""
    study = fit.study
    folder.mkdir(parents=True, exist_ok=True)
    text = report.fit_report(fit)
    (folder / "report.txt").write_text(text, encoding="utf-8")
    coefficient_table(fit).to_csv(folder / "coefficients.csv", index=False)
    run_diagnostics(fit).to_csv(folder / "run_diagnostics.csv", index=False)
    if fit.residual_df > 0:
        anova_table(fit).to_csv(folder / "anova.csv", index=False)
    _save(plots.plot_effects(fit), folder, "effects", study)
    _save(plots.plot_diagnostics(fit), folder, "model_checks", study)
    _save(plots.plot_main_effects(fit), folder, "main_effects", study)
    _save(plots.plot_interactions(fit), folder, "interactions", study)
    for a, b in plots.surface_pairs(fit):
        _save(plots.plot_surface(fit, a, b), folder / "surfaces", f"surface_{a}_vs_{b}", study)
    for components in plots.ternary_triples(study):
        _save(plots.plot_ternary(fit, components), folder / "surfaces", "blend_" + "_".join(components), study)
    return text


def analyse_step(config) -> list[ResponseFit]:
    """Fit and check a model for each response that has results."""
    study = load_study(config)
    sheet = read_runsheet(study)
    fits, skipped = fit_all(study, sheet)
    if not fits:
        raise StudyError(f"No results found in {study.runsheet_path}. Fill in the response columns first.")
    for note in skipped:
        print(f"Note: {note}")
    for fit in fits:
        folder = study.results_dir / fit.response.name
        print(save_fit_outputs(fit, folder))
        print(f"Figures and tables: {folder}\n")
    return fits


def save_optimum_outputs(study: Study, fits: list[ResponseFit], solutions: pd.DataFrame, folder: Path) -> str:
    """Save the best settings, plus figures drawn through them."""
    folder.mkdir(parents=True, exist_ok=True)
    text = report.optimum_report(study, fits, solutions)
    (folder / "report.txt").write_text(text, encoding="utf-8")
    solutions.to_csv(folder / "solutions.csv", index=False)
    best = {name: solutions.iloc[0][name] for name in study.factor_names}
    active = [fit for fit in fits if fit.response.goal != "none"]
    limits = [desirability_limits(fit) for fit in active]

    pairs = plots.surface_pairs(active[0], max_factors=4) if active else []
    for a, b in pairs:
        _save(plots.plot_desirability_map(active, limits, a, b, best, best), folder, f"sweet_spot_{a}_vs_{b}", study)
    for components in plots.ternary_triples(study):
        _save(plots.plot_desirability_ternary(active, limits, components, best, best), folder, "sweet_spot_" + "_".join(components), study)
    for fit in active:
        for a, b in plots.surface_pairs(fit, max_factors=4):
            _save(plots.plot_surface(fit, a, b, best, best), folder / fit.response.name, f"surface_{a}_vs_{b}", study)
        for components in plots.ternary_triples(study):
            _save(plots.plot_ternary(fit, components, best, best), folder / fit.response.name, "blend_" + "_".join(components), study)
    return text


def optimise_step(config) -> pd.DataFrame:
    """Find the settings that best meet every response's goal."""
    study = load_study(config)
    sheet = read_runsheet(study)
    fits, skipped = fit_all(study, sheet)
    if not fits:
        raise StudyError(f"No results found in {study.runsheet_path}. Fill in the response columns first.")
    for note in skipped:
        print(f"Note: {note}")
    solutions = find_optimum(study, fits)
    folder = study.results_dir / "optimum"
    print(save_optimum_outputs(study, fits, solutions, folder))
    print(f"Figures and tables: {folder}")
    return solutions


def recommend_step(config, objective: str, max_runs: int | None, avoid_extremes: bool, signal_to_noise: float) -> None:
    """Print design suggestions for a study."""
    study = load_study(config)
    results = recommend_designs(study, objective, max_runs, avoid_extremes, signal_to_noise)
    if not results:
        raise StudyError("No built-in design fits this study. Check its factors and constraints.")
    print(f"\nDesigns for: {study.name}\n")
    for index, item in enumerate(results, start=1):
        tag = "" if item.within_budget else "  [over your run budget]"
        print(f"{index}. {item.title}: {item.runs} runs{tag}")
        print(f"   Why:     {item.why}")
        print(f"   Caution: {item.caution}")
        print(f"   Config:  design: {item.design}")
        if item.model:
            print(f"            model: {item.model}")
        print(f"   Quality: {item.quality.summary()}\n")


def _parse_design_spec(text: str) -> dict:
    """Read a design written like 'i_optimal:runs=30,starts=10'."""
    kind, _, settings = text.partition(":")
    design = {"type": kind}
    for pair in filter(None, settings.split(",")):
        key, _, value = pair.partition("=")
        design[key] = yaml.safe_load(value)
    return design


def compare_step(config, specs: list[str], signal_to_noise: float = 2.0) -> pd.DataFrame:
    """Compare several designs for one study, side by side."""
    study = load_study(config)
    rows, qualities = [], {}
    for text in specs:
        trial = replace(study, design=_parse_design_spec(text))
        try:
            design = create_design(trial)
            quality = evaluate_design(trial, design.frame, model_terms(trial, design.frame), signal_to_noise)
        except StudyError as error:
            print(f"{text}: not possible here. {error}\n")
            continue
        if not quality.estimable:
            print(f"{text}: not possible here. {quality.warnings[0]}\n")
            continue
        rows.append({"design": text, **quality.summary()})
        qualities[text] = quality
    if not rows:
        raise StudyError("None of those designs can be built for this study.")
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    folder = study.results_dir / "design"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "design_comparison.csv", index=False)
    path = _save(plots.plot_fraction_of_design_space(qualities), folder, "design_comparison", study)
    print(f"\nPrediction error comparison: {path}")
    return table
