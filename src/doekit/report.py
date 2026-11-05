"""Write plain-text reports that explain results as well as listing them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import (
    ResponseFit,
    anova_table,
    coefficient_table,
    fit_metrics,
    interpret_fit,
    lack_of_fit,
    lenth_analysis,
)
from .designs import DESIGNS, Design
from .evaluate import DesignQuality
from .model import term_label
from .optimise import desirability_limits
from .study import Study

_RULE = "=" * 78


def _heading(text: str) -> str:
    return f"\n{text}\n{'-' * len(text)}"


def _table(frame: pd.DataFrame, digits: int = 4) -> str:
    def number(value):
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return ""
            return f"{value:.{digits}g}"
        return value

    return frame.map(number).to_string(index=False)


def describe_study(study: Study) -> str:
    lines = []
    for factor in study.factors:
        if factor.kind in ("categorical", "discrete"):
            detail = "levels " + ", ".join(str(level) for level in factor.levels)
        else:
            detail = f"{float(factor.low):g} to {float(factor.high):g}"
        lines.append(f"  {factor.label:<34} {factor.kind:<12} {detail}")
    if study.mixture_factors:
        lines.append(f"  Mixture components add up to {study.mixture_total:g}.")
    for constraint in study.constraints:
        lines.append(f"  Constraint: {constraint.describe()}")
    return "\n".join(lines)


def design_report(study: Study, design: Design, quality: DesignQuality) -> str:
    entry = DESIGNS[design.kind]
    out = [_RULE, f"DESIGN REPORT: {study.name}", _RULE]
    if study.description:
        out.append(study.description)
    out.append(_heading("Factors"))
    out.append(describe_study(study))
    out.append(_heading("Design"))
    out.append(f"  Type:          {entry.title} ({entry.purpose})")
    out.append(f"  What it does:  {entry.summary}")
    out.append(f"  Runs:          {quality.runs}, of which {quality.distinct_runs} are distinct")
    out.append(f"  Seed:          {design.seed} (the same seed recreates this exact design)")
    if design.notes:
        out.append(_heading("Notes on this design"))
        out.extend(f"  - {note}" for note in design.notes)
    out.append(_heading("Model it is planned for"))
    out.append("  " + ", ".join(term_label(term) for term in design.terms))
    out.append(_heading("What this design can tell you"))
    out.append(f"  Model terms to estimate:           {quality.terms}")
    out.append(f"  Runs left over to check the model: {quality.residual_df}")
    out.append(f"  of which from replicates (noise):  {quality.pure_error_df}")
    if quality.estimable:
        out.append(f"  Typical prediction error:          {quality.mean_prediction_se:.2f} x noise")
        out.append(f"  Worst prediction error:            {quality.max_prediction_se:.2f} x noise")
        out.append(f"  D-efficiency:                      {quality.d_efficiency:.1f}%")
        out.append(_heading(f"Precision of each term (power: chance of detecting a {quality.signal_to_noise:g} x noise change)"))
        out.append(_table(quality.term_table))
    if quality.warnings:
        out.append(_heading("Things to consider before starting"))
        out.extend(f"  ! {warning}" for warning in quality.warnings)
    out.append(_heading("Next steps"))
    out.append("  1. Run the experiments in the order of the 'run' column. The order is random on purpose.")
    out.append("  2. Type each result into its response column in the run sheet.")
    out.append("  3. If a run fails, leave its result blank and say why in the notes column.")
    return "\n".join(out) + "\n"


def fit_report(fit: ResponseFit) -> str:
    response = fit.response
    metrics = fit_metrics(fit)
    out = [_RULE, f"ANALYSIS OF {response.label}", _RULE]
    out.append(f"Runs used: {metrics['runs']} of {fit.total_runs}.")
    if response.transform != "none":
        out.append(f"Fitted on the {response.transform} scale. Statistics below are on that scale, predictions in original units.")
    if fit.removed_terms:
        out.append("Terms removed by model reduction: " + ", ".join(fit.removed_terms))

    out.append(_heading("Verdict"))
    for check in interpret_fit(fit):
        out.append(f"  [{check.status:<5}] {check.topic}: {check.message}")

    out.append(_heading("Model fit"))
    out.append(f"  R2           {metrics['r2']:.4f}   share of the variation the model explains")
    out.append(f"  Adjusted R2  {metrics['adj_r2']:.4f}   the same, with a penalty for extra terms")
    out.append(f"  Predicted R2 {metrics['pred_r2']:.4f}   how well it predicts runs it hasn't seen")
    out.append(f"  RMSE         {metrics['rmse']:.4g}   typical size of the unexplained scatter")

    coefficients = coefficient_table(fit)
    out.append(_heading("Coefficients (coded units, so sizes are directly comparable)"))
    out.append(_table(coefficients.drop(columns=["group"])))
    if fit.residual_df <= 0:
        lenth = lenth_analysis(coefficients)
        if lenth:
            out.append(_heading("Lenth's method (no degrees of freedom are left for ordinary tests)"))
            out.append(f"  Margin of error for effects: {lenth['margin_of_error']:.4g}")
            active = ", ".join(lenth["active"]) if lenth["active"] else "none"
            out.append(f"  Effects standing out from the noise: {active}")
    else:
        out.append(_heading("Analysis of variance"))
        out.append(_table(anova_table(fit)))
        lof = lack_of_fit(fit)
        if lof is None:
            out.append("\n  Lack of fit couldn't be tested (it needs replicates and spare distinct runs).")
    out.append(_heading("How to read the coefficients"))
    if fit.study.mixture_factors:
        out.append("  Component terms (like 'gelma') give the predicted response for that component alone,")
        out.append("  at the top of its allowed range. Compare them with each other, not with zero.")
        out.append("  Blend terms (like 'gelma:alginate') show synergy (positive) or antagonism (negative).")
    if fit.study.process_factors:
        out.append("  For a process factor, the coefficient is half the change from its low to its high setting.")
    return "\n".join(out) + "\n"


def optimum_report(study: Study, fits: list[ResponseFit], solutions: pd.DataFrame) -> str:
    active = [fit for fit in fits if fit.response.goal != "none"]
    out = [_RULE, f"BEST SETTINGS: {study.name}", _RULE]
    out.append(_heading("Goals"))
    for fit in active:
        limits = desirability_limits(fit)
        response = fit.response
        if response.goal == "maximise":
            goal = f"maximise: worst acceptable {limits.low:g}, as good as it gets {limits.high:g}"
        elif response.goal == "minimise":
            goal = f"minimise: as good as it gets {limits.low:g}, worst acceptable {limits.high:g}"
        elif response.goal == "target":
            goal = f"target {limits.target:g}: acceptable from {limits.low:g} to {limits.high:g}"
        else:
            goal = f"stay between {limits.low:g} and {limits.high:g}"
        out.append(f"  {response.label}: {goal} (importance {response.importance:g})")
    best = solutions.iloc[0]
    out.append(_heading("Best settings found"))
    for name in study.factor_names:
        factor = study.factor(name)
        value = best[name]
        text = value if isinstance(value, str) else f"{float(value):.4g}"
        out.append(f"  {factor.label:<34} {text}")
    out.append(_heading("What the models predict at these settings"))
    out.append(f"  Overall desirability: {best['desirability']:.3f} (0 is unacceptable, 1 is ideal)")
    for fit in active:
        name = fit.response.name
        low, high = best[f"{name}_pi_low"], best[f"{name}_pi_high"]
        interval = f"   a single new run should fall between {low:.4g} and {high:.4g}" if np.isfinite(low) else ""
        out.append(f"  {fit.response.label:<34} {best[f'{name}_predicted']:.4g}{interval}")
    if best["desirability"] <= 0:
        out.append("\n  ! No settings meet all the limits at once. These are the closest found. Relax a limit,")
        out.append("    widen the factor ranges, or accept a trade-off.")
    if not bool(best["well_supported"]):
        out.append("\n  ! These settings are far from the runs you did, so the prediction is an extrapolation.")
    if len(solutions) > 1:
        out.append(_heading("Other good options"))
        columns = ["desirability", *study.factor_names, *[f"{fit.response.name}_predicted" for fit in active]]
        out.append(_table(solutions[columns]))
    out.append(_heading("Before you rely on this"))
    out.append("  This is a prediction, not a measurement. Prepare the best settings and run at least 3 confirmation")
    out.append("  runs. If they fall inside the ranges above, the model is confirmed. If not, something is missing")
    out.append("  from the model, and the next experiment should focus around this region.")
    return "\n".join(out) + "\n"
