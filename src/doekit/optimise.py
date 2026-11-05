"""Find the settings that best satisfy every response at once.

Each predicted response is converted to a desirability between 0 (unacceptable) and
1 (ideal), and these are combined with a weighted geometric mean (Derringer & Suich,
1980). A geometric mean means that one unacceptable response makes the whole setting
unacceptable, however good the others are, which matches how formulations are judged.

The search itself is scipy's differential evolution, which copes with the several
peaks a combined objective like this can have.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .analysis import ResponseFit
from .study import Response, Study, StudyError

# Among equally desirable settings, prefer those where the model is best supported.
# This is small enough not to change the answer when there's one clear optimum.
_TIE_BREAK = 1e-4
_MAX_COMBINATIONS = 200


@dataclass(frozen=True)
class Limits:
    """The desirability limits for one response, with defaults filled in from the data."""

    response: Response
    low: float
    high: float
    target: float | None


def desirability_limits(fit: ResponseFit) -> Limits:
    response = fit.response
    low = response.low if response.low is not None else float(np.min(fit.observed))
    high = response.high if response.high is not None else float(np.max(fit.observed))
    if response.goal == "target":
        return Limits(response, low, high, float(response.target))
    return Limits(response, low, high, None)


def desirability(limits: Limits, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert responses into desirability.

    Also returns how far each value falls short of being acceptable (0 if it's
    acceptable). This gives the search a slope to follow in regions where desirability
    is a flat 0, which could otherwise leave it with no direction to move in.
    """
    goal = limits.response.goal
    shape = limits.response.shape
    low, high = limits.low, limits.high
    values = np.asarray(values, dtype=float)
    span = max(high - low, 1e-12)
    if goal == "maximise":
        d = np.clip((values - low) / span, 0, 1) ** shape
        short = np.maximum(0.0, (low - values) / span)
    elif goal == "minimise":
        d = np.clip((high - values) / span, 0, 1) ** shape
        short = np.maximum(0.0, (values - high) / span)
    elif goal == "target":
        target = limits.target
        below = np.clip((values - low) / max(target - low, 1e-12), 0, 1) ** shape
        above = np.clip((high - values) / max(high - target, 1e-12), 0, 1) ** shape
        d = np.where(values <= target, below, above)
        short = np.maximum(0.0, (low - values) / span) + np.maximum(0.0, (values - high) / span)
    elif goal == "in_range":
        d = ((values >= low) & (values <= high)).astype(float)
        short = np.maximum(0.0, (low - values) / span) + np.maximum(0.0, (values - high) / span)
    else:
        d = np.ones_like(values)
        short = np.zeros_like(values)
    return d, short


def overall_desirability(
    fits: list[ResponseFit], limits: list[Limits], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Combine each response's desirability into one score per row."""
    weights = np.array([item.response.importance for item in limits], dtype=float)
    log_total = np.zeros(len(frame))
    shortfall = np.zeros(len(frame))
    each = {}
    for fit, item, weight in zip(fits, limits, weights):
        d, short = desirability(item, fit.predict_mean(frame))
        each[item.response.name] = d
        with np.errstate(divide="ignore"):
            log_total += weight * np.log(d)
        shortfall += short
    return np.exp(log_total / weights.sum()), shortfall, each


def _fixed_combinations(study: Study) -> list[dict]:
    """List every combination of the factors that can only take set values."""
    fixed = [f for f in study.process_factors if f.kind in ("categorical", "discrete")]
    if not fixed:
        return [{}]
    names = [f.name for f in fixed]
    combinations = [dict(zip(names, values)) for values in itertools.product(*[f.levels for f in fixed])]
    if len(combinations) > _MAX_COMBINATIONS:
        raise StudyError(
            f"These categorical and discrete factors have {len(combinations):,} combinations, too many to search. "
            "Fix some of them at a chosen level, using what the analysis shows."
        )
    return combinations


def _search_space(study: Study):
    """Set up the search variables.

    For a mixture, all components but the last are searched and the last is whatever
    remains, so every candidate blend adds up to the total automatically.
    """
    continuous = [f for f in study.process_factors if f.kind == "continuous"]
    components = study.mixture_factors
    bounds = [(float(f.low), float(f.high)) for f in continuous]
    bounds += [(float(f.low), float(f.high)) for f in components[:-1]]
    total = float(study.mixture_total)

    def decode(x: np.ndarray, fixed: dict) -> pd.DataFrame:
        data = {f.name: x[:, index] for index, f in enumerate(continuous)}
        offset = len(continuous)
        if components:
            used = np.zeros(len(x))
            for index, factor in enumerate(components[:-1]):
                data[factor.name] = x[:, offset + index]
                used = used + x[:, offset + index]
            data[components[-1].name] = total - used
        for name, value in fixed.items():
            data[name] = np.full(len(x), value, dtype=object if isinstance(value, str) else float)
        return pd.DataFrame(data)[study.factor_names]

    return bounds, decode


def _objective(study, fits, limits, decode, fixed, x_matrix):
    """Score candidates, where lower is better. x_matrix has one row per candidate."""
    frame = decode(x_matrix, fixed)
    score, shortfall, _ = overall_desirability(fits, limits, frame)
    leverage = np.mean([fit.leverage(frame) for fit in fits], axis=0)
    return -score + shortfall + _TIE_BREAK * leverage + 10.0 * study.violation(frame)


def _describe_solutions(study, fits, limits, frame: pd.DataFrame) -> pd.DataFrame:
    score, _, each = overall_desirability(fits, limits, frame)
    table = frame[study.factor_names].copy()
    table.insert(0, "desirability", score)
    for fit in fits:
        prediction = fit.predict(frame)
        name = fit.response.name
        table[f"{name}_predicted"] = prediction["predicted"]
        table[f"{name}_pi_low"] = prediction["pi_low"]
        table[f"{name}_pi_high"] = prediction["pi_high"]
        table[f"{name}_desirability"] = each[name]
    # Above the largest leverage among the runs, a prediction is really an extrapolation.
    supported = np.ones(len(frame), dtype=bool)
    for fit in fits:
        design_leverage = np.sum((fit.matrix.values @ fit.xtx_inverse) * fit.matrix.values, axis=1).max()
        supported &= fit.leverage(frame) <= design_leverage + 1e-9
    table["well_supported"] = supported
    return table


def find_optimum(
    study: Study,
    fits: list[ResponseFit],
    solutions: int = 5,
    starts: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Search for the best settings, returning the distinct solutions found, best first."""
    active = [fit for fit in fits if fit.response.goal != "none"]
    if not active:
        raise StudyError(
            "No response has a goal, so there's nothing to optimise. Give at least one response a goal: "
            "maximise, minimise, target or in_range."
        )
    limits = [desirability_limits(fit) for fit in active]
    bounds, decode = _search_space(study)
    seed = study.seed if seed is None else seed
    generator = np.random.default_rng(seed)

    found = []
    combinations = _fixed_combinations(study)
    # Searching many combinations from several starts adds a lot of time for little gain.
    starts = starts if len(combinations) <= 4 else 1
    for fixed in combinations:
        if not bounds:
            found.append(decode(np.zeros((1, 0)), fixed))
            continue

        def vectorised(x_matrix):  # scipy passes one candidate per column
            return _objective(study, active, limits, decode, fixed, x_matrix.T)

        def single(x):
            return float(_objective(study, active, limits, decode, fixed, x[None, :])[0])

        for _ in range(starts):
            result = differential_evolution(
                vectorised,
                bounds,
                vectorized=True,
                updating="deferred",
                polish=False,
                maxiter=300,
                tol=1e-7,
                popsize=max(15, 4 * len(bounds)),
                rng=np.random.default_rng(generator.integers(2**31)),
            )
            refined = minimize(
                single, result.x, method="Nelder-Mead", bounds=bounds,
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9},
            )  # fmt: skip
            best = refined.x if refined.fun <= result.fun else result.x
            found.append(decode(best[None, :], fixed))

    frame = pd.concat(found, ignore_index=True)
    frame = frame[study.is_feasible(frame, tolerance=1e-4)].reset_index(drop=True)
    if frame.empty:
        raise StudyError("The search found no settings that satisfy the constraints.")
    table = _describe_solutions(study, active, limits, frame)
    table = table.sort_values("desirability", ascending=False, kind="stable").reset_index(drop=True)
    return _distinct_solutions(study, table, solutions)


def _distinct_solutions(study: Study, table: pd.DataFrame, solutions: int) -> pd.DataFrame:
    """Drop solutions that are effectively the same point, keeping the best of each."""
    coded = study.coded(table)
    numeric = np.column_stack(list(coded.values())) if coded else np.zeros((len(table), 0))
    categorical = table[study.categorical_names].astype(str).agg("|".join, axis=1) if study.categorical_names else None
    keep: list[int] = []
    for row in range(len(table)):
        duplicate = False
        for other in keep:
            same_category = categorical is None or categorical.iloc[row] == categorical.iloc[other]
            if same_category and np.linalg.norm(numeric[row] - numeric[other]) < 0.05:
                duplicate = True
                break
        if not duplicate:
            keep.append(row)
        if len(keep) == solutions:
            break
    return table.iloc[keep].reset_index(drop=True)
