"""Computer-generated optimal designs, the most flexible option.

You state the model you want to fit and how many runs you can afford. The algorithm
then picks the best set of runs from a list of candidates. This handles what classical
designs can't: mixtures combined with process factors, categorical factors, constraints
and any run budget.

    D-optimal   makes the coefficients as precise as possible. Best for learning which
                effects are real.
    I-optimal   makes predictions as precise as possible across the region. Best for
                finding optimum settings.

Runs are chosen by Fedorov exchange: start from a reasonable design, then keep making
the single swap (one run out, one candidate in) that improves the design the most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import model_matrix, model_terms
from ..study import Study, StudyError
from .mixture import mixture_candidates

_RIDGE = 1e-8  # keeps the arithmetic stable while a starting design is still singular
LACK_OF_FIT_RUNS = 4
DEFAULT_REPLICATE_RUNS = 4
_MAX_GRID = 100_000


def _process_candidates(study: Study, options: dict, rng=None) -> pd.DataFrame:
    terms = model_terms(study)
    highest_power = max((len(term) for term in terms), default=1)
    default = 5 if study.constraints or highest_power > 2 else 3
    count = int(options.get("candidate_levels", default))
    levels = []
    for factor in study.process_factors:
        if factor.kind in ("categorical", "discrete"):
            levels.append(list(factor.levels))
        else:
            levels.append(np.linspace(float(factor.low), float(factor.high), count).tolist())
    names = [f.name for f in study.process_factors]
    if np.prod([float(len(values)) for values in levels]) > _MAX_GRID:
        # Too many factors to list every combination, so draw a large random selection.
        rng = rng if rng is not None else np.random.default_rng(0)
        return pd.DataFrame({name: rng.choice(np.array(values, dtype=object), size=_MAX_GRID).tolist() for name, values in zip(names, levels)})
    return pd.MultiIndex.from_product(levels, names=names).to_frame(index=False)


def candidate_points(study: Study, options: dict, rng) -> pd.DataFrame:
    """List the allowed runs an optimal design can choose from."""
    parts = []
    if study.mixture_factors:
        parts.append(mixture_candidates(study))
    if study.process_factors:
        parts.append(_process_candidates(study, options, rng))
    if len(parts) == 2 and len(parts[0]) * len(parts[1]) > 2_000_000:
        # A full cross would use too much memory, so pair blends and settings at random.
        size = 200_000
        left = parts[0].iloc[rng.integers(len(parts[0]), size=size)].reset_index(drop=True)
        right = parts[1].iloc[rng.integers(len(parts[1]), size=size)].reset_index(drop=True)
        candidates = pd.concat([left, right], axis=1).drop_duplicates()
    else:
        candidates = parts[0] if len(parts) == 1 else parts[0].merge(parts[1], how="cross")
    candidates = candidates[study.is_feasible(candidates)].reset_index(drop=True)
    if candidates.empty:
        raise StudyError("No allowed runs exist. The constraints and limits rule out everything.")
    limit = int(options.get("max_candidates", 5000))
    if len(candidates) > limit:
        keep = np.sort(rng.choice(len(candidates), size=limit, replace=False))
        candidates = candidates.iloc[keep].reset_index(drop=True)
    return candidates[study.factor_names]


def _information(X: np.ndarray, fixed: np.ndarray | None) -> np.ndarray:
    """Add up the information from the chosen runs and any runs that are already fixed."""
    information = X.T @ X + _RIDGE * np.eye(X.shape[1])
    return information if fixed is None else information + fixed.T @ fixed


def _start(C: np.ndarray, runs: int, rng, fixed: np.ndarray | None = None) -> np.ndarray:
    """Build a starting design: some random runs, then the most informative ones."""
    size = C.shape[1]
    missing = size if fixed is None else size - int(np.linalg.matrix_rank(fixed))
    first = min(max(1, missing // 2), runs)
    chosen = list(rng.choice(len(C), size=first, replace=len(C) < first))
    inverse = np.linalg.inv(_information(C[chosen], fixed))
    while len(chosen) < runs:
        variance = np.sum((C @ inverse) * C, axis=1)
        best = int(np.argmax(variance + rng.random(len(C)) * 1e-9))
        chosen.append(best)
        x = inverse @ C[best]
        inverse = inverse - np.outer(x, x) / (1 + C[best] @ x)
    return np.array(chosen)


def _swap_gains(C: np.ndarray, chosen: np.ndarray, criterion: str, moments: np.ndarray | None, fixed: np.ndarray | None = None):
    """Calculate how much every possible swap would improve the design.

    Returns a matrix with one row per design run and one column per candidate. Using
    the matrix inversion lemma avoids refitting the design for each possible swap.
    """
    X = C[chosen]
    inverse = np.linalg.inv(_information(X, fixed))
    CM = C @ inverse
    XM = X @ inverse
    d_c = np.sum(CM * C, axis=1)
    d_x = np.sum(XM * X, axis=1)
    d_xc = XM @ C.T
    # The determinant is multiplied by this ratio if run x is swapped for candidate c.
    ratio = (1 + d_c)[None, :] * (1 - d_x)[:, None] + d_xc**2
    if criterion == "D":
        return ratio - 1
    G = inverse @ moments @ inverse
    g_c = np.sum((C @ G) * C, axis=1)
    XG = X @ G
    g_x = np.sum(XG * X, axis=1)
    g_xc = XG @ C.T
    numerator = (d_x - 1)[:, None] * g_c[None, :] - 2 * d_xc * g_xc + (1 + d_c)[None, :] * g_x[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = np.where(ratio > 1e-8, -numerator / ratio, -np.inf)
    return gains


def _exchange(C, chosen, criterion, moments, max_steps, fixed=None) -> np.ndarray:
    chosen = chosen.copy()
    for _ in range(max_steps):
        gains = _swap_gains(C, chosen, criterion, moments, fixed)
        i, j = np.unravel_index(np.argmax(gains), gains.shape)
        if gains[i, j] <= 1e-9:
            break
        chosen[i] = j
    return chosen


def design_score(C: np.ndarray, chosen: np.ndarray, criterion: str, moments: np.ndarray | None, fixed: np.ndarray | None = None) -> float:
    """Score a design so that lower is better, for either criterion."""
    information = _information(C[chosen], fixed)
    sign, log_det = np.linalg.slogdet(information)
    if criterion == "D":
        return -log_det if sign > 0 else np.inf
    return float(np.trace(np.linalg.solve(information, moments)))


def default_starts(candidates: int, runs: int, size: int) -> int:
    """Choose how many starting designs to try. Larger problems take longer per start, and
    many starts add little, so they get fewer."""
    work = candidates * runs * size
    return 20 if work < 1_000_000 else 8 if work < 5_000_000 else 3


def exchange_search(
    C: np.ndarray,
    runs: int,
    criterion: str,
    rng,
    starts: int = 20,
    fixed: np.ndarray | None = None,
    moments: np.ndarray | None = None,
) -> np.ndarray:
    """Run the exchange from several starting designs and keep the best result.

    ``fixed`` holds the model matrix of runs that can't change, such as experiments already
    done. They count towards the information, but the search only chooses the other runs.
    """
    if criterion not in ("D", "I"):
        raise ValueError("criterion must be 'D' or 'I'")
    moments = C.T @ C / len(C) if moments is None else moments
    max_steps = 25 * runs
    best, best_score = None, np.inf
    for _ in range(starts):
        chosen = _exchange(C, _start(C, runs, rng, fixed), "D", None, max_steps, fixed)
        if criterion == "I":
            chosen = _exchange(C, chosen, "I", moments, max_steps, fixed)
        score = design_score(C, chosen, criterion, moments, fixed)
        if score < best_score:
            best, best_score = chosen, score
    return best


def _optimal(study: Study, options: dict, rng, notes: list[str], criterion: str) -> pd.DataFrame:
    terms = model_terms(study)
    candidates = candidate_points(study, options, rng)
    C = model_matrix(study, candidates, terms, include_blocks=False).values
    size = C.shape[1]
    if np.linalg.matrix_rank(C) < size:
        raise StudyError(
            "The model has terms that can't be estimated from the allowed runs. For example, "
            "a factor with only 2 possible values can't have a squared term. Simplify the model."
        )

    # The design's own replicate_runs setting is applied afterwards, by finalise_design.
    replicate_runs = int(options.setdefault("replicate_runs", DEFAULT_REPLICATE_RUNS))
    default_runs = size + LACK_OF_FIT_RUNS + replicate_runs
    total = int(options.get("runs", default_runs))
    if "runs" not in options:
        notes.append(
            f"Using {total} runs: {size} to fit the {size} model terms, {LACK_OF_FIT_RUNS} extra "
            f"to check lack of fit, and {replicate_runs} replicates to measure noise."
        )
    unique_runs = total - replicate_runs
    if unique_runs < size:
        raise StudyError(
            f"This model needs at least {size} runs plus {replicate_runs} replicates "
            f"({size + replicate_runs} in total), but only {total} were requested. Add runs, "
            "or simplify the model."
        )
    if total < default_runs:
        notes.append(
            f"With {total} runs there is little left over to check the model. "
            f"{default_runs} would be more reliable."
        )
    starts = int(options.get("starts", default_starts(len(C), unique_runs, size)))
    chosen = exchange_search(C, unique_runs, criterion, rng, starts)
    if np.linalg.matrix_rank(C[chosen]) < size:
        raise StudyError(
            "Couldn't find runs that estimate every model term. Add runs or simplify the model."
        )
    return candidates.iloc[chosen].reset_index(drop=True)


def d_optimal(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    return _optimal(study, options, rng, notes, "D")


def i_optimal(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    return _optimal(study, options, rng, notes, "I")
