"""Judge a design BEFORE any material is spent on it.

Everything here depends only on where the runs are placed, not on any measurements.
That's what makes it possible to compare designs at the planning stage, and find out
whether an experiment is big enough while it can still be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .designs.optimal import candidate_points
from .model import Term, find_aliases, has_intercept, model_matrix, model_terms
from .study import Study


@dataclass
class DesignQuality:
    runs: int
    terms: int
    distinct_runs: int
    residual_df: int
    pure_error_df: int
    lack_of_fit_df: int
    estimable: bool
    aliases: list[tuple[str, str]]
    d_efficiency: float
    mean_prediction_se: float
    max_prediction_se: float
    prediction_se: np.ndarray
    term_table: pd.DataFrame
    signal_to_noise: float
    warnings: list[str] = field(default_factory=list)

    @property
    def lowest_power(self) -> float:
        power = self.term_table["power"].dropna()
        return float(power.min()) if len(power) else float("nan")

    def summary(self) -> dict:
        return {
            "runs": self.runs,
            "terms": self.terms,
            "spare runs": self.residual_df,
            "replicates": self.pure_error_df,
            "typical prediction error": round(self.mean_prediction_se, 2),
            "worst prediction error": round(self.max_prediction_se, 2),
            "lowest power": round(self.lowest_power, 2) if np.isfinite(self.lowest_power) else None,
        }


def term_power(standard_error: np.ndarray, residual_df: int, signal_to_noise: float, alpha: float = 0.05):
    """Calculate the chance of detecting an effect, for each coefficient.

    ``signal_to_noise`` is the change you want to detect, moving a factor from low to
    high, as a multiple of the noise standard deviation. In coded units that change
    corresponds to a coefficient of signal_to_noise / 2.
    """
    if residual_df <= 0:
        return np.full(len(standard_error), np.nan)
    critical = stats.t.ppf(1 - alpha / 2, residual_df)
    noncentrality = (signal_to_noise / 2) / standard_error
    return 1 - stats.nct.cdf(critical, residual_df, noncentrality) + stats.nct.cdf(-critical, residual_df, noncentrality)


def _distinct_settings(study: Study, frame: pd.DataFrame) -> pd.Series:
    """Label each run by its factor settings, so that replicates share a label."""
    labels = frame[study.factor_names].copy()
    for column in study.factor_names:
        if pd.api.types.is_float_dtype(labels[column]):
            labels[column] = labels[column].round(8)
    return labels.astype(str).agg("|".join, axis=1)


def _pure_error_df(frame: pd.DataFrame, settings: pd.Series, matrix) -> int:
    """Count the degrees of freedom that replicates give for measuring noise.

    This is what's left after fitting a separate mean for each distinct setting, plus any
    block effects. Replicates that land in different blocks still count, as they do in
    the lack-of-fit test.
    """
    cells = pd.get_dummies(settings, dtype=float).to_numpy()
    blocks = [i for i, group in enumerate(matrix.groups) if group == "block"]
    saturated = np.column_stack([cells, matrix.values[:, blocks]]) if blocks else cells
    return len(frame) - int(np.linalg.matrix_rank(saturated))


def evaluate_design(
    study: Study,
    frame: pd.DataFrame,
    terms: list[Term] | None = None,
    signal_to_noise: float = 2.0,
) -> DesignQuality:
    """Work out how much a design can tell you, in units of the (still unknown) noise."""
    terms = terms if terms is not None else model_terms(study, frame)
    matrix = model_matrix(study, frame, terms)
    X = matrix.values
    runs, size = X.shape
    rank = np.linalg.matrix_rank(X)
    estimable = rank == size
    settings = _distinct_settings(study, frame)
    distinct = int(settings.nunique())
    residual_df = runs - rank
    pure_error_df = _pure_error_df(frame, settings, matrix)
    lack_of_fit_df = max(residual_df - pure_error_df, 0)
    warnings: list[str] = []

    if not estimable:
        aliases = find_aliases(matrix)
        message = f"This design can't estimate the model: it has {size} coefficients but supports only {rank}."
        if aliases:
            pairs = "; ".join(f"{a} = {b}" for a, b in aliases[:8])
            message += f" These can't be told apart: {pairs}."
        message += " Use a simpler model, or a design with more runs."
        warnings.append(message)
        return DesignQuality(
            runs, size, distinct, residual_df, pure_error_df, lack_of_fit_df, False, aliases,
            float("nan"), float("nan"), float("nan"), np.array([]),
            pd.DataFrame(columns=["term", "standard_error", "vif", "power"]),
            signal_to_noise, warnings,
        )  # fmt: skip

    information = X.T @ X
    inverse = np.linalg.inv(information)
    standard_error = np.sqrt(np.diag(inverse))
    d_efficiency = 100 * np.exp(np.linalg.slogdet(information)[1] / size) / runs

    # How precise predictions are, across the whole allowed region.
    region = candidate_points(study, {"max_candidates": 5000}, np.random.default_rng(0))
    grid = model_matrix(study, region, terms, include_blocks=False)
    padded = np.zeros((len(region), size))
    positions = [matrix.names.index(name) for name in grid.names]
    padded[:, positions] = grid.values
    prediction_se = np.sqrt(np.sum((padded @ inverse) * padded, axis=1))

    # Variance inflation: 1 means a term is independent of the others, higher means entangled.
    spread = np.sum((X - X.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vif = np.where(spread > 1e-12, np.diag(inverse) * spread, np.nan)

    # Power is only worked out for terms made entirely of process factors. A coefficient on
    # a mixture term doesn't describe a low-to-high change: its effect is largest at one
    # particular blend and smaller elsewhere, so the same calculation would be misleading.
    # For the same reason, blend terms are left out of the warning about entangled terms,
    # since some entanglement is unavoidable when components must add up to a fixed total.
    mixture = set(study.mixture_names)
    uses_mixture = [bool(set(group.split(":")) & mixture) for group in matrix.groups]
    linear_mixture = [name in mixture for name in matrix.names]
    power = term_power(standard_error, residual_df, signal_to_noise)
    power = np.where(uses_mixture, np.nan, power)
    vif = np.where(linear_mixture, np.nan, vif)
    if has_intercept(study):
        power[0] = np.nan
        vif[0] = np.nan
    blend_only = np.array([bool(group.split(":")[0] in mixture and set(group.split(":")) <= mixture) for group in matrix.groups])

    table = pd.DataFrame(
        {"term": matrix.names, "standard_error": standard_error, "vif": vif, "power": power}
    )

    if residual_df == 0:
        warnings.append(
            "The model uses every run, so nothing is left to estimate noise or check the fit. "
            "Effects can only be judged by size (half-normal plot)."
        )
    elif residual_df < 3:
        warnings.append(f"Only {residual_df} run(s) are left after fitting the model, so tests will be weak.")
    if pure_error_df == 0:
        warnings.append(
            "No runs are replicated, so noise can't be measured independently of the model, "
            "and lack of fit can't be tested. Adding centre points or replicates would fix this."
        )
    elif lack_of_fit_df == 0 and residual_df > 0:
        warnings.append("No degrees of freedom are left to test lack of fit. Consider adding distinct runs.")
    finite_vif = table["vif"].where(~blend_only).dropna()
    if len(finite_vif) and finite_vif.max() > 10:
        worst = table.loc[finite_vif.idxmax(), "term"]
        warnings.append(
            f"Some terms are entangled with each other (highest VIF is {finite_vif.max():.1f}, for {worst}). "
            "Their separate effects will be hard to pin down."
        )
    power_values = table["power"].dropna()
    if len(power_values) and power_values.min() < 0.8:
        weak = int((power_values < 0.8).sum())
        warnings.append(
            f"{weak} of {len(power_values)} terms have under 80% power to detect a low-to-high change "
            f"of {signal_to_noise:g} times the noise. A real effect that size could easily be missed."
        )

    return DesignQuality(
        runs=runs,
        terms=size,
        distinct_runs=distinct,
        residual_df=residual_df,
        pure_error_df=pure_error_df,
        lack_of_fit_df=lack_of_fit_df,
        estimable=True,
        aliases=[],
        d_efficiency=float(d_efficiency),
        mean_prediction_se=float(prediction_se.mean()),
        max_prediction_se=float(prediction_se.max()),
        prediction_se=prediction_se,
        term_table=table,
        signal_to_noise=signal_to_noise,
        warnings=warnings,
    )
