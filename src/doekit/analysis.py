"""Fit a model to each response, test it, and say in plain language how far to trust it.

Regression is done by statsmodels. This module adds what DOE needs on top of that:
lack-of-fit testing from replicates, predicted R-squared, term-by-term ANOVA that also
handles mixture models, hierarchical model reduction, and Lenth's method for designs
with no degrees of freedom left over.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson

from .model import ModelMatrix, Term, find_aliases, model_matrix, model_terms, term_label
from .study import Response, Study, StudyError

_FORWARD = {"none": lambda y: y, "log": np.log, "sqrt": np.sqrt}
_INVERSE = {"none": lambda y: y, "log": np.exp, "sqrt": np.square}


@dataclass
class Check:
    """One line of the plain-language verdict: status is OK, CHECK or INFO."""

    status: str
    topic: str
    message: str


@dataclass
class ResponseFit:
    """A fitted model for one response, with everything needed to judge and use it."""

    study: Study
    response: Response
    terms: list[Term]
    frame: pd.DataFrame
    matrix: ModelMatrix
    observed: np.ndarray
    y: np.ndarray
    result: object
    total_runs: int
    removed_terms: list[str] = field(default_factory=list)

    @property
    def residual_df(self) -> int:
        return int(round(self.result.df_resid))

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self.result.params)

    @property
    def fitted(self) -> np.ndarray:
        return np.asarray(self.result.fittedvalues)

    @property
    def residuals(self) -> np.ndarray:
        return np.asarray(self.result.resid)

    @property
    def sse(self) -> float:
        return float(self.residuals @ self.residuals)

    @property
    def xtx_inverse(self) -> np.ndarray:
        return np.asarray(self.result.normalized_cov_params)

    def matrix_for(self, frame: pd.DataFrame) -> np.ndarray:
        """Build the model matrix for new settings, with block effects averaged out."""
        new = model_matrix(self.study, frame, self.terms, include_blocks=False)
        values = np.zeros((len(frame), len(self.matrix.names)))
        positions = [self.matrix.names.index(name) for name in new.names]
        values[:, positions] = new.values
        return values

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict quickly, in the response's original units, with no intervals."""
        return _INVERSE[self.response.transform](self.matrix_for(frame) @ self.params)

    def leverage(self, frame: pd.DataFrame) -> np.ndarray:
        """How far settings are from the data, relative to where the model is well supported."""
        X = self.matrix_for(frame)
        return np.sum((X @ self.xtx_inverse) * X, axis=1)

    def predict(self, frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """Predict with a confidence interval (for the average) and a prediction
        interval (for one future run), in the response's original units."""
        X = self.matrix_for(frame)
        mean = X @ self.params
        inverse = _INVERSE[self.response.transform]
        out = pd.DataFrame({"predicted": inverse(mean)}, index=frame.index)
        df = self.residual_df
        if df <= 0:
            for column in ("ci_low", "ci_high", "pi_low", "pi_high"):
                out[column] = np.nan
            return out
        mse = self.sse / df
        leverage = np.sum((X @ self.xtx_inverse) * X, axis=1)
        t = stats.t.ppf(1 - alpha / 2, df)
        ci = t * np.sqrt(mse * leverage)
        pi = t * np.sqrt(mse * (1 + leverage))
        out["ci_low"], out["ci_high"] = inverse(mean - ci), inverse(mean + ci)
        out["pi_low"], out["pi_high"] = inverse(mean - pi), inverse(mean + pi)
        return out


def _solve_sse(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    return float(residual @ residual), int(rank)


def _transform(response: Response, values: np.ndarray) -> np.ndarray:
    if response.transform == "log" and (values <= 0).any():
        raise StudyError(f"Response '{response.name}' has zero or negative values, so it can't use a log transform.")
    if response.transform == "sqrt" and (values < 0).any():
        raise StudyError(f"Response '{response.name}' has negative values, so it can't use a sqrt transform.")
    return _FORWARD[response.transform](values)


def _fit_terms(
    study: Study,
    frame: pd.DataFrame,
    response: Response,
    terms: list[Term],
    total_runs: int,
    removed: list[str] | None = None,
) -> ResponseFit:
    matrix = model_matrix(study, frame, terms)
    observed = frame[response.name].to_numpy(dtype=float)
    y = _transform(response, observed)
    if matrix.rank < matrix.values.shape[1]:
        aliases = find_aliases(matrix)
        message = (
            f"The model for '{response.name}' can't be estimated: it has {matrix.values.shape[1]} "
            f"coefficients, but the {len(frame)} completed runs support only {matrix.rank}."
        )
        if aliases:
            message += " These can't be told apart: " + "; ".join(f"{a} = {b}" for a, b in aliases[:8]) + "."
        message += " Use a simpler model in the config, or complete more runs."
        raise StudyError(message)
    result = sm.OLS(y, matrix.frame()).fit()
    return ResponseFit(study, response, terms, frame, matrix, observed, y, result, total_runs, removed or [])


def fit_metrics(fit: ResponseFit) -> dict:
    """Calculate R-squared and related measures of fit.

    R-squared is calculated here rather than read from statsmodels, so that it always
    uses the corrected total, including for Scheffe models with no intercept column.
    """
    y = fit.y
    runs = len(y)
    parameters = fit.matrix.rank
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = fit.sse
    df = runs - parameters
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    adjusted = 1 - (sse / df) / (sst / (runs - 1)) if df > 0 and sst > 0 else float("nan")
    leverage = np.sum((fit.matrix.values @ fit.xtx_inverse) * fit.matrix.values, axis=1)
    if df > 0 and (leverage < 1 - 1e-8).all():
        press = float(np.sum((fit.residuals / (1 - leverage)) ** 2))
        predicted = 1 - press / sst if sst > 0 else float("nan")
    else:
        press = predicted = float("nan")
    rmse = np.sqrt(sse / df) if df > 0 else float("nan")
    return {
        "runs": runs,
        "parameters": parameters,
        "residual_df": df,
        "r2": r2,
        "adj_r2": adjusted,
        "pred_r2": predicted,
        "press": press,
        "rmse": rmse,
        "cv_percent": 100 * rmse / abs(y.mean()) if df > 0 and y.mean() != 0 else float("nan"),
    }


def _partial_test(X: np.ndarray, y: np.ndarray, drop: list[int], add_constant: bool, sse_full, rank_full):
    """Test whether a group of columns improves the model, given all the others."""
    keep = [column for column in range(X.shape[1]) if column not in drop]
    reduced = X[:, keep]
    if add_constant:
        reduced = np.column_stack([np.ones(len(y)), reduced])
    if reduced.shape[1] == 0:
        reduced = np.zeros((len(y), 1))
    sse_reduced, rank_reduced = _solve_sse(reduced, y)
    df_term = rank_full - rank_reduced
    df_error = len(y) - rank_full
    ss = max(sse_reduced - sse_full, 0.0)
    if df_term <= 0 or df_error <= 0 or sse_full <= 0:
        return ss, df_term, float("nan"), float("nan")
    f_value = (ss / df_term) / (sse_full / df_error)
    return ss, df_term, f_value, float(stats.f.sf(f_value, df_term, df_error))


def lack_of_fit(fit: ResponseFit) -> dict | None:
    """Test whether the model misses a real pattern, by comparing it to the replicate noise.

    Pure error is measured by fitting a separate mean to every distinct setting (plus
    any block effects). No model can do better than that, so whatever it explains
    beyond the fitted model is lack of fit.
    """
    study, frame = fit.study, fit.frame
    labels = pd.DataFrame(index=frame.index)
    for name in study.factor_names:
        column = frame[name]
        labels[name] = column.round(8) if pd.api.types.is_float_dtype(column) else column
    cells = labels.astype(str).agg("|".join, axis=1)
    dummies = pd.get_dummies(cells, dtype=float).to_numpy()
    block_columns = [i for i, group in enumerate(fit.matrix.groups) if group == "block"]
    saturated = np.column_stack([dummies, fit.matrix.values[:, block_columns]]) if block_columns else dummies
    sse_pure, rank_pure = _solve_sse(saturated, fit.y)
    df_pure = len(fit.y) - rank_pure
    df_lof = fit.residual_df - df_pure
    if df_pure <= 0 or df_lof <= 0:
        return None
    ss_lof = max(fit.sse - sse_pure, 0.0)
    if sse_pure <= 0:
        return None
    f_value = (ss_lof / df_lof) / (sse_pure / df_pure)
    return {
        "ss_lack_of_fit": ss_lof,
        "df_lack_of_fit": df_lof,
        "ss_pure_error": sse_pure,
        "df_pure_error": df_pure,
        "f": f_value,
        "p": float(stats.f.sf(f_value, df_lof, df_pure)),
    }


def anova_table(fit: ResponseFit) -> pd.DataFrame:
    """Build the analysis of variance table.

    Each term's sum of squares is adjusted for all the others (Type III). In a design
    that isn't perfectly orthogonal, these won't add up exactly to the model total.
    """
    X, y = fit.matrix.values, fit.y
    groups = fit.matrix.groups
    sse_full, rank_full = fit.sse, fit.matrix.rank
    sst = float(np.sum((y - y.mean()) ** 2))
    rows = []
    mixture = set(fit.study.mixture_names)
    block_positions = [i for i, group in enumerate(groups) if group == "block"]

    # Model row: everything except the mean and any block effects.
    base = [np.ones(len(y))] + [X[:, i] for i in block_positions]
    sse_base, rank_base = _solve_sse(np.column_stack(base), y)
    df_model = rank_full - rank_base
    df_error = len(y) - rank_full
    ss_model = max(sse_base - sse_full, 0.0)
    if df_model > 0 and df_error > 0 and sse_full > 0:
        f_model = (ss_model / df_model) / (sse_full / df_error)
        p_model = float(stats.f.sf(f_model, df_model, df_error))
    else:
        f_model = p_model = float("nan")
    rows.append(("Model", ss_model, df_model, f_model, p_model))

    if block_positions:
        ss, df, f_value, p_value = _partial_test(X, y, block_positions, False, sse_full, rank_full)
        rows.append(("block", ss, df, f_value, p_value))

    if mixture:
        # Testing the blending terms together asks whether composition matters at all.
        positions = [i for i, group in enumerate(groups) if group in mixture]
        ss, df, f_value, p_value = _partial_test(X, y, positions, True, sse_full, rank_full)
        rows.append(("Linear mixture", ss, df, f_value, p_value))

    seen = []
    for group in groups:
        if group not in seen and group not in mixture and group not in ("Intercept", "block"):
            seen.append(group)
    for group in seen:
        positions = [i for i, name in enumerate(groups) if name == group]
        ss, df, f_value, p_value = _partial_test(X, y, positions, False, sse_full, rank_full)
        rows.append((group, ss, df, f_value, p_value))

    rows.append(("Residual", sse_full, df_error, float("nan"), float("nan")))
    lof = lack_of_fit(fit)
    if lof:
        rows.append(("  Lack of fit", lof["ss_lack_of_fit"], lof["df_lack_of_fit"], lof["f"], lof["p"]))
        rows.append(("  Pure error", lof["ss_pure_error"], lof["df_pure_error"], float("nan"), float("nan")))
    rows.append(("Total (corrected)", sst, len(y) - 1, float("nan"), float("nan")))

    table = pd.DataFrame(rows, columns=["source", "sum_sq", "df", "F", "p"])
    table["mean_sq"] = table["sum_sq"] / table["df"].replace(0, np.nan)
    return table[["source", "sum_sq", "df", "mean_sq", "F", "p"]]


def coefficient_table(fit: ResponseFit) -> pd.DataFrame:
    """List the coefficients in coded units, so their sizes can be compared directly.

    For a process factor, the coefficient is half the change going from low to high.
    """
    result = fit.result
    table = pd.DataFrame({"term": fit.matrix.names, "group": fit.matrix.groups, "coef": fit.params})
    if fit.residual_df <= 0:
        for column in ("std_err", "t", "p", "ci_low", "ci_high", "vif"):
            table[column] = np.nan
        return table
    table["std_err"] = np.asarray(result.bse)
    table["t"] = np.asarray(result.tvalues)
    table["p"] = np.asarray(result.pvalues)
    interval = np.asarray(result.conf_int())
    table["ci_low"], table["ci_high"] = interval[:, 0], interval[:, 1]
    X = fit.matrix.values
    spread = np.sum((X - X.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["vif"] = np.where(spread > 1e-12, np.diag(fit.xtx_inverse) * spread, np.nan)
    # Whether a blending coefficient differs from zero isn't a useful question.
    linear = table["term"].isin(fit.study.mixture_names)
    table.loc[linear, ["t", "p", "vif"]] = np.nan
    return table


def lenth_analysis(coefficients: pd.DataFrame, alpha: float = 0.05) -> dict | None:
    """Judge effects when no degrees of freedom remain (Lenth, 1989).

    Most effects in a screening experiment are small, so the small ones can stand in as
    a measure of noise. PSE is that noise estimate, and effects above the margin of
    error stand out from it.
    """
    table = coefficients[coefficients["term"] != "Intercept"]
    effects = 2 * table["coef"].to_numpy()
    if len(effects) < 3:
        return None
    magnitude = np.abs(effects)
    s0 = 1.5 * np.median(magnitude)
    trimmed = magnitude[magnitude < 2.5 * s0]
    pse = 1.5 * np.median(trimmed) if len(trimmed) else s0
    df = len(effects) / 3
    margin = float(stats.t.ppf(1 - alpha / 2, df) * pse)
    active = [name for name, value in zip(table["term"], magnitude) if value > margin]
    return {"pse": float(pse), "margin_of_error": margin, "df": df, "active": active, "effects": dict(zip(table["term"], effects))}


def run_diagnostics(fit: ResponseFit) -> pd.DataFrame:
    """Build the run-by-run table of residuals and influence."""
    frame = fit.frame
    table = pd.DataFrame({"run": frame["run"].to_numpy() if "run" in frame else np.arange(1, len(frame) + 1)})
    table["observed"] = fit.observed
    table["predicted"] = _INVERSE[fit.response.transform](fit.fitted)
    table["residual"] = fit.residuals
    leverage = np.sum((fit.matrix.values @ fit.xtx_inverse) * fit.matrix.values, axis=1)
    table["leverage"] = leverage
    if fit.residual_df > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            influence = OLSInfluence(fit.result)
            table["studentised_residual"] = np.asarray(influence.resid_studentized_external)
            table["cooks_distance"] = np.asarray(influence.cooks_distance[0])
    else:
        table["studentised_residual"] = np.nan
        table["cooks_distance"] = np.nan
    return table


def suggest_transform(fit: ResponseFit) -> dict | None:
    """Look for a helpful transformation using the Box-Cox profile likelihood."""
    y = fit.observed
    if (y <= 0).any() or fit.residual_df < 3:
        return None
    X = fit.matrix.values
    geometric_mean = np.exp(np.mean(np.log(y)))
    grid = np.round(np.arange(-2, 2.01, 0.1), 2)
    likelihood = []
    for value in grid:
        if abs(value) < 1e-9:
            scaled = geometric_mean * np.log(y)
        else:
            scaled = (y**value - 1) / (value * geometric_mean ** (value - 1))
        sse, _ = _solve_sse(X, scaled)
        likelihood.append(-len(y) / 2 * np.log(max(sse, 1e-300) / len(y)))
    likelihood = np.array(likelihood)
    best = float(grid[np.argmax(likelihood)])
    inside = grid[likelihood >= likelihood.max() - stats.chi2.ppf(0.95, 1) / 2]
    contains = lambda value: inside.min() - 1e-9 <= value <= inside.max() + 1e-9  # noqa: E731
    if contains(1.0):
        suggestion = "none"
    elif contains(0.0):
        suggestion = "log"
    elif contains(0.5):
        suggestion = "sqrt"
    else:
        suggestion = f"power {best:g}"
    return {"best_lambda": best, "low": float(inside.min()), "high": float(inside.max()), "suggestion": suggestion}


def curvature_test(fit: ResponseFit) -> dict | None:
    """Compare centre points with the factorial runs in a two-level design.

    If the factors act in straight lines, the average at the centre should equal the
    average at the corners. A clear difference shows curvature, a sign that a response
    surface design is the right next step.
    """
    study = fit.study
    names = study.numeric_process_names
    if study.mixture_factors or not names or any(len(term) == 2 and term[0] == term[1] for term in fit.terms):
        return None
    coded = study.coded(fit.frame)
    values = np.column_stack([coded[name] for name in names])
    center = np.all(np.abs(values) < 1e-6, axis=1)
    corners = np.all(np.abs(np.abs(values) - 1) < 1e-6, axis=1)
    centers, factorials = fit.y[center], fit.y[corners]
    if len(centers) < 2 or len(factorials) < 2 or (center | corners).sum() != len(fit.y):
        return None
    deviation = centers.std(ddof=1)
    if deviation <= 0:
        return None
    difference = float(factorials.mean() - centers.mean())
    t_value = difference / (deviation * np.sqrt(1 / len(factorials) + 1 / len(centers)))
    return {
        "factorial_mean": float(factorials.mean()),
        "center_mean": float(centers.mean()),
        "difference": difference,
        "p": float(2 * stats.t.sf(abs(t_value), len(centers) - 1)),
    }


def interpret_fit(fit: ResponseFit) -> list[Check]:
    """Turn the statistics into plain-language checks a non-statistician can act on."""
    checks: list[Check] = []
    metrics = fit_metrics(fit)
    table = anova_table(fit)
    runs = len(fit.y)

    skipped = fit.total_runs - runs
    if skipped:
        checks.append(Check("INFO", "Data", f"{skipped} of {fit.total_runs} runs have no result yet and were left out."))

    if fit.residual_df <= 0:
        checks.append(
            Check(
                "CHECK", "Degrees of freedom",
                "The model uses every run, so nothing is left to test it with. Only the sizes of the "
                "effects can be judged (see the half-normal plot).",
            )  # fmt: skip
        )
        return checks

    model_row = table[table["source"] == "Model"].iloc[0]
    if np.isfinite(model_row["p"]):
        if model_row["p"] < 0.05:
            checks.append(Check("OK", "Overall model", f"The model explains more than noise alone would (p = {model_row['p']:.3g})."))
        else:
            checks.append(
                Check(
                    "CHECK", "Overall model",
                    f"The model isn't clearly better than noise (p = {model_row['p']:.3g}). The factors may have "
                    "little effect in these ranges, or the noise may be hiding them.",
                )  # fmt: skip
            )

    predicted, adjusted = metrics["pred_r2"], metrics["adj_r2"]
    if np.isfinite(predicted):
        if predicted < 0:
            checks.append(
                Check(
                    "CHECK", "Predictive power",
                    f"Predicted R2 is negative ({predicted:.2f}), so the model predicts new runs worse than "
                    "just using the average. Don't use it for predictions until it's improved. Too many terms "
                    "for the data is a common cause, so try model reduction.",
                )  # fmt: skip
            )
        elif adjusted - predicted > 0.2:
            checks.append(
                Check(
                    "CHECK", "Predictive power",
                    f"Predicted R2 ({predicted:.2f}) is well below adjusted R2 ({adjusted:.2f}). The model may be "
                    "fitting noise, or one run may be very influential. Try model reduction, and check the "
                    "influence plot.",
                )  # fmt: skip
            )
        else:
            checks.append(Check("OK", "Predictive power", f"Predicted R2 ({predicted:.2f}) is close to adjusted R2 ({adjusted:.2f})."))

    lof = lack_of_fit(fit)
    if lof is None:
        checks.append(Check("INFO", "Lack of fit", "Can't be tested, because it needs both replicated runs and spare distinct runs."))
    elif lof["p"] < 0.05:
        checks.append(
            Check(
                "CHECK", "Lack of fit",
                f"The model misses a real pattern (p = {lof['p']:.3g}): its errors are bigger than replicate noise "
                "can explain. Try a transform or a higher-order model, and be careful with predictions.",
            )  # fmt: skip
        )
    else:
        checks.append(Check("OK", "Lack of fit", f"No evidence the model is missing a pattern (p = {lof['p']:.3g})."))

    if runs >= 8 and fit.residual_df >= 3 and np.ptp(fit.residuals) > 0:
        _, normal_p = stats.shapiro(fit.residuals)
        if normal_p < 0.01:
            checks.append(
                Check(
                    "CHECK", "Residual normality",
                    f"The residuals don't look normally distributed (Shapiro-Wilk p = {normal_p:.3g}). "
                    "Check for outliers, or try a transform.",
                )  # fmt: skip
            )
        else:
            checks.append(Check("OK", "Residual normality", f"Residuals look acceptably normal (Shapiro-Wilk p = {normal_p:.3g})."))

    if runs >= 8 and np.ptp(fit.residuals) > 0 and np.ptp(fit.fitted) > 0:
        rho, spread_p = stats.spearmanr(fit.fitted, np.abs(fit.residuals))
        if spread_p < 0.05 and rho > 0:
            checks.append(
                Check(
                    "CHECK", "Constant variance",
                    "The scatter grows as the response gets bigger (a funnel shape in the residual plot). "
                    "A log or sqrt transform often fixes this.",
                )  # fmt: skip
            )

    if fit.response.transform == "none":
        suggestion = suggest_transform(fit)
        if suggestion and suggestion["suggestion"] != "none":
            checks.append(
                Check(
                    "CHECK", "Transform",
                    f"A Box-Cox analysis suggests transforming this response ({suggestion['suggestion']}). "
                    "Try setting 'transform' for it in the config and compare the results.",
                )  # fmt: skip
            )

    if "run" in fit.frame and fit.residual_df >= 3:
        order = np.argsort(fit.frame["run"].to_numpy())
        statistic = durbin_watson(fit.residuals[order])
        if statistic < 1.0 or statistic > 3.0:
            checks.append(
                Check(
                    "CHECK", "Run order",
                    f"Residuals seem to follow the run order (Durbin-Watson = {statistic:.2f}), so something may "
                    "have drifted during the experiment, such as a material ageing or a nozzle clogging.",
                )  # fmt: skip
            )

    diagnostics = run_diagnostics(fit)
    outliers = diagnostics[diagnostics["studentised_residual"].abs() > 3.5]
    if len(outliers):
        checks.append(
            Check(
                "CHECK", "Outliers",
                f"Run(s) {outliers['run'].tolist()} don't fit the pattern of the others. Check the lab notes for that run. "
                "Only remove a run if you can name a real cause.",
            )  # fmt: skip
        )
    influential = diagnostics[diagnostics["cooks_distance"] > 1]
    if len(influential):
        checks.append(
            Check(
                "CHECK", "Influential runs",
                f"Run(s) {influential['run'].tolist()} strongly affect the model. Verify those measurements carefully.",
            )  # fmt: skip
        )

    curvature = curvature_test(fit)
    if curvature:
        if curvature["p"] < 0.05:
            checks.append(
                Check(
                    "CHECK", "Curvature",
                    f"Centre points differ from the corners (p = {curvature['p']:.3g}), so the response curves. "
                    "A straight-line model can't find an optimum here, so use a response surface design next.",
                )  # fmt: skip
            )
        else:
            checks.append(Check("OK", "Curvature", f"No clear curvature across the region (p = {curvature['p']:.3g})."))
    return checks


def _removable_terms(study: Study, terms: list[Term]) -> list[Term]:
    """List the terms that can be dropped while keeping the model hierarchical.

    A main effect stays while an interaction or curvature term uses it, and the
    component terms of a mixture model are never dropped.
    """
    mixture = set(study.mixture_names)
    removable = []
    for term in terms:
        if len(term) == 1 and term[0] in mixture:
            continue
        counts = Counter(term)
        needed = any(
            other != term and all(Counter(other)[name] >= power for name, power in counts.items())
            for other in terms
        )
        if needed:
            continue
        # With mixture crossing, the component x process terms carry the process main effect.
        if len(term) == 2 and term[0] in mixture and term[1] not in mixture:
            process = term[1]
            siblings = [t for t in terms if len(t) == 2 and t[0] in mixture and t[1] == process and t != term]
            higher = [t for t in terms if process in t and not (len(t) == 2 and t[0] in mixture)]
            if higher and not siblings:
                continue
        removable.append(term)
    return removable


def reduce_model(
    study: Study,
    frame: pd.DataFrame,
    response: Response,
    terms: list[Term],
    total_runs: int,
    alpha: float = 0.10,
) -> ResponseFit:
    """Remove the least significant term, refit, and repeat until all are significant."""
    terms = list(terms)
    removed: list[str] = []
    fit = _fit_terms(study, frame, response, terms, total_runs)
    while fit.residual_df > 0:
        table = anova_table(fit).set_index("source")
        options = [
            (float(table.loc[term_label(term), "p"]), term)
            for term in _removable_terms(study, terms)
            if term_label(term) in table.index and np.isfinite(table.loc[term_label(term), "p"])
        ]
        if not options:
            break
        worst_p, worst = max(options, key=lambda item: item[0])
        if worst_p <= alpha:
            break
        terms.remove(worst)
        removed.append(term_label(worst))
        fit = _fit_terms(study, frame, response, terms, total_runs)
    fit.removed_terms = removed
    return fit


def fit_response(
    study: Study,
    frame: pd.DataFrame,
    response: Response,
    terms: list[Term] | None = None,
) -> ResponseFit:
    """Fit one response using the runs that have a result, reducing the model if asked."""
    spec = study.model if isinstance(study.model, dict) else {}
    complete = frame[frame[response.name].notna()].reset_index(drop=True)
    if complete.empty:
        raise StudyError(f"Response '{response.name}' has no results yet.")
    terms = terms if terms is not None else model_terms(study, complete)
    if spec.get("reduce", False):
        return reduce_model(study, complete, response, terms, len(frame), float(spec.get("alpha", 0.10)))
    return _fit_terms(study, complete, response, terms, len(frame))


def fit_all(study: Study, frame: pd.DataFrame) -> tuple[list[ResponseFit], list[str]]:
    """Fit every response that has results. Returns the fits, plus notes on any skipped."""
    fits, skipped = [], []
    for response in study.responses:
        if frame[response.name].notna().sum() == 0:
            skipped.append(f"'{response.name}' has no results yet, so it was skipped.")
            continue
        fits.append(fit_response(study, frame, response))
    return fits, skipped
