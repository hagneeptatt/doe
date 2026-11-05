"""Every figure, for the scripts and the web app alike.

Each function returns a matplotlib Figure, built without pyplot's global state, which
makes them safe for a web app where several people may be plotting at once.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib as mpl
import mpltern  # noqa: F401  (registers the ternary projection)
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from .analysis import ResponseFit, coefficient_table, lenth_analysis, run_diagnostics
from .designs import mixture as mixture_designs
from .evaluate import DesignQuality
from .model import Term, model_matrix
from .optimise import Limits, overall_desirability
from .study import Study

# Categorical colours are used in this fixed order, which keeps neighbouring colours
# distinguishable with colour-blindness. Magnitudes always use the one-hue blue ramp.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "doekit_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "svg.fonttype": "none",  # keep text editable in Illustrator or Inkscape
        "pdf.fonttype": 42,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained", facecolor="white")


def save_figure(figure: Figure, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path


def _dots(axis, x, y, color=CATEGORICAL[0], size=40, **kwargs):
    return axis.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=1, zorder=3, **kwargs)


def _hold_frame(study: Study, hold: dict, rows: int) -> pd.DataFrame:
    """Build rows with every factor held at one setting, ready for varying some of them."""
    data = {}
    for factor in study.factors:
        value = hold[factor.name]
        if factor.kind == "categorical":
            data[factor.name] = np.full(rows, value, dtype=object)
        else:
            data[factor.name] = np.full(rows, float(value))
    return pd.DataFrame(data)


def _format_value(value) -> str:
    return value if isinstance(value, str) else f"{float(value):.3g}"


def _hold_text(study: Study, hold: dict, varied: list[str]) -> str:
    parts = [f"{name} = {_format_value(hold[name])}" for name in study.factor_names if name not in varied]
    return "Held at: " + ", ".join(parts) if parts else ""


def _subplots(count: int, columns: int, width: float, height: float, share_y=False):
    columns = min(columns, count)
    rows = int(np.ceil(count / columns))
    figure = _figure(width * columns, height * rows)
    axes = figure.subplots(rows, columns, squeeze=False, sharey=share_y)
    flat = axes.ravel()
    for axis in flat[count:]:
        axis.set_visible(False)
    return figure, flat[:count]


# ---------------------------------------------------------------------- design stage
def _axis_values(study: Study, frame: pd.DataFrame, name: str):
    factor = study.factor(name)
    if factor.kind == "categorical":
        mapping = {level: index for index, level in enumerate(factor.levels)}
        return frame[name].map(mapping).to_numpy(dtype=float), list(factor.levels)
    return frame[name].to_numpy(dtype=float), None


def plot_design_points(study: Study, frame: pd.DataFrame) -> Figure | None:
    """Show where the runs sit, for each pair of process factors.

    A bigger dot means that combination appears more than once.
    """
    names = [factor.name for factor in study.process_factors]
    if len(names) < 2:
        return None
    pairs = list(itertools.combinations(names, 2))[:15]
    figure, axes = _subplots(len(pairs), 3, 3.1, 2.8)
    for axis, (a, b) in zip(axes, pairs):
        x, x_labels = _axis_values(study, frame, a)
        y, y_labels = _axis_values(study, frame, b)
        points, counts = np.unique(np.column_stack([x, y]).round(8), axis=0, return_counts=True)
        _dots(axis, points[:, 0], points[:, 1], size=28 + 18 * (counts - 1))
        axis.set_xlabel(study.factor(a).label)
        axis.set_ylabel(study.factor(b).label)
        if x_labels:
            axis.set_xticks(range(len(x_labels)), x_labels)
        if y_labels:
            axis.set_yticks(range(len(y_labels)), y_labels)
        axis.margins(0.12)
    figure.suptitle("Where the runs sit, one pair of factors at a time")
    return figure


def _ternary_axis(figure, position, study: Study, components: list[str], total: float):
    """Create a ternary axis zoomed to the allowed region for three components."""
    axis = figure.add_subplot(position, projection="ternary", ternary_sum=total)
    lower = [float(study.factor(name).low) for name in components]
    span = total - sum(lower)
    axis.set_ternary_lim(lower[0], lower[0] + span, lower[1], lower[1] + span, lower[2], lower[2] + span)
    axis.set_tlabel(study.factor(components[0]).label, color=INK_SECONDARY)
    axis.set_llabel(study.factor(components[1]).label, color=INK_SECONDARY)
    axis.set_rlabel(study.factor(components[2]).label, color=INK_SECONDARY)
    axis.grid(True, color=GRID, linewidth=0.6)
    return axis


def _selected_total(study: Study, components: list[str], hold: dict) -> float:
    others = [name for name in study.mixture_names if name not in components]
    return float(study.mixture_total) - sum(float(hold[name]) for name in others)


def plot_design_ternary(study: Study, frame: pd.DataFrame) -> Figure | None:
    """Show where the blends sit within the allowed composition region."""
    names = study.mixture_names
    if len(names) < 3:
        return None
    components = names[:3]
    figure = _figure(5.2, 4.8)
    if len(names) == 3:
        total = float(study.mixture_total)
        values = frame[components].to_numpy(dtype=float)
    else:
        # Rescale the blend onto the three shown, since the others vary between runs.
        total = float(study.mixture_total)
        values = frame[components].to_numpy(dtype=float)
        lower = np.array([float(study.factor(name).low) for name in components])
        free = total - lower.sum()
        share = (values - lower).clip(min=0)
        share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
        values = lower + share * free
    axis = _ternary_axis(figure, 111, study, components, total)
    points, counts = np.unique(values.round(6), axis=0, return_counts=True)
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=30 + 20 * (counts - 1), color=CATEGORICAL[0], edgecolor="white", linewidth=1, zorder=3,
    )  # fmt: skip
    figure.suptitle("Where the blends sit (bigger dot = repeated)")
    return figure


def plot_term_correlation(study: Study, frame: pd.DataFrame, terms: list[Term]) -> Figure:
    """Show how entangled the model terms are. Off-diagonal cells should be pale."""
    matrix = model_matrix(study, frame, terms)
    keep = [i for i, name in enumerate(matrix.names) if name != "Intercept" and np.ptp(matrix.values[:, i]) > 0]
    names = [matrix.names[i] for i in keep]
    correlation = np.abs(np.corrcoef(matrix.values[:, keep], rowvar=False))
    side = min(12, max(4.5, 0.32 * len(names) + 2.2))
    figure = _figure(side + 0.8, side)
    axis = figure.subplots()
    axis.grid(False)
    image = axis.imshow(correlation, cmap=SEQUENTIAL, vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=90)
    axis.set_yticks(range(len(names)), names)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.colorbar(image, ax=axis, shrink=0.75, label="|correlation| between terms")
    axis.set_title("How entangled are the model terms?")
    return figure


def plot_fraction_of_design_space(qualities: dict[str, DesignQuality]) -> Figure:
    """Compare designs by their prediction error across the region. Lower and flatter is better."""
    figure = _figure(6.4, 4.2)
    axis = figure.subplots()
    usable = {name: quality for name, quality in qualities.items() if quality.estimable}
    for index, (name, quality) in enumerate(list(usable.items())[: len(CATEGORICAL)]):
        ordered = np.sort(quality.prediction_se)
        fraction = np.linspace(0, 1, len(ordered))
        axis.plot(fraction, ordered, color=CATEGORICAL[index], label=name)
    axis.set_xlabel("Fraction of the design region")
    axis.set_ylabel("Prediction error (multiples of the noise)")
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.set_title("Prediction error across the design region")
    return figure


# ---------------------------------------------------------------------- model checks
def _scale_label(fit: ResponseFit) -> str:
    transform = fit.response.transform
    return f"{transform}({fit.response.name})" if transform != "none" else fit.response.label


def plot_diagnostics(fit: ResponseFit) -> Figure | None:
    """Six standard plots for checking whether a model can be trusted."""
    if fit.residual_df < 2:
        return None
    table = run_diagnostics(fit)
    residual = table["studentised_residual"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    figure = _figure(11.5, 6.8)
    axes = figure.subplots(2, 3).ravel()

    axes[0].plot([fit.y.min(), fit.y.max()], [fit.y.min(), fit.y.max()], color=AXIS, linewidth=1.2)
    _dots(axes[0], fit.fitted, fit.y)
    axes[0].set_xlabel(f"Predicted {_scale_label(fit)}")
    axes[0].set_ylabel(f"Observed {_scale_label(fit)}")
    axes[0].set_title("Observed vs predicted: dots should hug the line", fontsize=10.5)

    axes[1].axhline(0, color=AXIS, linewidth=1.2)
    _dots(axes[1], fit.fitted, residual)
    axes[1].set_xlabel(f"Predicted {_scale_label(fit)}")
    axes[1].set_ylabel("Studentised residual")
    axes[1].set_title("Residuals vs predicted: no funnel or curve", fontsize=10.5)

    axes[2].axhline(0, color=AXIS, linewidth=1.2)
    order = np.argsort(table["run"].to_numpy())
    axes[2].plot(table["run"].to_numpy()[order], residual[order], color=GRID, linewidth=1, zorder=1)
    _dots(axes[2], table["run"], residual)
    axes[2].set_xlabel("Run order")
    axes[2].set_ylabel("Studentised residual")
    axes[2].set_title("Residuals vs run order: no trend", fontsize=10.5)

    valid = residual[np.isfinite(residual)]
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(valid, dist="norm")
    axes[3].plot(theoretical, slope * theoretical + intercept, color=AXIS, linewidth=1.2)
    _dots(axes[3], theoretical, ordered)
    axes[3].set_xlabel("Normal quantile")
    axes[3].set_ylabel("Studentised residual")
    axes[3].set_title("Normal Q-Q: dots should follow the line", fontsize=10.5)

    axes[4].bar(table["run"], table["cooks_distance"], color=CATEGORICAL[0], width=0.6)
    axes[4].axhline(1, color=MUTED, linewidth=1)
    axes[4].set_xlabel("Run")
    axes[4].set_ylabel("Cook's distance")
    axes[4].set_title("Influence: bars above 1 are worth checking", fontsize=10.5)
    axes[4].set_ylim(bottom=0)

    limit = 2 * fit.matrix.rank / len(fit.y)
    axes[5].bar(table["run"], table["leverage"], color=CATEGORICAL[0], width=0.6)
    axes[5].axhline(limit, color=MUTED, linewidth=1)
    axes[5].set_xlabel("Run")
    axes[5].set_ylabel("Leverage")
    axes[5].set_title("Leverage: bars above the line pull hard on the model", fontsize=10.5)
    axes[5].set_ylim(0, 1.05)

    figure.suptitle(f"Model checks for {fit.response.label}")
    return figure


def plot_effects(fit: ResponseFit) -> Figure | None:
    """Rank the terms by how clearly they stand out from the noise."""
    table = coefficient_table(fit)
    if fit.residual_df <= 0:
        return plot_half_normal(fit)
    table = table[(table["term"] != "Intercept") & table["t"].notna() & ~table["group"].eq("block")]
    if table.empty:
        return None
    table = table.assign(size=table["t"].abs()).sort_values("size")
    critical = stats.t.ppf(0.975, fit.residual_df)
    figure = _figure(6.6, max(2.6, 0.28 * len(table) + 1.3))
    axis = figure.subplots()
    colours = [CATEGORICAL[0] if value > critical else AXIS for value in table["size"]]
    axis.barh(table["term"], table["size"], color=colours, height=0.62)
    axis.axvline(critical, color=INK_SECONDARY, linewidth=1)
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("Size of the effect relative to its uncertainty, |t|")
    axis.set_title(f"Which terms matter for {fit.response.label}?")
    axis.legend(
        handles=[
            Patch(color=CATEGORICAL[0], label="Clear effect (p < 0.05)"),
            Patch(color=AXIS, label="Could be noise"),
            Line2D([0], [0], color=INK_SECONDARY, linewidth=1, label="p = 0.05"),
        ],
        loc="lower right",
    )
    return figure


def plot_half_normal(fit: ResponseFit) -> Figure | None:
    """Judge effects by size when no noise estimate exists. Real effects leave the line."""
    table = coefficient_table(fit)
    lenth = lenth_analysis(table)
    if lenth is None:
        return None
    names = list(lenth["effects"])
    sizes = np.abs(np.array([lenth["effects"][name] for name in names]))
    order = np.argsort(sizes)
    count = len(sizes)
    quantiles = stats.norm.ppf(0.5 + 0.5 * (np.arange(1, count + 1) - 0.5) / count)
    figure = _figure(6.4, 4.4)
    axis = figure.subplots()
    axis.axvline(lenth["margin_of_error"], color=INK_SECONDARY, linewidth=1, label="Lenth margin of error")
    axis.plot(lenth["pse"] * quantiles, quantiles, color=AXIS, linewidth=1.2, label="Expected if only noise")
    _dots(axis, sizes[order], quantiles)
    for rank, index in enumerate(order):
        if names[index] in lenth["active"]:
            axis.annotate(names[index], (sizes[index], quantiles[rank]), xytext=(-6, 3), textcoords="offset points", ha="right", fontsize=9.5)
    axis.set_xlabel("Absolute effect")
    axis.set_ylabel("Half-normal quantile")
    axis.set_title(f"Half-normal plot for {fit.response.label}")
    axis.legend(loc="lower right")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    return figure


# ---------------------------------------------------------------------- effects
def _cox_trace(study: Study, name: str, hold: dict, points: int = 60):
    """Change one component while the others keep their proportions to each other."""
    factor = study.factor(name)
    names = study.mixture_names
    total = float(study.mixture_total)
    reference = float(hold[name])
    values = np.linspace(float(factor.low), float(factor.high), points)
    frame = _hold_frame(study, hold, points)
    scale = (total - values) / max(total - reference, 1e-12)
    for other in names:
        frame[other] = values if other == name else float(hold[other]) * scale
    keep = study.is_feasible(frame)
    return values[keep] - reference, frame[keep]


def plot_main_effects(fit: ResponseFit, hold: dict | None = None) -> Figure:
    """Show how the response changes with each factor, with the others held fixed."""
    study = fit.study
    hold = hold or study.center_point()
    process = study.process_factors
    panels = len(process) + (1 if study.mixture_factors else 0)
    figure, axes = _subplots(max(panels, 1), 3, 3.5, 3.0, share_y=True)
    panel = iter(axes)

    if study.mixture_factors:
        axis = next(panel)
        for index, name in enumerate(study.mixture_names):
            deviation, frame = _cox_trace(study, name, hold)
            if len(frame):
                axis.plot(deviation, fit.predict_mean(frame), color=CATEGORICAL[index % 8], label=name)
        axis.set_xlabel("Change from the reference blend")
        axis.set_ylabel(fit.response.label)
        axis.legend(fontsize=9)
        axis.set_title("Composition (Cox trace)", fontsize=10.5)

    for factor in process:
        axis = next(panel)
        if factor.kind == "categorical":
            frame = _hold_frame(study, hold, len(factor.levels))
            frame[factor.name] = np.array(factor.levels, dtype=object)
            prediction = fit.predict(frame)
            positions = np.arange(len(factor.levels))
            if prediction["ci_low"].notna().all():
                errors = np.vstack(
                    [prediction["predicted"] - prediction["ci_low"], prediction["ci_high"] - prediction["predicted"]]
                )
                axis.errorbar(positions, prediction["predicted"], yerr=errors, fmt="none", ecolor=CATEGORICAL[0], elinewidth=1.5, capsize=4)
            _dots(axis, positions, prediction["predicted"], size=56)
            axis.set_xticks(positions, factor.levels)
            axis.margins(x=0.25)
        else:
            frame = _hold_frame(study, hold, 60)
            values = np.linspace(float(factor.low), float(factor.high), 60)
            frame[factor.name] = values
            prediction = fit.predict(frame)
            if prediction["ci_low"].notna().all():
                axis.fill_between(values, prediction["ci_low"], prediction["ci_high"], color=CATEGORICAL[0], alpha=0.15, linewidth=0)
            axis.plot(values, prediction["predicted"], color=CATEGORICAL[0])
        axis.set_xlabel(factor.label)
        axis.set_ylabel(fit.response.label)
    for axis in axes:
        axis.label_outer(remove_inner_ticks=True)
    figure.suptitle(f"Effect of each factor on {fit.response.label}, with its 95% confidence band")
    return figure


def interaction_pairs(fit: ResponseFit, limit: int = 6) -> list[tuple[str, str]]:
    """List the process-factor interactions in the model, clearest first."""
    study = fit.study
    table = coefficient_table(fit).set_index("group")
    process = {factor.name for factor in study.process_factors}
    scored = []
    for term in fit.terms:
        if len(term) == 2 and term[0] != term[1] and set(term) <= process:
            label = ":".join(term)
            p_values = table.loc[[label], "p"] if label in table.index else pd.Series(dtype=float)
            scored.append((float(p_values.min()) if len(p_values) and p_values.notna().any() else 1.0, term))
    scored.sort(key=lambda item: item[0])
    return [item[1] for item in scored[:limit]]


def plot_interactions(fit: ResponseFit, hold: dict | None = None) -> Figure | None:
    """Show whether one factor's effect depends on another. Parallel lines mean it doesn't."""
    pairs = interaction_pairs(fit)
    if not pairs:
        return None
    study = fit.study
    hold = hold or study.center_point()
    figure, axes = _subplots(len(pairs), 3, 3.6, 3.0, share_y=True)
    for axis, (a, b) in zip(axes, pairs):
        # Put a numeric factor on the x axis where possible.
        if study.factor(a).kind == "categorical" and study.factor(b).kind != "categorical":
            a, b = b, a
        factor_a, factor_b = study.factor(a), study.factor(b)
        if factor_b.kind == "categorical":
            series = list(factor_b.levels)
        else:
            series = [float(factor_b.low), float(factor_b.high)]
        for index, value in enumerate(series[: len(CATEGORICAL)]):
            label = f"{b} = {_format_value(value)}"
            if factor_a.kind == "categorical":
                frame = _hold_frame(study, hold, len(factor_a.levels))
                frame[a] = np.array(factor_a.levels, dtype=object)
                frame[b] = value
                positions = np.arange(len(factor_a.levels))
                axis.plot(positions, fit.predict_mean(frame), color=CATEGORICAL[index], marker="o", label=label)
                axis.set_xticks(positions, factor_a.levels)
            else:
                frame = _hold_frame(study, hold, 40)
                values = np.linspace(float(factor_a.low), float(factor_a.high), 40)
                frame[a] = values
                frame[b] = value
                axis.plot(values, fit.predict_mean(frame), color=CATEGORICAL[index], label=label)
        axis.set_xlabel(factor_a.label)
        axis.set_ylabel(fit.response.label)
        axis.legend(fontsize=9)
    for axis in axes:
        axis.label_outer(remove_inner_ticks=True)
    figure.suptitle(f"Interactions for {fit.response.label}: parallel lines mean no interaction")
    return figure


# ---------------------------------------------------------------------- surfaces
def surface_pairs(fit: ResponseFit, max_factors: int = 5) -> list[tuple[str, str]]:
    """Choose which factor pairs are worth plotting, focusing on the most important factors."""
    study = fit.study
    table = coefficient_table(fit)
    scores: dict[str, float] = {}
    for name in study.numeric_process_names:
        used = table["group"].str.split(":").map(lambda parts: any(part.split("^")[0] == name for part in parts))
        if used.any():
            values = table.loc[used, "t"].abs() if table["t"].notna().any() else table.loc[used, "coef"].abs()
            scores[name] = float(values.fillna(0).sum())
    ranked = sorted(scores, key=scores.get, reverse=True)[:max_factors]
    chosen = [name for name in study.numeric_process_names if name in ranked]
    return list(itertools.combinations(chosen, 2))


def _pair_grid(study: Study, a: str, b: str, hold: dict, points: int = 45):
    factor_a, factor_b = study.factor(a), study.factor(b)
    values_a = np.linspace(float(factor_a.low), float(factor_a.high), points)
    values_b = np.linspace(float(factor_b.low), float(factor_b.high), points)
    grid_a, grid_b = np.meshgrid(values_a, values_b)
    frame = _hold_frame(study, hold, grid_a.size)
    frame[a] = grid_a.ravel()
    frame[b] = grid_b.ravel()
    return grid_a, grid_b, frame


def _masked(study: Study, frame: pd.DataFrame, values: np.ndarray, shape) -> np.ma.MaskedArray:
    return np.ma.masked_where(~study.is_feasible(frame).reshape(shape), values.reshape(shape))


def _draw_contour(axis, grid_a, grid_b, values, label: str, figure):
    axis.grid(False)
    filled = axis.contourf(grid_a, grid_b, values, levels=12, cmap=SEQUENTIAL)
    lines = axis.contour(grid_a, grid_b, values, levels=filled.levels, colors="white", linewidths=0.5, alpha=0.8)
    axis.clabel(lines, lines.levels[::2], fontsize=8, colors="white")
    figure.colorbar(filled, ax=axis, shrink=0.85, label=label)


def _mark_best(axis, x: float, y: float) -> None:
    """Mark the best settings, with a label placed towards the middle of the plot."""
    axis.scatter([x], [y], marker="X", s=130, color=INK, edgecolor="white", linewidth=1.2, zorder=10, clip_on=False)
    to_right = x < np.mean(axis.get_xlim())
    upward = y < np.mean(axis.get_ylim())
    axis.annotate(
        "Best",
        (x, y),
        xytext=(11 if to_right else -11, 11 if upward else -11),
        textcoords="offset points",
        ha="left" if to_right else "right",
        va="bottom" if upward else "top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        zorder=11,
    )


def _mark_best_ternary(axis, best: dict, components: list[str]) -> None:
    axis.scatter(
        [best[components[0]]], [best[components[1]]], [best[components[2]]],
        marker="X", s=130, color=INK, edgecolor="white", linewidth=1.2, zorder=10, clip_on=False,
    )  # fmt: skip


def _ternary_colorbar(figure, axis, mappable, label: str) -> None:
    """Add a colour bar far enough from the triangle to clear its axis label."""
    bar_axis = axis.inset_axes([1.3, 0.12, 0.035, 0.76], transform=axis.transAxes)
    figure.colorbar(mappable, cax=bar_axis, label=label)


def plot_surface(fit: ResponseFit, a: str, b: str, hold: dict | None = None, best: dict | None = None) -> Figure:
    """Draw a contour map and a 3D surface of the response over two factors.

    Other factors are held at ``hold``: the centre of the region by default, or the best
    settings after optimising. With ``best``, the marked point lies exactly on the surface,
    because the surface is drawn through it.
    """
    study = fit.study
    hold = hold or study.center_point()
    grid_a, grid_b, frame = _pair_grid(study, a, b, hold)
    values = fit.predict_mean(frame)
    masked = _masked(study, frame, values, grid_a.shape)
    figure = _figure(10.8, 4.5)
    contour_axis = figure.add_subplot(1, 2, 1)
    _draw_contour(contour_axis, grid_a, grid_b, masked, fit.response.label, figure)
    contour_axis.set_xlabel(study.factor(a).label)
    contour_axis.set_ylabel(study.factor(b).label)

    surface_axis = figure.add_subplot(1, 2, 2, projection="3d")
    surface_axis.plot_surface(
        grid_a, grid_b, masked, cmap=SEQUENTIAL, linewidth=0, antialiased=True,
        rstride=1, cstride=1, vmin=float(masked.min()), vmax=float(masked.max()),
    )  # fmt: skip
    surface_axis.set_xlabel(a, labelpad=6)
    surface_axis.set_ylabel(b, labelpad=6)
    surface_axis.set_zlabel(fit.response.name, labelpad=6)
    surface_axis.tick_params(labelsize=8.5)
    for pane in (surface_axis.xaxis, surface_axis.yaxis, surface_axis.zaxis):
        pane.set_pane_color("white")
        pane._axinfo["grid"].update(color=GRID, linewidth=0.6)

    if best is not None:
        _mark_best(contour_axis, float(best[a]), float(best[b]))
        point = _hold_frame(study, hold, 1)
        point[a], point[b] = best[a], best[b]
        surface_axis.scatter([best[a]], [best[b]], fit.predict_mean(point), marker="X", s=100, color=INK, depthshade=False)

    caption = _hold_text(study, hold, [a, b])
    figure.suptitle(f"{fit.response.label} vs {a} and {b}" + (f"\n{caption}" if caption else ""), fontsize=11)
    return figure


def ternary_triples(study: Study, limit: int = 6) -> list[list[str]]:
    return [list(triple) for triple in itertools.combinations(study.mixture_names, 3)][:limit]


def _ternary_grid(study: Study, components: list[str], hold: dict, degree: int = 70):
    """Build a grid of blends for three components, with the rest held fixed."""
    total = _selected_total(study, components, hold)
    lower = np.array([float(study.factor(name).low) for name in components])
    points = lower + mixture_designs.lattice_points(3, degree) * (total - lower.sum())
    frame = _hold_frame(study, hold, len(points))
    frame[components] = points
    keep = study.is_feasible(frame)
    return frame[keep].reset_index(drop=True), total


def plot_ternary(fit: ResponseFit, components: list[str], hold: dict | None = None, best: dict | None = None) -> Figure:
    """Draw the response over the allowed blends. Other factors are held at ``hold``."""
    study = fit.study
    hold = hold or study.center_point()
    frame, total = _ternary_grid(study, components, hold)
    figure = _figure(7.8, 5.4)
    axis = _ternary_axis(figure, 111, study, components, total)
    t, l, r = (frame[name].to_numpy(dtype=float) for name in components)
    values = fit.predict_mean(frame)
    filled = axis.tricontourf(t, l, r, values, levels=12, cmap=SEQUENTIAL)
    axis.tricontour(t, l, r, values, levels=filled.levels, colors="white", linewidths=0.5, alpha=0.8)
    _ternary_colorbar(figure, axis, filled, fit.response.label)
    title = f"{fit.response.label} over the blend region"
    if best is not None:
        _mark_best_ternary(axis, best, components)
        title += " (X marks the best settings)"
    caption = _hold_text(study, hold, components)
    figure.suptitle(title + (f"\n{caption}" if caption else ""), fontsize=11)
    return figure


def plot_desirability_map(fits: list[ResponseFit], limits: list[Limits], a: str, b: str, hold: dict, best: dict | None = None) -> Figure:
    """Map overall desirability over two process factors, to show how wide the sweet spot is."""
    study = fits[0].study
    grid_a, grid_b, frame = _pair_grid(study, a, b, hold)
    score, _, _ = overall_desirability(fits, limits, frame)
    figure = _figure(5.6, 4.4)
    axis = figure.subplots()
    axis.grid(False)
    masked = _masked(study, frame, score, grid_a.shape)
    filled = axis.contourf(grid_a, grid_b, masked, levels=np.linspace(0, 1, 11), cmap=SEQUENTIAL, vmin=0, vmax=1)
    figure.colorbar(filled, ax=axis, shrink=0.85, label="Overall desirability")
    axis.set_xlabel(study.factor(a).label)
    axis.set_ylabel(study.factor(b).label)
    if best is not None:
        _mark_best(axis, float(best[a]), float(best[b]))
    figure.suptitle(f"Sweet spot over {a} and {b}\n{_hold_text(study, hold, [a, b])}", fontsize=11)
    return figure


def plot_desirability_ternary(fits: list[ResponseFit], limits: list[Limits], components: list[str], hold: dict, best: dict | None = None) -> Figure:
    """Map overall desirability over the allowed blends."""
    study = fits[0].study
    frame, total = _ternary_grid(study, components, hold)
    score, _, _ = overall_desirability(fits, limits, frame)
    figure = _figure(7.8, 5.4)
    axis = _ternary_axis(figure, 111, study, components, total)
    t, l, r = (frame[name].to_numpy(dtype=float) for name in components)
    filled = axis.tricontourf(t, l, r, score, levels=np.linspace(0, 1, 11), cmap=SEQUENTIAL, vmin=0, vmax=1)
    _ternary_colorbar(figure, axis, filled, "Overall desirability")
    title = "Sweet spot over the blend region"
    if best is not None:
        _mark_best_ternary(axis, best, components)
        title += " (X marks the best settings)"
    caption = _hold_text(study, hold, components)
    figure.suptitle(title + (f"\n{caption}" if caption else ""), fontsize=11)
    return figure
