"""Turn a study into the regression model that will be fitted.

A term is a tuple of factor names:

    ("temp",)                  the main effect of temp
    ("temp", "pressure")       the temp x pressure interaction
    ("temp", "temp")           the curvature of temp

Mixture studies use Scheffe models, which have no intercept. Because the components
always add up to the same total, a constant can't be separated from them, so each
component's own term plays that role.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .study import Study, StudyError

Term = tuple[str, ...]

PROCESS_ORDERS = ("linear", "pure_quadratic", "interaction", "quadratic", "cubic")
MIXTURE_ORDERS = ("linear", "quadratic", "special_cubic")
CROSSING_ORDERS = ("none", "linear")


def term_label(term: Term) -> str:
    """Write a term for people to read: ('a', 'a', 'b') becomes 'a^2:b'."""
    counts = Counter(term)
    return ":".join(name if power == 1 else f"{name}^{power}" for name, power in counts.items())


def parse_term(text: str, study: Study) -> Term:
    """Read a term from a config file. Accepts 'a', 'a*b', 'a:b', 'a^2' and 'a^2*b'."""
    parts = []
    for piece in str(text).replace(":", "*").split("*"):
        name, _, power = piece.strip().partition("^")
        name = name.strip()
        if name not in study.factor_names:
            raise StudyError(f"The model term '{text}' uses an unknown factor: '{name}'.")
        try:
            parts.extend([name] * int(power or 1))
        except ValueError as error:
            raise StudyError(f"Couldn't read the power in the model term '{text}'.") from error
    return tuple(parts)


def default_process_order(study: Study) -> str:
    """Choose the richest model the chosen design type can normally support."""
    design = study.design
    kind = design.get("type")
    if kind == "plackett_burman":
        return "linear"
    if kind == "taguchi":
        return "pure_quadratic" if int(design.get("levels", 3)) > 2 else "linear"
    if kind == "fractional_factorial":
        return "interaction" if int(design.get("resolution", 4)) >= 5 else "linear"
    if kind in ("full_factorial", "gsd"):
        levels = design.get("levels", 2)
        most_levels = max(levels.values()) if isinstance(levels, dict) else int(levels)
        return "quadratic" if most_levels > 2 else "interaction"
    return "quadratic"


def default_mixture_order(study: Study) -> str:
    design = study.design
    kind = design.get("type")
    if kind == "simplex_lattice" and int(design.get("degree", 2)) == 1:
        return "linear"
    if kind == "simplex_centroid":
        return "special_cubic" if len(study.mixture_names) >= 3 else "quadratic"
    return "quadratic"


def _can_have_curvature(study: Study, name: str, frame: pd.DataFrame | None) -> bool:
    factor = study.factor(name)
    if factor.kind == "discrete" and len(factor.levels) < 3:
        return False
    if frame is not None and name in frame and frame[name].nunique() < 3:
        return False
    return True


def _process_terms(study: Study, order: str, frame: pd.DataFrame | None) -> tuple[list[Term], list[Term]]:
    """Return (main effects, higher-order terms) for the non-mixture factors."""
    numeric = study.numeric_process_names
    categorical = [f.name for f in study.process_factors if f.kind == "categorical"]
    names = numeric + categorical
    mains = [(name,) for name in names]
    pairs = list(itertools.combinations(names, 2))
    curved = [name for name in numeric if _can_have_curvature(study, name, frame)]
    squares = [(name, name) for name in curved]

    if order == "linear":
        return mains, []
    if order == "interaction":
        return mains, pairs
    if order == "pure_quadratic":
        return mains, squares
    if order == "quadratic":
        return mains, pairs + squares
    if order == "cubic":
        cubics = []
        for term in itertools.combinations_with_replacement(names, 3):
            counts = Counter(term)
            allowed = all(power == 1 or name in curved for name, power in counts.items())
            if allowed:
                cubics.append(term)
        return mains, pairs + squares + cubics
    raise StudyError(f"Unknown process model '{order}'. Choose from: {', '.join(PROCESS_ORDERS)}.")


def _mixture_terms(names: list[str], order: str) -> list[Term]:
    terms = [(name,) for name in names]
    if order == "linear":
        return terms
    if order not in MIXTURE_ORDERS:
        raise StudyError(f"Unknown mixture model '{order}'. Choose from: {', '.join(MIXTURE_ORDERS)}.")
    terms += list(itertools.combinations(names, 2))
    if order == "special_cubic":
        terms += list(itertools.combinations(names, 3))
    return terms


def model_terms(study: Study, frame: pd.DataFrame | None = None) -> list[Term]:
    """List the model's terms, from the config's explicit list or from the model orders.

    ``frame`` is the actual design. When it's given, curvature terms are left out for
    factors that only took two values in it, because they couldn't be estimated.
    """
    spec = study.model if isinstance(study.model, dict) else {"process": study.model}
    if spec.get("terms"):
        terms = [parse_term(text, study) for text in spec["terms"]]
        if study.mixture_names:
            terms = [(name,) for name in study.mixture_names if (name,) not in terms] + terms
        return terms

    mixture = study.mixture_names
    process = study.process_factors
    terms: list[Term] = []
    if mixture:
        terms += _mixture_terms(mixture, spec.get("mixture", default_mixture_order(study)))
    if process:
        mains, higher = _process_terms(study, spec.get("process", default_process_order(study)), frame)
        if mixture:
            crossing = spec.get("mixture_process", "linear")
            if crossing not in CROSSING_ORDERS:
                raise StudyError(
                    f"Unknown mixture_process setting '{crossing}'. "
                    f"Choose from: {', '.join(CROSSING_ORDERS)}."
                )
            if crossing == "linear":
                # Each process effect may depend on the blend, so give every
                # component its own version of it. These terms replace the plain
                # main effects, which would otherwise be redundant.
                mains = [(component,) + main for main in mains for component in mixture]
        terms += mains + higher
    return terms


def has_intercept(study: Study) -> bool:
    """Scheffe mixture models have no intercept. Every other model has one."""
    return not study.mixture_names


@dataclass
class ModelMatrix:
    """The numbers regression works on: one row per run, one column per coefficient."""

    values: np.ndarray
    names: list[str]
    groups: list[str]  # the term each column belongs to

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=self.names)

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.values))


def _effect_columns(values: np.ndarray, levels: tuple) -> dict[str, np.ndarray]:
    """Code a categorical factor so each coefficient is a difference from the overall mean.

    The last level gets -1 in every column, so its effect is minus the sum of the rest.
    """
    last = (values == levels[-1]).astype(float)
    return {str(level): (values == level).astype(float) - last for level in levels[:-1]}


def _term_columns(term: Term, coded: dict, effects: dict, rows: int) -> list[tuple[str, np.ndarray]]:
    columns = [("", np.ones(rows))]
    for name, power in Counter(term).items():
        if name in effects:
            columns = [
                (f"{label}:{name}[{level}]", values * dummy)
                for label, values in columns
                for level, dummy in effects[name].items()
            ]
        else:
            piece = name if power == 1 else f"{name}^{power}"
            columns = [(f"{label}:{piece}", values * coded[name] ** power) for label, values in columns]
    return [(label[1:], values) for label, values in columns]


def model_matrix(
    study: Study,
    frame: pd.DataFrame,
    terms: list[Term],
    include_blocks: bool = True,
) -> ModelMatrix:
    """Build the model matrix for a set of runs, using coded factor values."""
    rows = len(frame)
    coded = study.coded(frame)
    effects = {
        factor.name: _effect_columns(frame[factor.name].to_numpy(), factor.levels)
        for factor in study.factors
        if factor.kind == "categorical"
    }
    names: list[str] = []
    groups: list[str] = []
    columns: list[np.ndarray] = []

    if has_intercept(study):
        names.append("Intercept")
        groups.append("Intercept")
        columns.append(np.ones(rows))

    for term in terms:
        label = term_label(term)
        for name, values in _term_columns(term, coded, effects, rows):
            names.append(name)
            groups.append(label)
            columns.append(values)

    if include_blocks and "block" in frame and frame["block"].nunique() > 1:
        levels = tuple(sorted(frame["block"].unique().tolist()))
        for level, dummy in _effect_columns(frame["block"].to_numpy(), levels).items():
            names.append(f"block[{level}]")
            groups.append("block")
            columns.append(dummy)

    return ModelMatrix(np.column_stack(columns), names, groups)


def find_aliases(matrix: ModelMatrix) -> list[tuple[str, str]]:
    """Find pairs of model columns that are identical, up to scale.

    Two such columns can't be told apart by the data. This uses uncentred cosine
    similarity so that a column aliased with the intercept is also caught.
    """
    values = matrix.values
    norms = np.linalg.norm(values, axis=0)
    aliases = []
    for i, j in itertools.combinations(range(values.shape[1]), 2):
        if norms[i] < 1e-12 or norms[j] < 1e-12:
            continue
        if abs(values[:, i] @ values[:, j]) / (norms[i] * norms[j]) > 1 - 1e-9:
            aliases.append((matrix.names[i], matrix.names[j]))
    return aliases
