"""Describe an experiment: its factors, responses, constraints and settings.

A ``Study`` is what one YAML file in ``configs/`` turns into. Every other module
takes a ``Study``, so this file is the place to start when reading the code.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .expressions import evaluate, names_in

FACTOR_KINDS = ("continuous", "discrete", "categorical", "mixture")
GOALS = ("maximise", "minimise", "target", "in_range", "none")
TRANSFORMS = ("none", "log", "sqrt")
RESERVED_COLUMNS = {"run", "std_order", "block", "notes"}

_GOAL_ALIASES = {"maximize": "maximise", "max": "maximise", "minimize": "minimise", "min": "minimise"}
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOLERANCE = 1e-9


class StudyError(ValueError):
    """A problem with a study's config or data, explained in plain language."""


def _check_name(name, what: str) -> str:
    name = str(name)
    if not _NAME_PATTERN.match(name):
        raise StudyError(
            f"{what} name '{name}' isn't allowed. Use letters, numbers and underscores, "
            "starting with a letter, for example 'nozzle_temp'."
        )
    if name in RESERVED_COLUMNS:
        raise StudyError(f"{what} name '{name}' is reserved for the run sheet. Choose another name.")
    return name


@dataclass(frozen=True)
class Factor:
    """Something you deliberately change between runs.

    kind:
        continuous   any value between ``low`` and ``high`` (temperature, pressure)
        discrete     numbers, but only certain values exist (nozzle gauge: 22, 25, 27)
        categorical  named options with no numeric order (crosslinker: CaCl2, BaCl2)
        mixture      a component of a blend, where all components add up to a fixed total
    """

    name: str
    kind: str = "continuous"
    low: float | None = None
    high: float | None = None
    levels: tuple = ()
    units: str = ""

    def __post_init__(self):
        _check_name(self.name, "Factor")
        if self.kind not in FACTOR_KINDS:
            raise StudyError(
                f"Factor '{self.name}' has unknown type '{self.kind}'. "
                f"Choose from: {', '.join(FACTOR_KINDS)}."
            )
        if self.kind == "categorical":
            if len(self.levels) < 2:
                raise StudyError(f"Categorical factor '{self.name}' needs at least 2 levels.")
            object.__setattr__(self, "levels", tuple(str(level) for level in self.levels))
        elif self.kind == "discrete":
            if len(self.levels) < 2:
                raise StudyError(f"Discrete factor '{self.name}' needs at least 2 levels.")
            try:
                levels = tuple(sorted(float(level) for level in self.levels))
            except (TypeError, ValueError) as error:
                raise StudyError(
                    f"Discrete factor '{self.name}' needs numeric levels. "
                    "Use type: categorical for named options."
                ) from error
            object.__setattr__(self, "levels", levels)
            object.__setattr__(self, "low", levels[0])
            object.__setattr__(self, "high", levels[-1])
        elif self.kind == "continuous":
            if self.low is None or self.high is None:
                raise StudyError(f"Continuous factor '{self.name}' needs both low and high.")
            if not float(self.low) < float(self.high):
                raise StudyError(f"Factor '{self.name}': low must be smaller than high.")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.units})" if self.units else self.name

    @property
    def is_numeric(self) -> bool:
        return self.kind != "categorical"

    @property
    def center(self) -> float:
        return (float(self.low) + float(self.high)) / 2

    @property
    def half_range(self) -> float:
        return (float(self.high) - float(self.low)) / 2

    def code(self, values):
        """Convert actual values to coded units, where low is -1 and high is +1."""
        return (np.asarray(values, dtype=float) - self.center) / self.half_range

    def decode(self, coded):
        """Convert coded units back to actual values."""
        return self.center + np.asarray(coded, dtype=float) * self.half_range

    def nearest_level(self, values):
        """Snap values to the closest allowed level of a discrete factor."""
        levels = np.asarray(self.levels, dtype=float)
        values = np.asarray(values, dtype=float)
        return levels[np.abs(values[..., None] - levels).argmin(axis=-1)]


@dataclass(frozen=True)
class Response:
    """Something you measure, and what a good value looks like.

    goal:
        maximise   ``low`` is the worst acceptable value, ``high`` is as good as it gets
        minimise   ``low`` is as good as it gets, ``high`` is the worst acceptable value
        target     ``target`` is ideal, and ``low`` and ``high`` are the acceptable limits
        in_range   anything between ``low`` and ``high`` is equally good
        none       measured and modelled, but not optimised

    importance is how much this response counts relative to the others, and shape
    bends the desirability curve (above 1 is stricter, below 1 is more lenient).
    transform is applied before fitting, for example ``log`` for viscosity.
    """

    name: str
    units: str = ""
    goal: str = "none"
    low: float | None = None
    high: float | None = None
    target: float | None = None
    importance: float = 1.0
    shape: float = 1.0
    transform: str = "none"

    def __post_init__(self):
        _check_name(self.name, "Response")
        goal = _GOAL_ALIASES.get(str(self.goal).lower(), str(self.goal).lower())
        object.__setattr__(self, "goal", goal)
        if goal not in GOALS:
            raise StudyError(
                f"Response '{self.name}' has unknown goal '{self.goal}'. "
                f"Choose from: {', '.join(GOALS)}."
            )
        if self.transform not in TRANSFORMS:
            raise StudyError(
                f"Response '{self.name}' has unknown transform '{self.transform}'. "
                f"Choose from: {', '.join(TRANSFORMS)}."
            )
        if goal == "target" and self.target is None:
            raise StudyError(f"Response '{self.name}' has goal 'target', so it needs a target value.")
        if goal == "in_range" and (self.low is None or self.high is None):
            raise StudyError(f"Response '{self.name}' has goal 'in_range', so it needs low and high.")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise StudyError(f"Response '{self.name}': low must be smaller than high.")
        if goal == "target":
            if self.low is not None and not self.low < self.target:
                raise StudyError(f"Response '{self.name}': target must be above low.")
            if self.high is not None and not self.target < self.high:
                raise StudyError(f"Response '{self.name}': target must be below high.")
        if self.importance <= 0 or self.shape <= 0:
            raise StudyError(f"Response '{self.name}': importance and shape must be above 0.")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.units})" if self.units else self.name


@dataclass(frozen=True)
class Constraint:
    """A limit on a combination of factors, for example gelma + alginate <= 0.9."""

    expr: str
    min: float | None = None
    max: float | None = None

    def __post_init__(self):
        if self.min is None and self.max is None:
            raise StudyError(f"Constraint '{self.expr}' needs a min, a max, or both.")

    def describe(self) -> str:
        parts = []
        if self.min is not None:
            parts.append(f"{self.min} <=")
        parts.append(self.expr)
        if self.max is not None:
            parts.append(f"<= {self.max}")
        return " ".join(parts)


def mixture_vertices(lower, upper, total: float) -> np.ndarray:
    """Find the corners of a constrained mixture region (McLean & Anderson, 1966).

    Every corner has all but one component sitting on a bound, so try each
    combination of bounds and keep those where the remaining component fits.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    size = len(lower)
    found = {}
    for free in range(size):
        others = [i for i in range(size) if i != free]
        for bounds in itertools.product(*[(lower[i], upper[i]) for i in others]):
            remainder = total - sum(bounds)
            if lower[free] - _TOLERANCE <= remainder <= upper[free] + _TOLERANCE:
                point = np.zeros(size)
                point[others] = bounds
                point[free] = remainder
                found[tuple(np.round(point, 10))] = point
    return np.array(list(found.values()))


@dataclass
class Study:
    """Everything needed to design, analyse and optimise one experiment."""

    name: str
    factors: list[Factor]
    responses: list[Response] = field(default_factory=list)
    description: str = ""
    design: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    mixture_total: float = 1.0
    seed: int | None = None
    simulation: dict = field(default_factory=dict)
    figure_format: str = "svg"
    root: Path = field(default_factory=Path.cwd)
    data_file: Path | None = None

    def __post_init__(self):
        _check_name(self.name, "Study")
        names = [factor.name for factor in self.factors]
        response_names = [response.name for response in self.responses]
        repeated = {name for name in names + response_names if (names + response_names).count(name) > 1}
        if repeated:
            raise StudyError(f"These names are used more than once: {', '.join(sorted(repeated))}.")
        if not self.factors:
            raise StudyError("A study needs at least one factor.")
        self._prepare_mixture()
        for constraint in self.constraints:
            unknown = names_in(constraint.expr) - set(names)
            if unknown:
                raise StudyError(
                    f"Constraint '{constraint.expr}' uses unknown factors: {', '.join(sorted(unknown))}."
                )
        if isinstance(self.model, str):
            self.model = {"process": self.model}
        allowed = {"process", "mixture", "mixture_process", "terms", "reduce", "alpha"}
        unknown = set(self.model) - allowed
        if unknown:
            raise StudyError(
                f"Unknown model setting(s): {', '.join(sorted(unknown))}. "
                f"Available settings: {', '.join(sorted(allowed))}."
            )

    def _prepare_mixture(self):
        """Fill in default bounds and tighten any bound the others make unreachable."""
        components = [factor for factor in self.factors if factor.kind == "mixture"]
        if not components:
            return
        if len(components) < 2:
            raise StudyError("A mixture needs at least 2 components.")
        total = float(self.mixture_total)
        lower = np.array([0.0 if f.low is None else float(f.low) for f in components])
        upper = np.array([total if f.high is None else float(f.high) for f in components])
        if (upper <= lower).any():
            raise StudyError("Every mixture component needs a low that is smaller than its high.")
        if lower.sum() >= total - _TOLERANCE:
            raise StudyError(
                f"The lower limits of the mixture components add up to {lower.sum():g}, which leaves "
                f"no room to vary them within the total of {total:g}. Lower some of the limits."
            )
        if upper.sum() <= total + _TOLERANCE:
            raise StudyError(
                f"The upper limits of the mixture components add up to {upper.sum():g}, which leaves "
                f"no room to vary them within the total of {total:g}. Raise some of the limits."
            )
        new_upper = np.minimum(upper, total - (lower.sum() - lower))
        new_lower = np.maximum(lower, total - (upper.sum() - upper))
        bounds = dict(zip([f.name for f in components], zip(new_lower, new_upper)))
        self.factors = [
            replace(f, low=float(bounds[f.name][0]), high=float(bounds[f.name][1]))
            if f.kind == "mixture"
            else f
            for f in self.factors
        ]

    # ------------------------------------------------------------------ factor groups
    def factor(self, name: str) -> Factor:
        for factor in self.factors:
            if factor.name == name:
                return factor
        raise StudyError(f"There is no factor called '{name}' in this study.")

    @property
    def factor_names(self) -> list[str]:
        return [factor.name for factor in self.factors]

    @property
    def mixture_factors(self) -> list[Factor]:
        return [factor for factor in self.factors if factor.kind == "mixture"]

    @property
    def mixture_names(self) -> list[str]:
        return [factor.name for factor in self.mixture_factors]

    @property
    def process_factors(self) -> list[Factor]:
        """Everything that isn't a mixture component."""
        return [factor for factor in self.factors if factor.kind != "mixture"]

    @property
    def numeric_process_names(self) -> list[str]:
        return [factor.name for factor in self.process_factors if factor.is_numeric]

    @property
    def categorical_names(self) -> list[str]:
        return [factor.name for factor in self.factors if factor.kind == "categorical"]

    @property
    def response_names(self) -> list[str]:
        return [response.name for response in self.responses]

    def response(self, name: str) -> Response:
        for response in self.responses:
            if response.name == name:
                return response
        raise StudyError(f"There is no response called '{name}' in this study.")

    # ------------------------------------------------------------------ mixture geometry
    def mixture_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        components = self.mixture_factors
        return (
            np.array([float(f.low) for f in components]),
            np.array([float(f.high) for f in components]),
        )

    @property
    def pseudo_range(self) -> float:
        """The share of the total that is free to vary once every lower limit is met."""
        return float(self.mixture_total) - self.mixture_bounds()[0].sum()

    def mixture_centroid(self) -> dict[str, float]:
        lower, upper = self.mixture_bounds()
        centroid = mixture_vertices(lower, upper, float(self.mixture_total)).mean(axis=0)
        return dict(zip(self.mixture_names, centroid))

    # ------------------------------------------------------------------ coding
    def coded(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Put numeric factors on a common scale for modelling.

        Process factors run from -1 to +1. Mixture components become
        pseudo-components, which run from 0 to 1 within the allowed region. Coding
        makes coefficients directly comparable with each other, whatever the units.
        """
        coded = {}
        for factor in self.factors:
            if factor.kind == "mixture":
                values = np.asarray(frame[factor.name], dtype=float)
                coded[factor.name] = (values - float(factor.low)) / self.pseudo_range
            elif factor.is_numeric:
                coded[factor.name] = factor.code(frame[factor.name])
        return coded

    def center_point(self) -> dict:
        """A sensible point to hold factors at when plotting the others."""
        point = {}
        for factor in self.factors:
            if factor.kind == "categorical":
                point[factor.name] = factor.levels[0]
            elif factor.kind == "discrete":
                point[factor.name] = float(factor.nearest_level(factor.center))
            elif factor.kind == "continuous":
                point[factor.name] = factor.center
        if self.mixture_factors:
            point.update(self.mixture_centroid())
        return point

    # ------------------------------------------------------------------ feasibility
    def violation(self, frame: pd.DataFrame) -> np.ndarray:
        """How far each row is outside the constraints and mixture limits (0 means inside)."""
        total = np.zeros(len(frame))
        for constraint in self.constraints:
            variables = {name: frame[name].to_numpy() for name in self.factor_names if name in frame}
            values = np.asarray(evaluate(constraint.expr, variables), dtype=float)
            if constraint.min is not None:
                scale = max(abs(constraint.min), 1e-6)
                total += np.maximum(0.0, (constraint.min - values) / scale)
            if constraint.max is not None:
                scale = max(abs(constraint.max), 1e-6)
                total += np.maximum(0.0, (values - constraint.max) / scale)
        if self.mixture_factors:
            scale = self.pseudo_range
            for factor in self.mixture_factors:
                values = np.asarray(frame[factor.name], dtype=float)
                total += np.maximum(0.0, (float(factor.low) - values) / scale)
                total += np.maximum(0.0, (values - float(factor.high)) / scale)
        return total

    def is_feasible(self, frame: pd.DataFrame, tolerance: float = 1e-6) -> np.ndarray:
        return self.violation(frame) <= tolerance

    def only(self, names: list[str]) -> "Study":
        """A copy of this study containing only the named factors."""
        kept = set(names)
        return replace(
            self,
            factors=[factor for factor in self.factors if factor.name in kept],
            constraints=[c for c in self.constraints if names_in(c.expr) <= kept],
        )

    # ------------------------------------------------------------------ file locations
    @property
    def runsheet_path(self) -> Path:
        return self.data_file if self.data_file else self.root / "data" / self.name / "runsheet.csv"

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / self.name


def find_project_root(start: Path) -> Path:
    """Find the project folder by walking up to the nearest pyproject.toml."""
    start = Path(start).resolve()
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    return start


def _factor_from_config(entry: dict) -> Factor:
    if not isinstance(entry, dict) or "name" not in entry:
        raise StudyError(f"Each factor needs at least a name. Couldn't read: {entry}")
    allowed = {"name", "type", "low", "high", "levels", "units"}
    unknown = set(entry) - allowed
    if unknown:
        raise StudyError(
            f"Factor '{entry['name']}' has unknown setting(s): {', '.join(sorted(unknown))}. "
            f"Available settings: {', '.join(sorted(allowed))}."
        )
    levels = tuple(entry.get("levels") or ())
    kind = entry.get("type")
    if kind is None:
        if not levels:
            kind = "continuous"
        else:
            numeric = all(
                isinstance(level, (int, float)) and not isinstance(level, bool) for level in levels
            )
            kind = "discrete" if numeric else "categorical"
    return Factor(
        name=entry["name"],
        kind=kind,
        low=entry.get("low"),
        high=entry.get("high"),
        levels=levels,
        units=str(entry.get("units", "")),
    )


def _response_from_config(entry: dict) -> Response:
    if not isinstance(entry, dict) or "name" not in entry:
        raise StudyError(f"Each response needs at least a name. Couldn't read: {entry}")
    allowed = {"name", "units", "goal", "low", "high", "target", "importance", "shape", "transform"}
    unknown = set(entry) - allowed
    if unknown:
        raise StudyError(
            f"Response '{entry['name']}' has unknown setting(s): {', '.join(sorted(unknown))}. "
            f"Available settings: {', '.join(sorted(allowed))}."
        )
    settings = dict(entry)
    settings["units"] = str(settings.get("units", ""))
    return Response(**settings)


ALLOWED_SETTINGS = {
    "name", "description", "seed", "factors", "mixture_total", "constraints",
    "responses", "design", "model", "simulation", "figure_format", "data_file",
}  # fmt: skip


def without_server_paths(config: dict) -> dict:
    """Remove settings that point at files, from a config someone has uploaded.

    On a hosted app, a config's data_file could otherwise be used to read files from the
    server. Uploaded data goes through the run sheet upload instead.
    """
    if not isinstance(config, dict):
        raise StudyError("That doesn't look like a study config.")
    return {key: value for key, value in config.items() if key != "data_file"}


def study_from_config(config: dict, root: Path | None = None, source: str = "The config") -> Study:
    """Build a study from a dictionary, which is what a YAML config file contains."""
    if not isinstance(config, dict):
        raise StudyError(f"{source} doesn't look like a study config.")
    unknown = set(config) - ALLOWED_SETTINGS
    if unknown:
        raise StudyError(
            f"{source} has unknown setting(s): {', '.join(sorted(unknown))}. "
            f"Available settings: {', '.join(sorted(ALLOWED_SETTINGS))}."
        )
    for required in ("name", "factors"):
        if required not in config:
            raise StudyError(f"{source} needs a '{required}' setting.")

    root = Path(root) if root else Path.cwd()
    data_file = config.get("data_file")
    return Study(
        name=config["name"],
        description=str(config.get("description", "")).strip(),
        factors=[_factor_from_config(entry) for entry in config["factors"]],
        responses=[_response_from_config(entry) for entry in config.get("responses") or []],
        design=dict(config.get("design") or {}),
        model=config.get("model") or {},
        constraints=[Constraint(**entry) for entry in config.get("constraints") or []],
        mixture_total=float(config.get("mixture_total", 1.0)),
        seed=config.get("seed"),
        simulation=dict(config.get("simulation") or {}),
        figure_format=str(config.get("figure_format", "svg")),
        root=root,
        data_file=root / data_file if data_file else None,
    )


def load_study(path) -> Study:
    """Read a study's YAML config file."""
    path = Path(path)
    if not path.exists():
        raise StudyError(f"Couldn't find the config file '{path}'.")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise StudyError(f"Couldn't read '{path}'. Check its indentation and punctuation.\n{error}") from error
    return study_from_config(config, find_project_root(path.parent), f"'{path.name}'")
