"""Factorial-family designs, mostly used for screening: which factors matter?"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pyDOE3

from ..study import Factor, Study, StudyError

# Standard minimum-aberration two-level fractional factorials, as tabulated in
# Montgomery, Design and Analysis of Experiments. Each entry is (resolution, generators),
# from fewest runs to most. Letters are the base factors, and words like "abc" define
# the extra factors. tests/test_designs.py recalculates every resolution listed here.
#
# pyDOE3.fracfact_by_res isn't used because it refuses standard designs such as the
# 5-factor resolution V design, and doesn't always return the smallest design available.
FRACTIONAL_GENERATORS: dict[int, list[tuple[int, str]]] = {
    3: [(3, "a b ab")],
    4: [(4, "a b c abc")],
    5: [(3, "a b c ab ac"), (5, "a b c d abcd")],
    6: [(3, "a b c ab ac bc"), (4, "a b c d abc bcd"), (6, "a b c d e abcde")],
    7: [
        (3, "a b c ab ac bc abc"),
        (4, "a b c d abc bcd acd"),
        (7, "a b c d e f abcdef"),
    ],
    8: [
        (4, "a b c d bcd acd abc abd"),
        (5, "a b c d e f abcd abef"),
    ],
    9: [
        (3, "a b c d abc bcd acd abd abcd"),
        (4, "a b c d e bcde acde abde abce"),
        (6, "a b c d e f g acdfg bcefg"),
    ],
    10: [
        (3, "a b c d abc bcd acd abd abcd ab"),
        (4, "a b c d e abcd abce abde acde bcde"),
        (5, "a b c d e f g abcg bcde acdf"),
    ],
    11: [
        (3, "a b c d abc bcd acd abd abcd ab ac"),
        (4, "a b c d e abc bcd cde acd ade bde"),
    ],
}


def resolution_of(generators: str) -> int:
    """Calculate a fractional factorial's resolution from its generators.

    The resolution is the length of the shortest word in the defining relation. It
    says what is confounded: at resolution III main effects are mixed with two-factor
    interactions, at IV two-factor interactions are mixed with each other, and at V
    main effects and two-factor interactions are all clear of each other.
    """
    tokens = [token.lstrip("+-") for token in generators.split()]
    words = []
    for index, token in enumerate(tokens):
        if len(token) > 1:
            words.append(frozenset(token) | {f"new{index}"})
    if not words:
        return len(tokens) + 1  # a full factorial has no defining relation
    shortest = len(tokens) + 1
    for size in range(1, len(words) + 1):
        for chosen in itertools.combinations(words, size):
            combined: frozenset = frozenset()
            for word in chosen:
                combined = combined ^ word
            shortest = min(shortest, len(combined))
    return shortest


def require_process_only(study: Study, design_name: str) -> None:
    if study.mixture_factors:
        raise StudyError(
            f"A {design_name} design can't handle mixture components, because they must add up "
            "to a fixed total. Use simplex_lattice, simplex_centroid or extreme_vertices for a "
            "mixture alone, or d_optimal or i_optimal for a mixture with process factors."
        )


def coded_to_frame(coded: np.ndarray, factors: list[Factor], design_name: str) -> pd.DataFrame:
    """Convert a coded matrix (-1 to +1) into actual factor settings."""
    data = {}
    for column, factor in enumerate(factors):
        values = coded[:, column]
        if factor.kind == "categorical":
            if len(factor.levels) != 2:
                raise StudyError(
                    f"A {design_name} design only has two settings per factor, but '{factor.name}' "
                    f"has {len(factor.levels)} levels. Use full_factorial, taguchi, gsd or d_optimal."
                )
            data[factor.name] = np.array(factor.levels, dtype=object)[(values > 0).astype(int)]
        else:
            actual = factor.decode(values)
            if factor.kind == "discrete":
                snapped = factor.nearest_level(actual)
                if not np.allclose(snapped, actual):
                    raise StudyError(
                        f"A {design_name} design needs '{factor.name}' at settings that aren't among "
                        f"its allowed levels {list(factor.levels)}. Use d_optimal instead."
                    )
                actual = snapped
            data[factor.name] = actual
    return pd.DataFrame(data)


def center_rows(factors: list[Factor], count: int) -> pd.DataFrame:
    """Build centre points, cycling through the levels of any categorical factors."""
    if count <= 0:
        return pd.DataFrame(columns=[factor.name for factor in factors])
    categorical = [factor for factor in factors if factor.kind == "categorical"]
    combinations = list(itertools.product(*[factor.levels for factor in categorical])) or [()]
    rows = []
    for index in range(count):
        chosen = dict(zip([f.name for f in categorical], combinations[index % len(combinations)]))
        row = {}
        for factor in factors:
            if factor.kind == "categorical":
                row[factor.name] = chosen[factor.name]
            elif factor.kind == "discrete":
                row[factor.name] = float(factor.nearest_level(factor.center))
            else:
                row[factor.name] = factor.center
        rows.append(row)
    return pd.DataFrame(rows)


def add_center_points(frame: pd.DataFrame, factors: list[Factor], count: int) -> pd.DataFrame:
    if count <= 0:
        return frame
    return pd.concat([frame, center_rows(factors, count)], ignore_index=True)


def factor_settings(factor: Factor, levels, design_name: str) -> list:
    """List the settings a factor takes in a multi-level design."""
    if factor.kind in ("categorical", "discrete"):
        return list(factor.levels)
    count = levels.get(factor.name, 2) if isinstance(levels, dict) else int(levels)
    if count < 2:
        raise StudyError(f"Factor '{factor.name}' needs at least 2 levels in a {design_name} design.")
    return np.linspace(float(factor.low), float(factor.high), count).tolist()


MAX_FACTORIAL_RUNS = 5000


def _check_size(settings: dict[str, list], design_name: str, reduction: int = 1) -> None:
    runs = int(np.prod([float(len(values)) for values in settings.values()])) // reduction
    if runs > MAX_FACTORIAL_RUNS:
        raise StudyError(
            f"This {design_name} design would need about {runs:,} runs. Use a fractional_factorial or "
            "plackett_burman design to screen factors, or an optimal design with a set number of runs."
        )


def _indices_to_frame(indices: np.ndarray, settings: dict[str, list]) -> pd.DataFrame:
    data = {}
    for column, (name, values) in enumerate(settings.items()):
        data[name] = np.array(values, dtype=object)[indices[:, column].astype(int)]
    frame = pd.DataFrame(data)
    for name, values in settings.items():
        if not isinstance(values[0], str):
            frame[name] = frame[name].astype(float)
    return frame


def full_factorial(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Every combination of every factor's levels."""
    require_process_only(study, "full factorial")
    factors = study.factors
    settings = {f.name: factor_settings(f, options.get("levels", 2), "full factorial") for f in factors}
    _check_size(settings, "full factorial")
    indices = pyDOE3.fullfact([len(values) for values in settings.values()])
    frame = _indices_to_frame(indices, settings)
    return add_center_points(frame, factors, int(options.get("center_points", 0)))


def gsd(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A balanced fraction of a multi-level factorial (generalized subset design)."""
    require_process_only(study, "generalized subset")
    factors = study.factors
    settings = {f.name: factor_settings(f, options.get("levels", 2), "generalized subset") for f in factors}
    counts = [len(values) for values in settings.values()]
    reduction = int(options.get("reduction", 2))
    if reduction < 2:
        raise StudyError("A gsd design needs a reduction of at least 2.")
    _check_size(settings, "generalized subset", reduction)
    try:
        indices = np.asarray(pyDOE3.gsd(counts, reduction))
    except ValueError as error:
        raise StudyError(f"Couldn't build this gsd design: {error}") from error
    frame = _indices_to_frame(indices, settings)
    return add_center_points(frame, factors, int(options.get("center_points", 0)))


def _generators_for(count: int, resolution: int) -> tuple[str, int]:
    """Find the smallest tabulated design with at least the requested resolution."""
    available = FRACTIONAL_GENERATORS.get(count)
    if available is None:
        raise StudyError(
            f"The built-in fractional factorials cover 3 to 11 factors, and this study has {count}. "
            "Use plackett_burman, or write your own 'generators'."
        )
    for table_resolution, generators in available:
        if table_resolution >= resolution:
            return generators, table_resolution
    best = max(res for res, _ in available)
    raise StudyError(
        f"There's no built-in resolution {resolution} design for {count} factors (the highest is "
        f"{best}). Ask for resolution {best}, or use a full_factorial or d_optimal design."
    )


def _alias_summary(design: np.ndarray, factors: list[Factor]) -> list[str]:
    """Describe which main effects and two-factor interactions can't be told apart."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    legend = dict(zip(letters, [factor.name for factor in factors]))
    aliases, _ = pyDOE3.fracfact_aliasing(design)
    lines = []
    for alias in aliases:
        short = [word for word in alias.split(" = ") if len(word) <= 2]
        if len(short) >= 2:
            lines.append(" = ".join(":".join(legend[letter] for letter in word) for word in short))
    return lines


def fractional_factorial(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A carefully chosen fraction of a two-level factorial."""
    require_process_only(study, "fractional factorial")
    factors = study.factors
    count = len(factors)
    if options.get("generators"):
        generators = str(options["generators"])
        if len(generators.split()) != count:
            raise StudyError(
                f"The generators '{generators}' define {len(generators.split())} columns, "
                f"but this study has {count} factors."
            )
    else:
        generators, _ = _generators_for(count, int(options.get("resolution", 4)))
    resolution = resolution_of(generators)
    design = pyDOE3.fracfact(generators)
    notes.append(f"Resolution {resolution}, with generators: {generators}")
    aliases = _alias_summary(design, factors)
    if aliases:
        notes.append(
            "Effects that can't be told apart in this design (three-way and higher are not listed): "
            + "; ".join(aliases)
        )
    frame = coded_to_frame(design, factors, "fractional factorial")
    return add_center_points(frame, factors, int(options.get("center_points", 0)))


def plackett_burman(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A very economical two-level screening design, for main effects only."""
    require_process_only(study, "Plackett-Burman")
    factors = study.factors
    if len(factors) < 2:
        raise StudyError("A Plackett-Burman design needs at least 2 factors.")
    notes.append(
        "Plackett-Burman designs estimate main effects only. Each main effect is partly mixed "
        "with many two-factor interactions, so follow up on the important factors."
    )
    frame = coded_to_frame(pyDOE3.pbdesign(len(factors)).astype(float), factors, "Plackett-Burman")
    return add_center_points(frame, factors, int(options.get("center_points", 0)))


def _column_level_counts(array: np.ndarray) -> list[int]:
    return [len(np.unique(array[:, column])) for column in range(array.shape[1])]


def _assign_columns(factors: list[Factor], counts: list[int], default_levels: int) -> dict[str, int] | None:
    """Give each factor a column of an orthogonal array with the right number of levels."""
    free = list(range(len(counts)))
    assignment = {}
    # Assign the most restricted factors first, then the ones that can use any level count.
    for factor in sorted(factors, key=lambda f: f.kind == "continuous"):
        wanted = default_levels if factor.kind == "continuous" else len(factor.levels)
        match = next((column for column in free if counts[column] == wanted), None)
        if match is None:
            return None
        assignment[factor.name] = match
        free.remove(match)
    return assignment


def taguchi(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A Taguchi orthogonal array, chosen automatically unless ``array`` is given."""
    require_process_only(study, "Taguchi")
    factors = study.factors
    default_levels = int(options.get("levels", 3))
    names = pyDOE3.list_orthogonal_arrays()
    if options.get("array"):
        wanted = str(options["array"])
        names = [name for name in names if name == wanted or name.split("(")[0] == wanted]
        if not names:
            raise StudyError(
                f"Unknown orthogonal array '{wanted}'. Available: "
                f"{', '.join(pyDOE3.list_orthogonal_arrays())}."
            )
    for name in sorted(names, key=lambda n: int(n[1:].split("(")[0])):
        array = np.asarray(pyDOE3.get_orthogonal_array(name))
        assignment = _assign_columns(factors, _column_level_counts(array), default_levels)
        if assignment:
            settings = {}
            for factor in factors:
                if factor.kind == "continuous":
                    settings[factor.name] = np.linspace(float(factor.low), float(factor.high), default_levels).tolist()
                else:
                    settings[factor.name] = list(factor.levels)
            indices = array[:, [assignment[factor.name] for factor in factors]]
            notes.append(
                f"Orthogonal array {name}. It estimates main effects (and curvature at 3 or more "
                "levels) but not interactions, so they can distort the results if they are large."
            )
            return _indices_to_frame(indices, settings)
    raise StudyError(
        f"No orthogonal array fits these factors with {default_levels} levels each. Check that "
        "categorical and discrete factors have matching level counts, or use gsd or d_optimal."
    )
