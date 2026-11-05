"""Mixture designs, for blends whose components add up to a fixed total.

Changing one component has to change another, so ordinary factorial thinking doesn't
apply. These designs place runs across the simplex, the triangle (for 3 components)
or tetrahedron (for 4) of all possible blends.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from ..study import Study, StudyError, mixture_vertices


def lattice_points(components: int, degree: int) -> np.ndarray:
    """Every blend whose proportions are multiples of 1/degree: the {q, m} simplex lattice."""
    points = []
    for chosen in itertools.combinations_with_replacement(range(components), degree):
        points.append(np.bincount(chosen, minlength=components) / degree)
    return np.array(points)


def centroid_points(components: int) -> np.ndarray:
    """All pure components, equal 2-way blends, equal 3-way blends, and so on."""
    points = []
    for size in range(1, components + 1):
        for subset in itertools.combinations(range(components), size):
            point = np.zeros(components)
            point[list(subset)] = 1 / size
            points.append(point)
    return np.array(points)


def _require_mixture(study: Study, design_name: str) -> None:
    if not study.mixture_factors:
        raise StudyError(f"A {design_name} design needs factors of type: mixture.")
    if study.process_factors:
        raise StudyError(
            f"A {design_name} design covers mixture components only, but this study also has "
            f"{', '.join(f.name for f in study.process_factors)}. Use i_optimal or d_optimal "
            "(recommended), or crossed."
        )


def _from_pseudo(study: Study, pseudo: np.ndarray, design_name: str) -> pd.DataFrame:
    """Convert pseudo-components (0 to 1) to actual amounts, and check the upper limits."""
    lower, upper = study.mixture_bounds()
    actual = lower + pseudo * study.pseudo_range
    if (actual > upper + 1e-9).any():
        raise StudyError(
            f"A {design_name} design needs blends beyond your upper limits, because those limits "
            "cut off corners of the simplex. Use extreme_vertices or i_optimal instead."
        )
    return pd.DataFrame(actual, columns=study.mixture_names)


def _check_points_pseudo(pseudo: np.ndarray, components: int) -> np.ndarray:
    """The overall centroid, plus points halfway from it to each pure component."""
    centroid = np.full(components, 1 / components)
    axial = (np.eye(components) + centroid) / 2
    return np.vstack([pseudo, centroid, axial])


def simplex_lattice(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A regular grid of blends over the simplex."""
    _require_mixture(study, "simplex lattice")
    components = len(study.mixture_names)
    degree = int(options.get("degree", 2))
    if degree < 1:
        raise StudyError("A simplex lattice needs a degree of at least 1.")
    pseudo = lattice_points(components, degree)
    if options.get("check_points", False):
        pseudo = _check_points_pseudo(pseudo, components)
    return _from_pseudo(study, pseudo, "simplex lattice")


def simplex_centroid(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Pure components and every equal-parts blend, which supports a special cubic model."""
    _require_mixture(study, "simplex centroid")
    components = len(study.mixture_names)
    pseudo = centroid_points(components)
    if options.get("check_points", False):
        pseudo = _check_points_pseudo(pseudo, components)
    return _from_pseudo(study, pseudo, "simplex centroid")


def edge_midpoints(vertices: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Find points halfway along each edge of a constrained mixture region.

    Two corners share an edge when at least (components - 2) components sit on the
    same bound at both corners.
    """
    if len(vertices) < 2:
        return np.empty((0, vertices.shape[1]))
    required = vertices.shape[1] - 2
    at_lower = np.isclose(vertices, lower)
    at_upper = np.isclose(vertices, upper)
    midpoints = []
    for i, j in itertools.combinations(range(len(vertices)), 2):
        shared = (at_lower[i] & at_lower[j]) | (at_upper[i] & at_upper[j])
        if shared.sum() >= required:
            midpoints.append((vertices[i] + vertices[j]) / 2)
    return np.array(midpoints) if midpoints else np.empty((0, vertices.shape[1]))


def extreme_vertices(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Corners, edge midpoints and the centre of a region cut down by component limits."""
    _require_mixture(study, "extreme vertices")
    lower, upper = study.mixture_bounds()
    vertices = mixture_vertices(lower, upper, float(study.mixture_total))
    points = [vertices]
    if options.get("edge_midpoints", True):
        points.append(edge_midpoints(vertices, lower, upper))
    centroid = vertices.mean(axis=0)
    points.append(centroid[None, :])
    if options.get("check_points", False):
        points.append((vertices + centroid) / 2)
    return pd.DataFrame(np.vstack(points), columns=study.mixture_names)


def mixture_candidates(study: Study, lattice_limit: int = 1500) -> pd.DataFrame:
    """A dense set of allowed blends for an optimal design to choose from."""
    lower, upper = study.mixture_bounds()
    total = float(study.mixture_total)
    components = len(lower)
    vertices = mixture_vertices(lower, upper, total)

    # Use the finest lattice within the size limit, spread over the allowed region.
    degree = 2
    while degree < 20 and math.comb(components + degree, degree + 1) <= lattice_limit:
        degree += 1
    grid = lower + lattice_points(components, degree) * study.pseudo_range
    grid = grid[(grid <= upper + 1e-9).all(axis=1)]

    points = np.vstack([vertices, edge_midpoints(vertices, lower, upper), vertices.mean(axis=0)[None, :], grid])
    frame = pd.DataFrame(np.round(points, 10), columns=study.mixture_names).drop_duplicates()
    return frame.reset_index(drop=True)
