"""Response surface designs, for mapping curvature and finding an optimum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyDOE3

from ..study import Study, StudyError
from .factorial import add_center_points, coded_to_frame, require_process_only

CCD_VARIANTS = {
    "face_centred": "ccf",
    "circumscribed": "ccc",
    "inscribed": "cci",
}


def _require_numeric(study: Study, design_name: str) -> None:
    require_process_only(study, design_name)
    if study.categorical_names:
        raise StudyError(
            f"A {design_name} design needs numeric factors, but "
            f"{', '.join(study.categorical_names)} is categorical. Use d_optimal or i_optimal, "
            "or run a separate design for each category."
        )


def box_behnken(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Three levels per factor, with no runs at the extreme corners of the region."""
    _require_numeric(study, "Box-Behnken")
    factors = study.factors
    if len(factors) < 3:
        raise StudyError(
            "A Box-Behnken design needs at least 3 factors. With 2 factors, use "
            "central_composite with the face_centred variant."
        )
    if len(factors) > 12:
        raise StudyError(
            f"A Box-Behnken design for {len(factors)} factors needs too many runs. "
            "Screen out some factors first, or use i_optimal with a set number of runs."
        )
    if len(factors) > 5:
        notes.append(
            f"With {len(factors)} factors this uses the pairwise construction, which needs more "
            "runs than the published Box-Behnken tables. Consider screening out some factors first."
        )
    coded = pyDOE3.bbdesign(len(factors), center=0)
    frame = coded_to_frame(coded, factors, "Box-Behnken")
    return add_center_points(frame, factors, int(options.get("center_points", 3)))


def central_composite(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A factorial core, plus star points that add curvature information."""
    _require_numeric(study, "central composite")
    factors = study.factors
    if len(factors) < 2:
        raise StudyError("A central composite design needs at least 2 factors.")
    if len(factors) > 8:
        raise StudyError(
            f"A central composite design for {len(factors)} factors needs over {2 ** len(factors)} runs. "
            "Screen out some factors first, or use i_optimal with a set number of runs."
        )
    variant = options.get("variant", "face_centred")
    if variant not in CCD_VARIANTS:
        raise StudyError(
            f"Unknown central composite variant '{variant}'. Choose from: {', '.join(CCD_VARIANTS)}."
        )
    alpha = options.get("alpha", "rotatable")
    if alpha not in ("rotatable", "orthogonal"):
        raise StudyError("A central composite 'alpha' must be rotatable or orthogonal.")
    coded = pyDOE3.ccdesign(len(factors), center=(0, 0), alpha=alpha, face=CCD_VARIANTS[variant])
    if variant == "circumscribed":
        reach = float(np.abs(coded).max())
        notes.append(
            f"Star points go beyond your low and high limits, to {reach:.3f} in coded units. Check "
            "that those settings are physically possible, or use the face_centred variant."
        )
    elif variant == "inscribed":
        notes.append("All runs stay within your limits, but the corner runs sit inside them.")
    frame = coded_to_frame(coded, factors, "central composite")
    return add_center_points(frame, factors, int(options.get("center_points", 4)))


def doehlert_matrix(count: int) -> np.ndarray:
    """Build a Doehlert uniform shell from its definition.

    Take a regular simplex with one corner at the origin. The shell is every
    difference between two of its corners, plus the centre, which gives
    count**2 + count + 1 evenly spread points, all at distance 1 from the centre.

    pyDOE3's Doehlert functions aren't used, because their points don't all lie at
    the same distance from the centre.
    """
    simplex = np.zeros((count + 1, count))
    for vertex in range(1, count + 1):
        centroid = simplex[:vertex].mean(axis=0)
        simplex[vertex, : vertex - 1] = centroid[: vertex - 1]
        simplex[vertex, vertex - 1] = np.sqrt(1 - np.sum(simplex[vertex, : vertex - 1] ** 2))
    points = [np.zeros(count)]
    for i in range(count + 1):
        for j in range(count + 1):
            if i != j:
                points.append(simplex[i] - simplex[j])
    return np.array(points)


def doehlert(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """An evenly spread shell of points, which is easy to extend in any direction later."""
    _require_numeric(study, "Doehlert")
    factors = study.factors
    if len(factors) < 2:
        raise StudyError("A Doehlert design needs at least 2 factors.")
    if len(factors) > 10:
        raise StudyError(f"A Doehlert design for {len(factors)} factors needs too many runs. Screen out some factors first.")
    coded = doehlert_matrix(len(factors))
    # Stretch each axis so the design reaches the low and high limits you gave.
    coded = coded / np.abs(coded).max(axis=0)
    notes.append(
        "Doehlert designs give each factor a different number of levels. Put the factor you most "
        f"want detail on first in the config: here that is '{factors[0].name}'."
    )
    frame = coded_to_frame(coded, factors, "Doehlert")
    return add_center_points(frame, factors, int(options.get("center_points", 2)))
