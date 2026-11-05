"""Space-filling designs, which spread runs evenly without assuming a model.

They suit processes with unknown or strongly non-linear behaviour, and data that will
be modelled with something more flexible than a polynomial, such as machine learning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..study import Study, StudyError
from .factorial import require_process_only


def _scale_to_frame(unit: np.ndarray, study: Study, rng) -> pd.DataFrame:
    """Convert points from the unit cube into actual factor settings."""
    data = {}
    for column, factor in enumerate(study.factors):
        values = unit[:, column]
        if factor.kind == "categorical":
            # Share runs evenly between levels, in an order that doesn't track other factors.
            repeats = int(np.ceil(len(values) / len(factor.levels)))
            assigned = np.tile(np.array(factor.levels, dtype=object), repeats)[: len(values)]
            rng.shuffle(assigned)
            data[factor.name] = assigned
        else:
            actual = float(factor.low) + values * (float(factor.high) - float(factor.low))
            data[factor.name] = factor.nearest_level(actual) if factor.kind == "discrete" else actual
    return pd.DataFrame(data)


def _space_filling(study: Study, options: dict, rng, notes: list[str], draw, name: str) -> pd.DataFrame:
    require_process_only(study, name)
    wanted = int(options.get("runs", 10 * len(study.factors)))
    if wanted < 2:
        raise StudyError(f"A {name} design needs at least 2 runs.")
    if "runs" not in options:
        notes.append(f"Using the rule of thumb of 10 runs per factor ({wanted} runs).")
    if not study.constraints:
        return _scale_to_frame(draw(wanted), study, rng)

    # With constraints, draw extra points and keep the ones that satisfy them.
    for attempt in (4, 16, 64):
        frame = _scale_to_frame(draw(wanted * attempt), study, rng)
        frame = frame[study.is_feasible(frame)].reset_index(drop=True)
        if len(frame) >= wanted:
            notes.append("Runs breaking your constraints were dropped, so coverage isn't perfectly even.")
            return frame.iloc[:wanted].copy()
    raise StudyError(f"Couldn't find {wanted} runs that satisfy the constraints. Check that they aren't too strict.")


def latin_hypercube(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Each factor's range is cut into as many slices as runs, and each slice is used once."""
    sampler = qmc.LatinHypercube(d=len(study.factors), rng=rng, optimization="random-cd")
    return _space_filling(study, options, rng, notes, sampler.random, "Latin hypercube")


def sobol(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """A low-discrepancy sequence, which can be extended later and stay evenly spread."""
    sampler = qmc.Sobol(d=len(study.factors), scramble=True, rng=rng)

    def draw(count: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Sobol prefers powers of 2
            return sampler.random(count)

    return _space_filling(study, options, rng, notes, draw, "Sobol")
