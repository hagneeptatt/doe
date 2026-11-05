"""Create designs: a registry of every design type, and the finishing steps they share."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..model import Term, model_matrix, model_terms
from ..study import Study, StudyError
from . import factorial, mixture, optimal, response_surface, space_filling

COMMON_OPTIONS = {"type", "repeats", "replicate_runs", "blocks", "randomise"}


@dataclass
class Design:
    """A finished design, ready to become a run sheet."""

    frame: pd.DataFrame
    terms: list[Term]
    kind: str
    seed: int
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DesignType:
    name: str
    build: Callable
    title: str
    purpose: str
    summary: str
    options: frozenset = frozenset()


def _crossed(study: Study, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Run every blend at every process setting, by crossing two separate designs."""
    if not study.mixture_factors or not study.process_factors:
        raise StudyError("A crossed design needs both mixture components and process factors.")
    parts = []
    for section, names in (("mixture", study.mixture_names), ("process", [f.name for f in study.process_factors])):
        settings = dict(options.get(section) or {})
        kind = settings.get("type")
        if kind is None:
            raise StudyError(f"A crossed design needs a '{section}' section with its own design type.")
        if kind not in DESIGNS or kind == "crossed":
            raise StudyError(f"Unknown {section} design type '{kind}'.")
        parts.append(DESIGNS[kind].build(study.only(names), settings, rng, notes))
    return parts[0].merge(parts[1], how="cross")


_ENTRIES = [
    DesignType(
        "full_factorial", factorial.full_factorial, "Full factorial", "screening",
        "Every combination of levels. Measures all interactions, but grows quickly with factors.",
        frozenset({"levels", "center_points"}),
    ),
    DesignType(
        "fractional_factorial", factorial.fractional_factorial, "Fractional factorial", "screening",
        "A chosen fraction of a two-level factorial. Saves runs by mixing up higher-order interactions.",
        frozenset({"resolution", "generators", "center_points"}),
    ),
    DesignType(
        "plackett_burman", factorial.plackett_burman, "Plackett-Burman", "screening",
        "Very few runs for many factors. Main effects only.",
        frozenset({"center_points"}),
    ),
    DesignType(
        "taguchi", factorial.taguchi, "Taguchi orthogonal array", "screening",
        "A balanced multi-level array. Main effects only, with interactions assumed small.",
        frozenset({"array", "levels"}),
    ),
    DesignType(
        "gsd", factorial.gsd, "Generalized subset", "screening",
        "A balanced fraction of a multi-level factorial, useful with several categorical factors.",
        frozenset({"levels", "reduction", "center_points"}),
    ),
    DesignType(
        "box_behnken", response_surface.box_behnken, "Box-Behnken", "response surface",
        "Three levels per factor with no extreme corners. Good when all-high or all-low is risky.",
        frozenset({"center_points"}),
    ),
    DesignType(
        "central_composite", response_surface.central_composite, "Central composite", "response surface",
        "A factorial core plus star points. The standard way to fit a quadratic model.",
        frozenset({"variant", "alpha", "center_points"}),
    ),
    DesignType(
        "doehlert", response_surface.doehlert, "Doehlert", "response surface",
        "Evenly spread points that are easy to extend towards a promising direction.",
        frozenset({"center_points"}),
    ),
    DesignType(
        "latin_hypercube", space_filling.latin_hypercube, "Latin hypercube", "space-filling",
        "Runs spread evenly with no model assumed. Good for unknown behaviour or machine learning.",
        frozenset({"runs"}),
    ),
    DesignType(
        "sobol", space_filling.sobol, "Sobol sequence", "space-filling",
        "An evenly spreading sequence that can be extended with more runs later.",
        frozenset({"runs"}),
    ),
    DesignType(
        "simplex_lattice", mixture.simplex_lattice, "Simplex lattice", "mixture",
        "A regular grid of blends. For mixtures with few or no limits on the components.",
        frozenset({"degree", "check_points"}),
    ),
    DesignType(
        "simplex_centroid", mixture.simplex_centroid, "Simplex centroid", "mixture",
        "Pure components and every equal-parts blend. Supports a special cubic model.",
        frozenset({"check_points"}),
    ),
    DesignType(
        "extreme_vertices", mixture.extreme_vertices, "Extreme vertices", "mixture",
        "Corners and edges of a blend region cut down by component limits.",
        frozenset({"edge_midpoints", "check_points"}),
    ),
    DesignType(
        "crossed", _crossed, "Crossed mixture-process", "mixture and process",
        "Every blend at every process setting. Thorough, but needs many runs.",
        frozenset({"mixture", "process"}),
    ),
    DesignType(
        "d_optimal", optimal.d_optimal, "D-optimal", "optimal",
        "Computer-chosen runs for the most precise coefficients. Handles any factors and budget.",
        frozenset({"runs", "candidate_levels", "max_candidates", "starts"}),
    ),
    DesignType(
        "i_optimal", optimal.i_optimal, "I-optimal", "optimal",
        "Computer-chosen runs for the most precise predictions. The best general choice for optimising.",
        frozenset({"runs", "candidate_levels", "max_candidates", "starts"}),
    ),
]  # fmt: skip

DESIGNS: dict[str, DesignType] = {entry.name: entry for entry in _ENTRIES}


def _decimals(span: float) -> int:
    """Choose a sensible number of decimals for a factor with this range."""
    return max(0, 3 - math.floor(math.log10(span))) if span > 0 else 4


def round_frame(study: Study, frame: pd.DataFrame) -> pd.DataFrame:
    """Round settings to values you could dial in, keeping every blend on its total."""
    out = frame.copy()
    for factor in study.process_factors:
        if factor.is_numeric:
            out[factor.name] = out[factor.name].astype(float).round(_decimals(float(factor.high) - float(factor.low)))
    if study.mixture_factors:
        names = study.mixture_names
        lower, upper = study.mixture_bounds()
        decimals = _decimals(study.pseudo_range)
        values = np.round(out[names].to_numpy(dtype=float), decimals)
        total = float(study.mixture_total)
        for row in values:
            gap = round(total - row.sum(), decimals + 2)
            if gap:
                # Put the rounding error on the component with the most room to take it.
                room = upper - row if gap > 0 else row - lower
                row[np.argmax(room)] += gap
        out[names] = np.round(values, decimals + 2)
    return out


def _replicate_runs(study: Study, frame: pd.DataFrame, terms: list[Term], count: int) -> pd.DataFrame:
    """Add replicates of the most influential distinct runs.

    Replicates measure pure experimental noise. Repeating the highest-leverage points
    also makes the model less dependent on any single measurement.
    """
    if count <= 0:
        return frame
    matrix = model_matrix(study, frame, terms, include_blocks=False).values
    leverage = np.sum((matrix @ np.linalg.pinv(matrix.T @ matrix)) * matrix, axis=1)
    order = pd.Series(leverage).sort_values(ascending=False, kind="stable").index
    distinct = frame.drop_duplicates(subset=study.factor_names).index
    ranked = [index for index in order if index in distinct]
    picks = [ranked[i % len(ranked)] for i in range(count)]
    return pd.concat([frame, frame.iloc[picks]], ignore_index=True)


def assign_blocks(study: Study, frame: pd.DataFrame, terms: list[Term], blocks: int, rng) -> np.ndarray:
    """Divide runs into equal blocks so the block differences disturb the model least.

    A block is a group of runs done together, such as one day or one batch of material.
    Pairs of runs are swapped between blocks for as long as that improves the determinant
    of the model including block effects.
    """
    runs = len(frame)
    if blocks < 2:
        return np.ones(runs, dtype=int)
    if blocks > runs:
        raise StudyError(f"Can't divide {runs} runs into {blocks} blocks.")
    X = model_matrix(study, frame, terms, include_blocks=False).values

    def score(labels: np.ndarray) -> float:
        dummies = [(labels == b).astype(float) - (labels == blocks).astype(float) for b in range(1, blocks)]
        full = np.column_stack([X, *dummies])
        return np.linalg.slogdet(full.T @ full + 1e-8 * np.eye(full.shape[1]))[1]

    balanced = np.resize(np.arange(1, blocks + 1), runs)
    if runs > 300:
        # The swap search would take too long, so settle for a random balanced split.
        return rng.permutation(balanced)
    best_labels, best_score = balanced, -np.inf
    for _ in range(5 if runs <= 60 else 1):
        labels = rng.permutation(balanced)
        current = score(labels)
        improved = True
        while improved:
            improved = False
            for i in range(runs):
                for j in range(i + 1, runs):
                    if labels[i] == labels[j]:
                        continue
                    labels[i], labels[j] = labels[j], labels[i]
                    trial = score(labels)
                    if trial > current + 1e-9:
                        current, improved = trial, True
                    else:
                        labels[i], labels[j] = labels[j], labels[i]
        if current > best_score:
            best_labels, best_score = labels.copy(), current
    return best_labels


def finalise_design(study: Study, frame: pd.DataFrame, options: dict, rng, notes: list[str]) -> pd.DataFrame:
    """Round, replicate, block and randomise. These steps are the same for every design."""
    frame = round_frame(study, frame[study.factor_names]).reset_index(drop=True)

    infeasible = ~study.is_feasible(frame)
    if infeasible.any():
        raise StudyError(
            f"{int(infeasible.sum())} of the {len(frame)} runs in this design break your "
            "constraints. Classical designs can't work around constraints, so use i_optimal or d_optimal."
        )

    terms = model_terms(study, frame)
    repeats = int(options.get("repeats", 1))
    if repeats < 1:
        raise StudyError("The 'repeats' setting must be at least 1.")
    if repeats > 1:
        frame = pd.concat([frame] * repeats, ignore_index=True)
    frame = _replicate_runs(study, frame, terms, int(options.get("replicate_runs", 0)))
    frame.insert(0, "std_order", np.arange(1, len(frame) + 1))

    blocks = int(options.get("blocks", 1))
    labels = assign_blocks(study, frame, terms, blocks, rng)
    if blocks > 1:
        frame.insert(0, "block", labels)
        notes.append(
            f"Runs are divided into {blocks} blocks. Do each block together, for example one per day "
            "or one per batch of material."
        )

    if options.get("randomise", True):
        order = rng.permutation(len(frame))
        frame = frame.iloc[order].reset_index(drop=True)
    else:
        notes.append(
            "Run order is NOT randomised. Any drift during the experiment could be mistaken for a factor effect."
        )
    if blocks > 1:
        frame = frame.sort_values("block", kind="stable").reset_index(drop=True)
    frame.insert(0, "run", np.arange(1, len(frame) + 1))
    return frame


def create_design(study: Study) -> Design:
    """Generate a study's design from its config."""
    options = dict(study.design)
    kind = options.get("type")
    if not kind:
        raise StudyError(f"The study needs a design type. Choose from: {', '.join(DESIGNS)}.")
    if kind == "existing":
        raise StudyError(
            "This study reads existing data (design type: existing), so there's no design to create. "
            "Go straight to the analysis."
        )
    if kind not in DESIGNS:
        raise StudyError(f"Unknown design type '{kind}'. Choose from: {', '.join(DESIGNS)}.")
    entry = DESIGNS[kind]
    allowed = COMMON_OPTIONS | entry.options
    unknown = set(options) - allowed
    if unknown:
        raise StudyError(
            f"The {kind} design has unknown setting(s): {', '.join(sorted(unknown))}. "
            f"Available settings: {', '.join(sorted(allowed))}."
        )
    # Record the seed even when none was set, so any design can be recreated exactly.
    seed = study.seed if study.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)
    notes: list[str] = []
    frame = entry.build(study, options, rng, notes)
    frame = finalise_design(study, frame, options, rng, notes)
    return Design(frame=frame, terms=model_terms(study, frame), kind=kind, seed=seed, notes=notes)
