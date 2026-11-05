"""Recommend designs for a study, and explain why.

The rules here follow the "Choosing a design" section of docs/manual.md. The app and the
scripts both use them, so all three give the same advice. Every suggestion is actually
built and evaluated, so the run counts and quality figures are real.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .designs import DESIGNS, create_design
from .evaluate import DesignQuality, evaluate_design
from .model import model_terms
from .study import Study, StudyError

OBJECTIVES = {
    "screen": "Screening: I have many factors and want to find the few that matter",
    "optimise": "Optimising: I know the key factors and want the best settings",
    "explore": "Exploring: behaviour is unknown, or the data is for machine learning",
}


@dataclass
class Recommendation:
    kind: str
    title: str
    design: dict
    model: dict | None
    why: str
    caution: str
    runs: int
    quality: DesignQuality
    within_budget: bool

    @property
    def settings(self) -> dict:
        """The settings to put in a config file."""
        return self.design


def _limits_cut_simplex(study: Study) -> bool:
    """Do the upper limits rule out any corner of the (lower-limit) simplex?"""
    lower, upper = study.mixture_bounds()
    return bool(((lower + study.pseudo_range) > upper + 1e-9).any())


def _mixture_suggestions(study: Study, objective: str, max_runs: int | None) -> list[dict]:
    has_process = bool(study.process_factors)
    linear = {"mixture": "linear", "process": "linear", "mixture_process": "none"}
    budget = {"runs": max_runs} if max_runs else {}

    if objective == "screen":
        return [
            dict(
                design={"type": "d_optimal", **budget},
                model=linear,
                why="Fits a simple straight-line model, which uses the fewest runs and shows "
                "which components and process factors have the largest effects.",
                caution="A straight-line model can't locate an optimum. Follow up with an "
                "I-optimal design on the factors that matter.",
            )
        ]
    if has_process:
        suggestions = [
            dict(
                design={"type": "i_optimal", **budget},
                model=None,
                why="Composition and process factors together rule out classical designs, because "
                "changing one component changes another. I-optimal chooses runs for the most "
                "precise predictions, which is what finding the best settings depends on.",
                caution="The design depends on the model you specify, so check that it's rich "
                "enough before you start.",
            ),
            dict(
                design={"type": "d_optimal", **budget},
                model=None,
                why="Handles the same situation, but aims for the most precise coefficients. "
                "Prefer it if understanding each effect matters more than prediction.",
                caution="It concentrates runs on the edges of the region, so predictions in the "
                "middle are a little less precise than with I-optimal.",
            ),
        ]
        numeric_process = len(study.numeric_process_names)
        if not study.constraints and len(study.process_factors) <= 3 and numeric_process == len(study.process_factors):
            mixture_design = {"type": "extreme_vertices"} if _limits_cut_simplex(study) else {"type": "simplex_centroid"}
            suggestions.append(
                dict(
                    design={
                        "type": "crossed",
                        "mixture": mixture_design,
                        "process": {"type": "full_factorial", "levels": 2},
                    },
                    model={"process": "interaction"},
                    why="Runs every blend at every process setting, so it's easy to explain and "
                    "doesn't depend on a particular model.",
                    caution="The number of runs multiplies quickly, and two process levels can't show curvature.",
                )
            )
        return suggestions

    if study.constraints or _limits_cut_simplex(study):
        suggestions = [
            dict(
                design={"type": "i_optimal", **budget},
                model=None,
                why="Your limits cut corners off the blend region, so standard lattices no longer "
                "fit. I-optimal places runs for the best predictions within what's allowed.",
                caution="Check that the model is rich enough before you start.",
            )
        ]
        if not study.constraints:
            suggestions.append(
                dict(
                    design={"type": "extreme_vertices", "check_points": True, "replicate_runs": 3},
                    model=None,
                    why="Uses the corners and edges of the allowed region, plus check points, so it "
                    "covers the boundary thoroughly and doesn't depend on a particular model.",
                    caution="With many components, the number of corners can grow quickly.",
                )
            )
        return suggestions
    return [
        dict(
            design={"type": "simplex_centroid", "check_points": True, "replicate_runs": 3},
            model=None,
            why="The standard design for blends with no tight limits. Pure components, equal-parts "
            "blends and check points support a model that captures blending effects.",
            caution="Pure components can be impractical for hydrogels. If so, set lower limits on the components.",
        ),
        dict(
            design={"type": "simplex_lattice", "degree": 2, "check_points": True, "replicate_runs": 3},
            model={"mixture": "quadratic"},
            why="A regular grid of blends that fits a quadratic blending model, with a few check "
            "points to test whether that model is adequate.",
            caution="Pure components can be impractical for hydrogels. If so, set lower limits on the components.",
        ),
    ]


def _screening_suggestions(study: Study, max_runs: int | None) -> list[dict]:
    count = len(study.factors)
    many_levels = any(f.kind == "categorical" and len(f.levels) > 2 for f in study.factors)
    budget = {"runs": max_runs} if max_runs else {}
    suggestions = []
    if study.constraints or many_levels:
        reason = "constraints" if study.constraints else "a categorical factor with more than two levels"
        suggestions.append(
            dict(
                design={"type": "d_optimal", **budget},
                model={"process": "linear"},
                why=f"Because of {reason}, two-level screening designs don't fit. D-optimal picks the "
                "most informative runs for a main-effects model.",
                caution="Interactions aren't estimated, so follow up on the factors that matter.",
            )
        )
        if many_levels and not study.constraints:
            suggestions.append(
                dict(
                    design={"type": "gsd", "levels": 2, "reduction": 2},
                    model={"process": "linear"},
                    why="A balanced fraction of all the level combinations, which suits several "
                    "categorical factors.",
                    caution="Interactions aren't estimated, so follow up on the factors that matter.",
                )
            )
        return suggestions

    if count <= 4:
        suggestions.append(
            dict(
                design={"type": "full_factorial", "levels": 2, "center_points": 3},
                model={"process": "interaction"},
                why="With this few factors you can afford every combination, which measures all main "
                "effects and interactions. Centre points measure noise and reveal curvature.",
                caution="Centre points need numeric factors to be meaningful.",
            )
        )
    if 4 <= count <= 11:
        suggestions.append(
            dict(
                design={"type": "fractional_factorial", "resolution": 4, "center_points": 3},
                model={"process": "linear"},
                why="Halves the work or better. Main effects stay clear of two-factor interactions, "
                "although some interactions stay mixed with each other.",
                caution="Interactions can't all be separated, so a follow-up may be needed.",
            )
        )
    if 5 <= count <= 8:
        suggestions.append(
            dict(
                design={"type": "fractional_factorial", "resolution": 5, "center_points": 3},
                model={"process": "interaction"},
                why="Main effects and two-factor interactions are all clear of each other, so "
                "you learn nearly as much as from a full factorial.",
                caution="Costs more runs than the resolution IV version.",
            )
        )
    if count >= 6:
        suggestions.append(
            dict(
                design={"type": "plackett_burman", "center_points": 3},
                model={"process": "linear"},
                why="The fewest runs for this many factors. Good for a first pass, when you expect "
                "only a few factors to matter.",
                caution="Main effects are partly mixed with many interactions, so confirm the "
                "results with a follow-up experiment.",
            )
        )
    return suggestions


def _optimising_suggestions(study: Study, max_runs: int | None, avoid_extremes: bool) -> list[dict]:
    count = len(study.factors)
    budget = {"runs": max_runs} if max_runs else {}
    has_categorical = bool(study.categorical_names)
    optimal = dict(
        design={"type": "i_optimal", **budget},
        model=None,
        why="Chooses runs for the most precise predictions across the region, and adapts to any "
        "run budget, constraint or factor type.",
        caution="The design depends on the model you specify, so check that it's rich enough before you start.",
    )
    if study.constraints or has_categorical:
        reason = "constraints" if study.constraints else "categorical factors"
        optimal["why"] = f"Because of your {reason}, classical response surface designs don't fit. " + optimal["why"]
        return [
            optimal,
            dict(
                design={"type": "d_optimal", **budget},
                model=None,
                why="Handles the same situation, but aims for the most precise coefficients rather than predictions.",
                caution="It concentrates runs on the edges of the region.",
            ),
        ]
    if count == 1:
        return [
            dict(
                design={"type": "full_factorial", "levels": 5, "repeats": 2},
                model={"process": "quadratic"},
                why="With one factor, five evenly spaced levels run twice show the shape of the "
                "curve and measure noise.",
                caution="If other factors could matter, vary them together rather than one at a time.",
            )
        ]

    box_behnken = dict(
        design={"type": "box_behnken", "center_points": max(3, count)},
        model=None,
        why="Three levels per factor, and never all factors at their extremes together. A good "
        "fit for bioprinting and other processes where combined extremes fail.",
        caution="Says less about the corners of the region, so don't predict there.",
    )
    central_composite = dict(
        design={"type": "central_composite", "variant": "face_centred", "center_points": 4},
        model=None,
        why="The standard response surface design. The face-centred version stays inside your "
        "limits and includes the corners, so it describes the whole region.",
        caution="Includes runs with every factor at an extreme, so check those are physically possible.",
    )
    suggestions = []
    if count == 2:
        suggestions = [
            central_composite,
            dict(
                design={"type": "full_factorial", "levels": 3, "repeats": 1, "replicate_runs": 3},
                model={"process": "quadratic"},
                why="All nine combinations of three levels, which is simple to run and explain.",
                caution="A few more runs than strictly necessary for a quadratic model.",
            ),
        ]
    elif count <= 5:
        suggestions = [box_behnken, central_composite] if avoid_extremes else [central_composite, box_behnken]
    else:
        screening_note = (
            " With this many factors, consider screening first. Reducing to 3 to 5 factors usually "
            "makes optimising far cheaper."
        )
        optimal["why"] += screening_note
        suggestions = [optimal]
        return suggestions
    suggestions.append(optimal)
    if count >= 2:
        suggestions.append(
            dict(
                design={"type": "doehlert", "center_points": 3},
                model=None,
                why="Evenly spread points that are easy to extend, if the optimum turns out to be "
                "near an edge of the region.",
                caution="Factors get different numbers of levels, so put the most important one first.",
            )
        )
    return suggestions


def _exploring_suggestions(study: Study, max_runs: int | None) -> list[dict]:
    if study.mixture_factors:
        return _mixture_suggestions(study, "optimise", max_runs)
    budget = {"runs": max_runs} if max_runs else {}
    return [
        dict(
            design={"type": "latin_hypercube", **budget},
            model={"process": "quadratic"},
            why="Spreads runs evenly and assumes no model, so unexpected behaviour can still show up. "
            "It suits a machine learning model fitted afterwards.",
            caution="Less efficient than a response surface design if a quadratic model turns out to be enough.",
        ),
        dict(
            design={"type": "sobol", **budget},
            model={"process": "quadratic"},
            why="Also evenly spread, and more runs can be added later without losing that evenness.",
            caution="Less efficient than a response surface design if a quadratic model turns out to be enough.",
        ),
    ]


def recommend_designs(
    study: Study,
    objective: str = "optimise",
    max_runs: int | None = None,
    avoid_extremes: bool = False,
    signal_to_noise: float = 2.0,
) -> list[Recommendation]:
    """Suggest designs for a study, best first, with real run counts and quality figures."""
    if objective not in OBJECTIVES:
        raise StudyError(f"Unknown objective '{objective}'. Choose from: {', '.join(OBJECTIVES)}.")
    if study.mixture_factors:
        suggestions = _mixture_suggestions(study, objective, max_runs)
        if objective == "explore":
            suggestions = _exploring_suggestions(study, max_runs)
    elif objective == "screen":
        suggestions = _screening_suggestions(study, max_runs)
    elif objective == "optimise":
        suggestions = _optimising_suggestions(study, max_runs, avoid_extremes)
    else:
        suggestions = _exploring_suggestions(study, max_runs)

    results: list[Recommendation] = []
    for suggestion in suggestions:
        model = suggestion["model"] if suggestion["model"] is not None else study.model
        trial = replace(study, design=dict(suggestion["design"]), model=model)
        try:
            design = create_design(trial)
            quality = evaluate_design(trial, design.frame, model_terms(trial, design.frame), signal_to_noise)
        except StudyError:
            continue  # this design doesn't suit these factors, so don't suggest it
        if not quality.estimable:
            continue
        kind = suggestion["design"]["type"]
        results.append(
            Recommendation(
                kind=kind,
                title=DESIGNS[kind].title,
                design=suggestion["design"],
                model=suggestion["model"],
                why=suggestion["why"],
                caution=suggestion["caution"],
                runs=len(design.frame),
                quality=quality,
                within_budget=max_runs is None or len(design.frame) <= max_runs,
            )
        )
    return sorted(results, key=lambda item: not item.within_budget)
