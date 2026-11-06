"""Add more runs to an experiment you've already started.

Experimenting in stages is usually the most efficient way to work: learn something, then
decide what to do next. Here, the runs you already have stay fixed, and the optimal design
search chooses the new runs that add the most to them, for the model you want to fit.

By default the new runs form a new block, because later experiments often differ from
earlier ones in small ways, such as a new batch of material. The block effect keeps those
differences from being mistaken for factor effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .designs import round_frame
from .designs.optimal import candidate_points, default_starts, exchange_search
from .evaluate import evaluate_design
from .model import Term, model_matrix, model_terms
from .runsheet import clean_runsheet, notes_as_text
from .study import Study, StudyError

GOALS = {"I": "Better predictions (I-optimal)", "D": "Better estimates of each effect (D-optimal)"}
SPARE_RUNS = 3


@dataclass
class Augmentation:
    """The new runs, the run sheet with them added, and the designs before and after."""

    new_runs: pd.DataFrame
    sheet: pd.DataFrame
    design_before: pd.DataFrame
    design_after: pd.DataFrame
    terms: list[Term]
    criterion: str
    seed: int
    block: int | None
    counted_runs: int
    ignored_runs: int
    notes: list[str] = field(default_factory=list)


def default_extra_runs(minimum: int) -> int:
    """Suggest how many runs to add: enough for the model, with a few spare to check it."""
    return max(minimum + SPARE_RUNS, 4)


def _runs_to_build_on(study: Study, sheet: pd.DataFrame) -> pd.DataFrame:
    """Choose the runs that count as information already gathered.

    A run with no result, such as a failed print, has told you nothing, so the search
    shouldn't rely on it. If nothing has been measured yet, the whole plan counts, which
    is what you'd want when enlarging a design before starting.
    """
    names = [name for name in study.response_names if name in sheet]
    done = sheet[names].notna().any(axis=1) if names else pd.Series(False, index=sheet.index)
    return sheet[done] if done.any() else sheet


def _block_labels(sheet: pd.DataFrame, new_block: bool) -> tuple[pd.Series | None, int | None]:
    """Work out each existing run's block, and the block the new runs belong in."""
    if "block" in sheet:
        labels = pd.to_numeric(sheet["block"], errors="coerce")
        if labels.isna().any():
            rows = (labels[labels.isna()].index + 1).tolist()
            raise StudyError(f"The block column has a blank or non-numeric value in row(s) {rows}.")
        labels = labels.astype(int)
        return labels, int(labels.max()) + (1 if new_block else 0)
    if new_block:
        return pd.Series(1, index=sheet.index), 2
    return None, None


def _region_study(study: Study, region: dict | None) -> Study:
    """Make a copy of the study whose limits cover only the region the new runs may use."""
    if not region:
        return study
    unknown = set(region) - set(study.factor_names)
    if unknown:
        raise StudyError(f"The region names unknown factors: {', '.join(sorted(unknown))}.")
    factors = []
    for factor in study.factors:
        if factor.name not in region:
            factors.append(factor)
            continue
        if factor.kind == "categorical":
            raise StudyError(f"'{factor.name}' is categorical, so its region can't be limited by low and high.")
        low, high = (float(value) for value in region[factor.name])
        if low >= high:
            raise StudyError(f"For '{factor.name}', the low end of the region must be below the high end.")
        tolerance = 1e-9 * max(abs(float(factor.high)), 1.0)
        if low < float(factor.low) - tolerance or high > float(factor.high) + tolerance:
            raise StudyError(
                f"The region for '{factor.name}' goes beyond its limits of {float(factor.low):g} to "
                f"{float(factor.high):g}. Widen the limits in the study's definition first."
            )
        if factor.kind == "discrete":
            levels = [level for level in factor.levels if low <= level <= high]
            if len(levels) < 2:
                raise StudyError(f"The region for '{factor.name}' needs to include at least 2 of its levels.")
            factors.append(replace(factor, levels=tuple(levels)))
        else:
            factors.append(replace(factor, low=low, high=high))
    return replace(study, factors=factors)


def _inside_region(frame: pd.DataFrame, region: dict | None) -> np.ndarray:
    inside = np.ones(len(frame), dtype=bool)
    for name, (low, high) in (region or {}).items():
        values = frame[name].to_numpy(dtype=float)
        inside &= (values >= float(low) - 1e-9) & (values <= float(high) + 1e-9)
    return inside


@dataclass
class _Search:
    """What the search for new runs works on."""

    terms: list[Term]
    counted: pd.DataFrame
    labels: pd.Series | None
    label: int | None
    candidates: pd.DataFrame
    F: np.ndarray  # the runs that count, which can't change
    C: np.ndarray  # the candidates, which all belong to the new runs' block
    repeatable: np.ndarray  # rows of F that can be repeated. Row i is also row i of C
    block_columns: list[int]

    def choose_repeats(self, count: int) -> np.ndarray:
        """Pick rows of C that repeat the earlier runs the model leans on most.

        A run's leverage is how much a repeat of it would add to det(X'X), so this is the
        same aim as D-optimality. Repeats go round the distinct runs before doubling up.
        """
        if not count:
            return np.array([], dtype=int)
        if not len(self.repeatable):
            raise StudyError("None of the earlier runs are inside the allowed region, so none can be repeated. Set the repeats to 0.")
        size = self.F.shape[1]
        inverse = np.linalg.inv(self.F.T @ self.F + 1e-8 * np.eye(size))
        rows = self.F[self.repeatable]
        order = np.argsort(-np.sum((rows @ inverse) * rows, axis=1), kind="stable")
        return order[np.arange(count) % len(order)]

    def still_missing(self, repeats: np.ndarray) -> int:
        """Count how many new runs, besides these repeats, the model still needs."""
        known = np.vstack([self.F, self.C[repeats]])
        return self.F.shape[1] - int(np.linalg.matrix_rank(known))


def _setup(study: Study, sheet: pd.DataFrame, new_block: bool, region: dict | None, rng) -> _Search:
    """Build everything the search needs: the fixed runs, the candidates and their matrices."""
    terms = model_terms(study)
    counted = _runs_to_build_on(study, sheet)
    labels, label = _block_labels(sheet, new_block)
    region_study = _region_study(study, region)

    # Earlier settings come first among the candidates, so the search can choose to repeat one.
    earlier = counted[study.factor_names].reset_index(drop=True)
    usable = study.is_feasible(earlier) & _inside_region(earlier, region) & ~earlier.duplicated().to_numpy()
    repeatable = np.flatnonzero(usable)
    grid = candidate_points(region_study, {}, rng)
    candidates = pd.concat([earlier.iloc[repeatable], grid], ignore_index=True).drop_duplicates(ignore_index=True)

    fixed_frame = counted[study.factor_names].copy()
    candidate_frame = candidates.copy()
    if label is not None:
        fixed_frame["block"] = labels.loc[counted.index]
        candidate_frame["block"] = label
    combined = pd.concat([fixed_frame, candidate_frame], ignore_index=True)
    matrix = model_matrix(study, combined, terms)
    split = len(fixed_frame)
    F, C = matrix.values[:split], matrix.values[split:]

    size = F.shape[1]
    if np.linalg.matrix_rank(np.vstack([F, C])) < size:
        raise StudyError(
            "This model can't be estimated even with more runs at the allowed settings. For example, "
            "curvature needs at least three settings of a factor. Simplify the model or widen the region."
        )
    block_columns = [i for i, group in enumerate(matrix.groups) if group == "block"]
    return _Search(terms, counted, labels, label, candidates, F, C, repeatable, block_columns)


def minimum_extra_runs(
    study: Study,
    sheet: pd.DataFrame,
    new_block: bool = True,
    region: dict | None = None,
    replicate_runs: int = 0,
) -> int:
    """Count the fewest new runs, including any repeats, that make the model estimable."""
    search = _setup(study, clean_runsheet(study, sheet), new_block, region, np.random.default_rng(0))
    repeats = search.choose_repeats(int(replicate_runs))
    return len(repeats) + search.still_missing(repeats)


def _combined_sheet(sheet: pd.DataFrame, new: pd.DataFrame, labels, label, study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the new runs after the existing ones, which stay exactly as they were."""
    out = sheet.copy()
    if "run" not in out:
        out.insert(0, "run", np.arange(1, len(out) + 1))
    if labels is not None:
        if "block" in out:
            out["block"] = labels
        else:
            out.insert(1, "block", labels)
    last_run = pd.to_numeric(out["run"], errors="coerce").max()
    last_run = int(last_run) if pd.notna(last_run) else len(out)

    rows = new[study.factor_names].copy()
    rows.insert(0, "run", np.arange(last_run + 1, last_run + 1 + len(rows)))
    if label is not None:
        rows.insert(1, "block", label)
    if "std_order" in out:
        last_order = pd.to_numeric(out["std_order"], errors="coerce").max()
        last_order = int(last_order) if pd.notna(last_order) else len(out)
        rows["std_order"] = np.arange(last_order + 1, last_order + 1 + len(rows))
    for column in out.columns:
        if column not in rows:
            rows[column] = np.nan
    combined = pd.concat([out, rows[list(out.columns)]], ignore_index=True)
    if "notes" in combined:
        combined["notes"] = notes_as_text(combined["notes"])
    return combined, rows


def augment_design(
    study: Study,
    sheet: pd.DataFrame,
    runs: int,
    criterion: str = "I",
    replicate_runs: int = 0,
    new_block: bool = True,
    region: dict | None = None,
    starts: int | None = None,
) -> Augmentation:
    """Choose the best new runs to add to a run sheet, without changing the existing runs.

    runs            how many new runs to add, including any repeats
    criterion       "I" for the most precise predictions, "D" for the most precise effects
    replicate_runs  how many of the new runs repeat earlier ones, to measure noise and link
                    the new session to the old one
    new_block       whether the new runs will be done in a separate session
    region          optional {factor: (low, high)} limits for the new runs only
    """
    if criterion not in GOALS:
        raise StudyError(f"Unknown goal '{criterion}'. Use 'I' for predictions or 'D' for effects.")
    runs, replicate_runs = int(runs), int(replicate_runs)
    if runs < 1:
        raise StudyError("Add at least 1 run.")
    if replicate_runs < 0 or replicate_runs > runs:
        raise StudyError("The number of repeats must be between 0 and the number of new runs.")
    sheet = clean_runsheet(study, sheet).reset_index(drop=True)

    # The same inputs give the same runs, while a later round of adding runs differs.
    seed = study.seed if study.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng([int(seed), len(sheet)])
    search = _setup(study, sheet, new_block, region, rng)
    counted, labels, label, candidates, C = search.counted, search.labels, search.label, search.candidates, search.C

    # Repeats are chosen first, so the search for the other runs can take them into account.
    repeats = search.choose_repeats(replicate_runs)
    fresh_runs = runs - replicate_runs
    missing = search.still_missing(repeats)
    if fresh_runs < missing:
        raise StudyError(
            f"This model needs at least {missing} new runs that aren't repeats, on top of the {len(counted)} "
            f"you can build on, but this would add {fresh_runs}. Add more runs, or simplify the model."
        )

    notes: list[str] = []
    ignored = len(sheet) - len(counted)
    if ignored:
        notes.append(f"{ignored} run(s) with no result weren't counted as information, but stay in the run sheet.")

    if fresh_runs:
        fixed = np.vstack([search.F, C[repeats]])
        region_grid = C.copy()
        region_grid[:, search.block_columns] = 0.0  # predictions average over blocks
        moments = region_grid.T @ region_grid / len(region_grid)
        effort = starts if starts is not None else default_starts(len(C), fresh_runs, C.shape[1])
        chosen = exchange_search(C, fresh_runs, criterion, rng, effort, fixed=fixed, moments=moments)
    else:
        chosen = np.array([], dtype=int)

    picked = candidates.iloc[np.concatenate([chosen, repeats]).astype(int)][study.factor_names]
    new = round_frame(study, picked).reset_index(drop=True)
    new = new.iloc[rng.permutation(len(new))].reset_index(drop=True)  # run them in random order

    combined, rows = _combined_sheet(sheet, new, labels, label, study)

    before = counted[study.factor_names].copy()
    if labels is not None:
        before["block"] = labels.loc[counted.index]
    after_rows = new.copy()
    if label is not None:
        after_rows["block"] = label
    after = pd.concat([before, after_rows], ignore_index=True)

    shown = [column for column in ("run", "block", *study.factor_names) if column in rows]
    return Augmentation(
        new_runs=rows[shown].reset_index(drop=True),
        sheet=combined,
        design_before=before,
        design_after=after,
        terms=search.terms,
        criterion=criterion,
        seed=int(seed),
        block=label,
        counted_runs=len(counted),
        ignored_runs=ignored,
        notes=notes,
    )


def compare_quality(study: Study, augmentation: Augmentation, signal_to_noise: float = 2.0) -> pd.DataFrame:
    """Show what the new runs would gain, as a before-and-after table."""
    before = evaluate_design(study, augmentation.design_before, augmentation.terms, signal_to_noise)
    after = evaluate_design(study, augmentation.design_after, augmentation.terms, signal_to_noise)

    def column(quality) -> list:
        spare = quality.residual_df
        if not quality.estimable:
            return [quality.runs, "can't fit this model", quality.pure_error_df, "", "", ""]
        power = quality.lowest_power
        return [
            quality.runs,
            spare,
            quality.pure_error_df,
            f"{quality.mean_prediction_se:.2f}",
            f"{quality.max_prediction_se:.2f}",
            f"{power:.2f}" if np.isfinite(power) else "",
        ]

    return pd.DataFrame(
        {
            "": ["Runs", "Spare runs to check the model", "Replicates to measure noise", "Typical prediction error (x noise)", "Worst prediction error (x noise)", "Lowest power"],
            "Now": [str(value) for value in column(before)],
            "With the new runs": [str(value) for value in column(after)],
        }
    )
