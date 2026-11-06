"""Fill a run sheet with made-up results, to practise the workflow before using real materials.

A config's ``simulation`` section gives each response a true formula and an amount of
noise. Because the truth is known, this is a good way to learn: you can check whether
the analysis recovers it, and see what happens with a smaller design or noisier data.

Simulated results are never real data. Every simulated row says so in its notes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .expressions import evaluate
from .study import Study, StudyError

SIMULATED_NOTE = "SIMULATED - not a real measurement"


def fill_blank_results(study: Study, sheet: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Simulate only the results that are still blank, leaving existing results alone."""
    simulated = simulate_responses(study, sheet, seed)
    out = sheet.copy()
    filled = pd.Series(False, index=out.index)
    for name in study.response_names:
        blank = out[name].isna()
        out[name] = out[name].where(~blank, simulated[name])
        filled |= blank
    if "notes" not in out:
        out["notes"] = ""
    out["notes"] = out["notes"].astype(object).where(~filled, SIMULATED_NOTE)
    return out


def simulate_responses(study: Study, sheet: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Calculate each response from its formula, then add noise.

    noise          standard deviation added to every run, in the response's units
    noise_percent  random scatter proportional to the value, for responses that span a
                   wide range, such as viscosity or modulus
    block_sd       a random shift shared by all runs in a block, like a batch difference
    """
    if not study.simulation:
        raise StudyError(
            f"Study '{study.name}' has no 'simulation' section, so there's nothing to simulate. "
            "Results for this study have to be measured."
        )
    generator = np.random.default_rng(seed if seed is not None else study.seed)
    out = sheet.copy()
    variables = {name: out[name].to_numpy() for name in study.factor_names}
    for response in study.responses:
        settings = study.simulation.get(response.name)
        if settings is None:
            raise StudyError(f"The 'simulation' section has no entry for the response '{response.name}'.")
        unknown = set(settings) - {"formula", "noise", "noise_percent", "block_sd"}
        if unknown:
            raise StudyError(f"Unknown simulation setting(s) for '{response.name}': {', '.join(sorted(unknown))}.")
        try:
            values = np.asarray(evaluate(settings["formula"], variables), dtype=float)
        except (KeyError, ValueError) as error:
            raise StudyError(f"Couldn't simulate '{response.name}': {error}") from error
        values = values * np.ones(len(out))
        if "block" in out and settings.get("block_sd"):
            shifts = {block: generator.normal(0, float(settings["block_sd"])) for block in out["block"].unique()}
            values = values + out["block"].map(shifts).to_numpy(dtype=float)
        if settings.get("noise_percent"):
            values = values * np.exp(generator.normal(0, float(settings["noise_percent"]) / 100, len(out)))
        if settings.get("noise"):
            values = values + generator.normal(0, float(settings["noise"]), len(out))
        out[response.name] = np.round(values, 4)
    out["notes"] = SIMULATED_NOTE
    return out
