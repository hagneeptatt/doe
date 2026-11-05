"""Read and write run sheets: the CSV files that carry an experiment through the lab."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .designs import Design
from .study import Study, StudyError


def build_runsheet(study: Study, design: Design) -> pd.DataFrame:
    """Add empty columns for each response, plus one for notes."""
    sheet = design.frame.copy()
    for name in study.response_names:
        sheet[name] = np.nan
    sheet["notes"] = ""
    return sheet


def write_runsheet(study: Study, design: Design, path: Path | None = None, force: bool = False) -> Path:
    """Save a run sheet. An existing one is never replaced unless ``force`` is set,
    because it may already hold results that took weeks to measure."""
    path = Path(path) if path else study.runsheet_path
    if path.exists() and not force:
        raise StudyError(
            f"A run sheet already exists at {path} and it may contain your results, so it was not "
            "replaced. Move or rename it first, or use --force if you're sure it can be overwritten."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    build_runsheet(study, design).to_csv(path, index=False)
    return path


def _decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _parse_table(text: str) -> pd.DataFrame:
    """Parse CSV text, including the variants Excel saves on European computers.

    Where the decimal mark is a comma, Excel separates columns with semicolons and
    writes 1,5 for 1.5. Both forms are handled, so nobody needs to change their settings.
    """
    header = text.splitlines()[0] if text.strip() else ""
    separators = {";": header.count(";"), "\t": header.count("\t"), ",": header.count(",")}
    separator = max(separators, key=separators.get)
    decimal = "," if separator in (";", "\t") else "."
    return pd.read_csv(io.StringIO(text), sep=separator, decimal=decimal, skipinitialspace=True)


def _clean_numbers(series: pd.Series) -> pd.Series:
    """Accept decimal commas left in a column that pandas read as text."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("object").map(lambda value: value.strip().replace(",", ".") if isinstance(value, str) else value)
    text = text.map(lambda value: np.nan if isinstance(value, str) and value == "" else value)
    return pd.to_numeric(text, errors="coerce")


def clean_runsheet(study: Study, sheet: pd.DataFrame, source: str = "The run sheet") -> pd.DataFrame:
    """Check a run sheet against its study, and explain clearly anything that's wrong."""
    sheet = sheet.copy()
    sheet.columns = [str(column).strip() for column in sheet.columns]
    sheet = sheet.dropna(how="all").reset_index(drop=True)  # Excel often leaves blank rows

    missing = [name for name in study.factor_names if name not in sheet.columns]
    if missing:
        raise StudyError(
            f"{source} has no column for: {', '.join(missing)}. "
            f"Its columns are: {', '.join(sheet.columns)}."
        )
    missing = [name for name in study.response_names if name not in sheet.columns]
    if missing:
        raise StudyError(
            f"{source} has no column for the response(s): {', '.join(missing)}. "
            f"Its columns are: {', '.join(sheet.columns)}."
        )

    for factor in study.factors:
        column = sheet[factor.name]
        if factor.kind == "categorical":
            values = column.astype("object").map(lambda v: str(v).strip() if pd.notna(v) else "")
            unknown = sorted(set(values) - set(factor.levels))
            if unknown:
                raise StudyError(
                    f"{source}: factor '{factor.name}' has unknown value(s) {unknown}. "
                    f"Allowed levels are {list(factor.levels)}."
                )
            sheet[factor.name] = values
        else:
            numbers = _clean_numbers(column)
            if numbers.isna().any():
                rows = (numbers[numbers.isna()].index + 1).tolist()
                raise StudyError(
                    f"{source}: factor '{factor.name}' has a blank or non-numeric value in row(s) {rows}."
                )
            sheet[factor.name] = numbers

    for name in study.response_names:
        numbers = _clean_numbers(sheet[name])
        filled = sheet[name].astype("object").map(lambda v: pd.notna(v) and str(v).strip() != "")
        bad = filled & numbers.isna()
        if bad.any():
            rows = (bad[bad].index + 1).tolist()
            raise StudyError(
                f"{source}: response '{name}' has text instead of a number in row(s) {rows}. "
                "If a run failed, leave its result blank and explain why in the notes column."
            )
        sheet[name] = numbers

    if study.mixture_factors:
        totals = sheet[study.mixture_names].sum(axis=1)
        allowed = 0.01 * float(study.mixture_total)
        off = (totals - float(study.mixture_total)).abs() > allowed
        if off.any():
            rows = (off[off].index + 1).tolist()
            raise StudyError(
                f"{source}: the mixture components should add up to {study.mixture_total:g}, "
                f"but don't in row(s) {rows}."
            )
    return sheet


def read_runsheet(study: Study, path: Path | None = None) -> pd.DataFrame:
    """Read and check a study's run sheet from a file."""
    path = Path(path) if path else study.runsheet_path
    if not path.exists():
        raise StudyError(f"Couldn't find a run sheet at {path}. Create the design first.")
    return clean_runsheet(study, _parse_table(_decode_bytes(path.read_bytes())), f"'{path.name}'")


def read_runsheet_bytes(study: Study, raw: bytes, name: str = "The uploaded file") -> pd.DataFrame:
    """Read and check a run sheet from uploaded bytes (used by the web app)."""
    return clean_runsheet(study, _parse_table(_decode_bytes(raw)), name)
