"""Save a whole study in one file, so work survives a refresh, a closed page or a crash.

The file holds the study's settings, its run sheet with any results, the chosen design
and the best settings found. It's plain text, written so that a half-finished write can
never damage the previous save. It may describe a study that's only partly filled in, so
nothing here checks that the study is valid: that happens when it's used.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .study import StudyError

FORMAT_KEY = "doekit_saved_study"
FORMAT_VERSION = 1
FILE_NAME = "saved_study.yaml"
HISTORY_FOLDER = "history"
MAX_HISTORY = 20


def build_document(config: dict, sheet: pd.DataFrame | None, solutions: pd.DataFrame | None, state: dict) -> dict:
    """Gather everything about a study into one document."""
    return {
        FORMAT_KEY: FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "state": state,
        "runsheet_csv": None if sheet is None else sheet.to_csv(index=False),
        "solutions_csv": None if solutions is None else solutions.to_csv(index=False),
    }


def dump_document(document: dict) -> bytes:
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True).encode("utf-8")


def is_saved_study(data) -> bool:
    return isinstance(data, dict) and FORMAT_KEY in data


def read_table(text: str) -> pd.DataFrame:
    """Read a table from a document. tests/test_saved_study.py checks that numbers come back exactly."""
    return pd.read_csv(io.StringIO(text))


def read_document(path) -> dict:
    """Read a saved study from a file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as error:
        raise StudyError(f"Couldn't read the saved study at {path}. {error}") from error
    if not is_saved_study(data):
        raise StudyError(f"{path} isn't a saved study.")
    return data


def write_atomically(path: Path, raw: bytes) -> None:
    """Replace a file in one step, so a crash mid-write can't leave a damaged save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".saving_", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def new_save_path(folder, name: str) -> Path:
    """Choose a new location for a study. A study never takes over another study's location,
    even when they share a name, so one can't overwrite the other."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")[:40] or "study"
    return Path(folder) / f"{safe_name}-{uuid.uuid4().hex[:8]}" / FILE_NAME


def save_document(path, document: dict) -> None:
    write_atomically(Path(path), dump_document(document))


def keep_earlier_version(path) -> Path | None:
    """Copy a save into its history folder, before something replaces its results."""
    path = Path(path)
    if not path.exists():
        return None
    history = path.parent / HISTORY_FOLDER
    history.mkdir(parents=True, exist_ok=True)
    copy = history / f"saved_study_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.yaml"
    shutil.copy2(path, copy)
    for old in sorted(history.glob("saved_study_*.yaml"))[:-MAX_HISTORY]:
        old.unlink()
    return copy


@dataclass(frozen=True)
class SavedStudy:
    """A summary of a saved study, for choosing which one to carry on with."""

    path: Path
    name: str
    saved_at: datetime
    runs: int
    complete_runs: int

    @property
    def label(self) -> str:
        when = self.saved_at.strftime("%d %b %Y, %H:%M")
        progress = f"{self.complete_runs} of {self.runs} runs complete" if self.runs else "no run sheet yet"
        return f"{self.name} (saved {when}, {progress})"


def _summarise(path: Path) -> SavedStudy:
    document = read_document(path)
    config = document.get("config") or {}
    runs = complete = 0
    if document.get("runsheet_csv"):
        sheet = read_table(document["runsheet_csv"])
        runs = len(sheet)
        responses = [entry["name"] for entry in config.get("responses") or [] if entry.get("name") in sheet]
        complete = int(sheet[responses].notna().all(axis=1).sum()) if responses else 0
    return SavedStudy(path, str(config.get("name", path.parent.name)), datetime.fromtimestamp(path.stat().st_mtime), runs, complete)


def list_saved_studies(folder, limit: int = 30) -> list[SavedStudy]:
    """List the studies saved in a folder, most recent first. Unreadable files are skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    studies = []
    for path in folder.glob(f"*/{FILE_NAME}"):
        try:
            studies.append(_summarise(path))
        except (StudyError, ValueError, KeyError, AttributeError, TypeError, OSError):
            continue
    studies.sort(key=lambda study: study.saved_at, reverse=True)
    return studies[:limit]
