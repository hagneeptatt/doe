"""Check that the run counts printed in the manual match what doekit really generates."""

from pathlib import Path

import pytest

from doekit.designs import create_design
from doekit.study import StudyError, study_from_config

ROOT = Path(__file__).resolve().parents[1]
MANUAL = (ROOT / "docs/manual.md").read_text(encoding="utf-8")


def table_after(heading: str) -> list[list[str]]:
    """Read the rows of the first table after a heading, leaving out the header."""
    lines = MANUAL[MANUAL.index(heading) :].splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[first + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def expected_runs(count: int, **design) -> str:
    """Give the design's run count as the manual would show it."""
    factors = [{"name": f"x{i + 1}", "low": 0, "high": 1} for i in range(count)]
    try:
        study = study_from_config({"name": "t", "seed": 1, "factors": factors, "design": design})
        return str(len(create_design(study).frame))
    except StudyError:
        return ""


def test_the_screening_table_matches_the_code():
    rows = table_after("**To find out which factors matter (screening).**")
    assert len(rows) == 7
    for row in rows:
        k = int(row[0])
        assert row[1] == expected_runs(k, type="full_factorial", center_points=3), row
        assert row[2] == expected_runs(k, type="fractional_factorial", resolution=4, center_points=3), row
        assert row[3] == expected_runs(k, type="fractional_factorial", resolution=5, center_points=3), row
        assert row[4] == expected_runs(k, type="plackett_burman", center_points=3), row


def test_the_optimising_table_matches_the_code():
    rows = table_after("**To find the best settings (optimising).**")
    assert len(rows) == 7
    for row in rows:
        k = int(row[0])
        assert row[1] == str((k + 1) * (k + 2) // 2), row
        three_levels = expected_runs(k, type="full_factorial", levels=3)
        assert row[2] == (three_levels or "Too many"), row
        assert row[3] == expected_runs(k, type="box_behnken", center_points=3), row
        assert row[4] == expected_runs(k, type="central_composite", center_points=4), row
        assert row[5] == expected_runs(k, type="i_optimal", starts=2), row


@pytest.mark.parametrize("heading", ["**To find out which factors matter (screening).**", "**To find the best settings (optimising).**"])
def test_table_headings_do_not_use_unexplained_shorthand(heading):
    start = MANUAL.index(heading)
    header = next(line for line in MANUAL[start:].splitlines() if line.startswith("|"))
    assert "+" not in header
