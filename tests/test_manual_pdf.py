"""Check the page that the PDF manual is printed from."""

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("markdown_it")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_manual_pdf", ROOT / "scripts/build_manual_pdf.py")
build_manual_pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_manual_pdf)


@pytest.fixture(scope="module")
def page():
    return build_manual_pdf.render_html((ROOT / "docs/manual.md").read_text(encoding="utf-8"))


def test_every_link_in_the_pdf_leads_somewhere(page):
    targets = set(re.findall(r'id="([^"]+)"', page))
    links = re.findall(r'href="#([^"]+)"', page)
    assert len(links) > 15
    assert not [link for link in links if link not in targets]


def test_tables_and_lists_are_formatted_rather_than_left_as_text(page):
    assert page.count("<table>") > 15
    assert "|---" not in page
    assert page.count('class="task"') == 7
    assert "[ ]" not in page
    assert "**" not in page


def test_indented_lists_stay_nested(page):
    """The install steps have sub-bullets that some converters flatten."""
    start = page.index("Install Git, if you")
    assert "<ul>" in page[start : page.index("In the terminal, go to the folder")]


def test_the_pdf_exists_and_is_a_pdf():
    assert (ROOT / "docs/doekit_manual.pdf").read_bytes().startswith(b"%PDF")
