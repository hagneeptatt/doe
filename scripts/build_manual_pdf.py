"""Build docs/doekit_manual.pdf from docs/manual.md, so the manual can be read without running anything.

Run this again whenever the manual changes. It needs Google Chrome (or Chromium), plus one
library that the app itself doesn't use:

    pip install markdown-it-py
    python scripts/build_manual_pdf.py
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "manual.md"
OUTPUT = ROOT / "docs" / "doekit_manual.pdf"
TITLE = "doekit manual and standard operating procedure"

CHROME_LOCATIONS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page {
  size: A4;
  margin: 19mm 17mm 20mm;
  @bottom-left { content: "doekit manual"; font: 8.5pt Arial, Helvetica, sans-serif; color: #52514e; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8.5pt Arial, Helvetica, sans-serif; color: #52514e; }
}
html { font: 10.5pt/1.45 Arial, Helvetica, sans-serif; color: #0b0b0b; }
body { margin: 0; }
a { color: #1c5cab; text-decoration: none; }
h1 { font-size: 23pt; line-height: 1.2; margin: 0 0 14pt; }
h2 { font-size: 16pt; line-height: 1.25; margin: 0 0 9pt; break-before: page; break-after: avoid; }
h3 { font-size: 12pt; margin: 15pt 0 5pt; break-after: avoid; }
p { margin: 0 0 7pt; }
p:has(+ table) { break-after: avoid; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
li > ul, li > ol { margin: 3pt 0 0; }
li.task { list-style: none; margin-left: -14pt; }
hr { display: none; }
code { font: 9pt Menlo, Consolas, "Courier New", monospace; background: #f5f6f8; padding: 0 2pt; overflow-wrap: anywhere; }
pre { background: #f5f6f8; padding: 7pt 9pt; margin: 0 0 8pt; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { padding: 0; background: none; }
blockquote { margin: 0 0 8pt; padding: 3pt 0 3pt 10pt; border-left: 2pt solid #c3c2b7; color: #52514e; }
blockquote p { margin: 0; }
table { width: 100%; border-collapse: collapse; margin: 0 0 10pt; font-size: 8.8pt; line-height: 1.35; break-inside: avoid; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { text-align: left; background: #f5f6f8; font-weight: bold; }
th, td { border: 0.5pt solid #c3c2b7; padding: 3pt 4.5pt; vertical-align: top; overflow-wrap: break-word; }
"""


def slug(heading: str) -> str:
    """Make a heading's link target, the same way GitHub does, so the manual's links keep working."""
    return re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")


def render_html(markdown_text: str) -> str:
    """Turn the manual into a complete web page, ready to print."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        sys.exit("This script needs one extra library. Install it with: pip install markdown-it-py")

    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(markdown_text)
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            token.attrSet("id", slug(tokens[index + 1].content))
    body = parser.renderer.render(tokens, parser.options, {})
    body = re.sub(r"<li>\s*(<p>)?\[ \] ", lambda m: f'<li class="task">{m.group(1) or ""}☐ ', body)
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title>'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def find_chrome(given: str | None) -> str:
    for candidate in [given, *CHROME_LOCATIONS, shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome")]:
        if candidate and (os.path.exists(candidate) or shutil.which(candidate)):
            return candidate
    sys.exit("Couldn't find Google Chrome. Install it, or give its location with --chrome.")


def build_pdf(chrome: str, page_html: str, output: Path) -> None:
    """Print the page to a PDF with a separate Chrome profile, so an open Chrome isn't disturbed."""
    with tempfile.TemporaryDirectory() as folder:
        page = Path(folder) / "manual.html"
        page.write_text(page_html, encoding="utf-8")
        temporary = Path(folder) / "manual.pdf"
        result = subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                f"--user-data-dir={Path(folder) / 'profile'}",
                "--no-pdf-header-footer",
                "--generate-pdf-document-outline",
                f"--print-to-pdf={temporary}",
                page.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if not temporary.exists():
            sys.exit(f"Chrome couldn't make the PDF.\n{result.stderr[-1500:]}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrome", help="where Chrome is, if it isn't found automatically")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="where to save the PDF")
    args = parser.parse_args()
    build_pdf(find_chrome(args.chrome), render_html(SOURCE.read_text(encoding="utf-8")), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
