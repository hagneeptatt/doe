"""Check the one-line installers. GitHub also runs them for real on Windows, macOS and Linux."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/hagneeptatt/doe/main"
MAC_LINE = f"curl -fsSL {RAW}/install.sh | bash"
WINDOWS_LINE = f"irm {RAW}/install.ps1 | iex"


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="needs bash")
def test_mac_and_linux_installer_has_no_syntax_errors():
    assert subprocess.run(["bash", "-n", str(ROOT / "install.sh")]).returncode == 0


@pytest.mark.skipif(not (shutil.which("powershell") or shutil.which("pwsh")), reason="needs PowerShell")
def test_windows_installer_has_no_syntax_errors():
    shell = shutil.which("powershell") or shutil.which("pwsh")
    check = (
        "$errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{ROOT / 'install.ps1'}', [ref]$null, [ref]$errors) | Out-Null; "
        "$errors.Count"
    )
    result = subprocess.run([shell, "-NoProfile", "-Command", check], capture_output=True, text=True)
    assert result.stdout.strip() == "0", result.stdout + result.stderr


def test_the_lines_people_paste_are_the_ones_in_the_instructions():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manual = (ROOT / "docs" / "manual.md").read_text(encoding="utf-8")
    for line, script in [(MAC_LINE, "install.sh"), (WINDOWS_LINE, "install.ps1")]:
        assert line in readme, line
        assert line in manual, line
        assert line in (ROOT / script).read_text(encoding="utf-8"), script


def test_installers_use_files_that_exist():
    for script in ("install.sh", "install.ps1"):
        text = (ROOT / script).read_text(encoding="utf-8")
        for needed in ("environment.yml", "make_shortcut.py", "doekit.launcher", "app", "streamlit_app.py"):
            assert needed in text, (script, needed)
    assert (ROOT / "environment.yml").exists()
    assert (ROOT / "scripts" / "make_shortcut.py").exists()


def test_installers_keep_saved_studies_when_updating():
    """An update replaces the code, so the data and results folders have to come along."""
    for script in ("install.sh", "install.ps1"):
        text = (ROOT / script).read_text(encoding="utf-8")
        assert re.search(r"data\W{0,5}results", text), script
        assert "doe-previous" in text, script
