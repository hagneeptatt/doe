"""Check an installation made by install.sh or install.ps1, for the tests on GitHub.

Run it with the installed environment's Python, after the installer has run with
DOEKIT_HOME and DOEKIT_ICON_FOLDER set.
"""

import os
import sys
from pathlib import Path

home = Path(os.environ["DOEKIT_HOME"]).resolve()
code = home / "doe"

import doekit  # noqa: E402
import streamlit  # noqa: E402,F401
from doekit import launcher  # noqa: E402

# doekit runs from the installed copy, in the installed environment.
assert Path(sys.executable).resolve().is_relative_to(home / "env"), sys.executable
assert Path(doekit.__file__).resolve().is_relative_to(code), doekit.__file__

icons = sorted(Path(os.environ["DOEKIT_ICON_FOLDER"]).glob("doekit*"))
assert icons, "no icon was made"
print("icon:", ", ".join(icon.name for icon in icons))

# It starts the way the icon starts it, and stops again.
port = int(os.environ.get("DOEKIT_PORT", "8598"))
launcher.launch(code, port, open_browser=False)
assert launcher.is_ready(port), "doekit didn't start"
assert launcher.stop_server(port), "doekit didn't stop"
assert not launcher.is_listening(port)

if len(sys.argv) > 1 and sys.argv[1] == "--after-update":
    assert (code / "data" / "app_studies" / "kept.txt").read_text() == "my study", "a saved study was lost"
    assert (home / "doe-previous" / "app" / "streamlit_app.py").exists(), "the previous version wasn't kept"
    print("the update kept saved studies and the previous version")
print("installed doekit", doekit.__version__, "starts and stops")
