"""Shared test settings."""

import matplotlib
import pytest
from streamlit.runtime.context import ContextProxy

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def app_runs_on_this_computer(tmp_path, monkeypatch):
    """Treat the visitor as local, and keep the app's automatic saves away from real studies.

    Streamlit gives an address of None to someone on the same computer. Its test tool supplies
    something else, which the app would treat as a visitor from elsewhere.
    """
    monkeypatch.setattr(ContextProxy, "ip_address", property(lambda self: None))
    monkeypatch.setenv("DOEKIT_SAVE_DIR", str(tmp_path / "app_studies"))
    monkeypatch.delenv("DOEKIT_AUTOSAVE", raising=False)
