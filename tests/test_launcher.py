"""Check the icon that starts the app, and the button that stops it."""

import os
import plistlib
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from doekit import launcher
from doekit.shortcut import ShortcutError, desktop_entry, launch_command, mac_script, make_mac_app, stop_command, windows_script

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app/streamlit_app.py")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeProcess:
    def __init__(self, ends=False):
        self.ends = ends

    def poll(self):
        return 1 if self.ends else None


@pytest.fixture
def no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    monkeypatch.setattr(launcher, "notify", lambda message: None)
    monkeypatch.delenv("DOEKIT_NO_BROWSER", raising=False)
    return opened


# ---------------------------------------------------------------------- starting the app
def test_the_app_is_only_reachable_from_this_computer():
    command = launcher.server_command(ROOT, 8510)
    assert command[command.index("--server.address") + 1] == "localhost"
    assert command[command.index("--server.port") + 1] == "8510"
    assert str(ROOT / "app" / "streamlit_app.py") in command
    assert launcher.server_environment()["DOEKIT_LAUNCHER"] == "1"


def test_a_running_app_is_just_opened(monkeypatch, no_browser):
    monkeypatch.setattr(launcher, "is_ready", lambda port: True)
    monkeypatch.setattr(launcher, "start_server", lambda *args: pytest.fail("started a second copy"))
    assert launcher.launch(ROOT, 8510) == "http://localhost:8510"
    assert no_browser == ["http://localhost:8510"]


def test_a_stopped_app_is_started_then_opened_once_ready(monkeypatch, no_browser):
    answers = iter([False, False, False, True])
    started = []
    monkeypatch.setattr(launcher, "is_ready", lambda port: next(answers))
    monkeypatch.setattr(launcher, "is_listening", lambda port: False)
    monkeypatch.setattr(launcher, "start_server", lambda root, port: started.append(port) or FakeProcess())
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    launcher.launch(ROOT, 8510)
    assert started == [8510]
    assert no_browser == ["http://localhost:8510"]


def test_a_second_click_while_starting_does_not_start_another_copy(monkeypatch, no_browser):
    answers = iter([False, False, True])
    monkeypatch.setattr(launcher, "is_ready", lambda port: next(answers))
    monkeypatch.setattr(launcher, "is_listening", lambda port: True)
    monkeypatch.setattr(launcher, "start_server", lambda *args: pytest.fail("started a second copy"))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    launcher.launch(ROOT, 8510)
    assert len(no_browser) == 1


def test_a_failed_start_points_to_the_log(monkeypatch, tmp_path, no_browser):
    log = tmp_path / "app.log"
    log.write_text("ModuleNotFoundError: No module named 'streamlit'\n")
    monkeypatch.setenv("DOEKIT_LOG", str(log))
    monkeypatch.setattr(launcher, "is_ready", lambda port: False)
    monkeypatch.setattr(launcher, "is_listening", lambda port: False)
    monkeypatch.setattr(launcher, "start_server", lambda root, port: FakeProcess(ends=True))
    with pytest.raises(launcher.LaunchError, match="stopped while starting") as error:
        launcher.launch(ROOT, 8510)
    assert str(log) in str(error.value) and "No module named" in str(error.value)
    assert not no_browser


def test_a_slow_start_gives_up_with_a_message(monkeypatch, no_browser):
    monkeypatch.setattr(launcher, "is_ready", lambda port: False)
    monkeypatch.setattr(launcher, "is_listening", lambda port: True)
    with pytest.raises(launcher.LaunchError, match="didn't start within"):
        launcher.launch(ROOT, 8510, timeout=0.1)


def test_a_moved_folder_is_explained(tmp_path, no_browser):
    with pytest.raises(launcher.LaunchError, match="make the icon again"):
        launcher.launch(tmp_path, 8510)


# ---------------------------------------------------------------------- making the icon
def test_mac_icon_script_starts_reopens_watches_and_stops_doekit():
    root = PurePosixPath("/Users/me/My Documents/doe")  # spaces must survive, and so must / on Windows
    script = mac_script(root=root, python="/opt/conda/envs/doe/bin/python", port=8510)
    start = "cd '/Users/me/My Documents/doe' && /opt/conda/envs/doe/bin/python -m doekit.launcher --root '/Users/me/My Documents/doe' --port 8510"
    assert f'do shell script "{start}"' in script
    assert 'property doekitAddress : "http://localhost:8510"' in script
    assert "on reopen\n\topen location doekitAddress" in script  # clicking the Dock icon opens doekit
    assert "/_stcore/health" in script and "quit -- doekit was closed from its sidebar" in script
    assert 'do shell script "/opt/conda/envs/doe/bin/python -m doekit.launcher --stop --port 8510"' in script
    assert "continue quit" in script


@pytest.mark.skipif(sys.platform != "darwin", reason="the Mac icon is built with macOS's own tools")
def test_mac_icon_is_a_signed_app_that_stays_open_with_doekits_icon(tmp_path):
    root = tmp_path / "My Documents" / "doe"
    app = make_mac_app(tmp_path / "Desktop", root=root, python="/opt/conda/envs/doe/bin/python")
    details = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert details["OSAAppletStayOpen"] is True  # keeps its icon in the Dock
    assert details["CFBundleIdentifier"] == "local.doekit.launcher"
    assert details["CFBundleName"] == "doekit"
    assert "CFBundleIconName" not in details and not (app / "Contents/Resources/Assets.car").exists()
    assert not [key for key in details if key.endswith("UsageDescription")]
    with Image.open(app / "Contents" / "Resources" / "applet.icns") as icon:
        assert icon.size[0] >= 512
    assert subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True).returncode == 0
    compiled = subprocess.run(["osadecompile", str(app / "Contents/Resources/Scripts/main.scpt")], capture_output=True, text=True).stdout
    assert f"--root '{root}'" in compiled

    # Making it again replaces it, but an unrelated app with the same name is never touched.
    make_mac_app(tmp_path / "Desktop", root=root)
    other = tmp_path / "Elsewhere" / "doekit.app" / "Contents"
    other.mkdir(parents=True)
    (other / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.someone.else"}))
    with pytest.raises(ShortcutError, match="already a different"):
        make_mac_app(tmp_path / "Elsewhere", root=root)
    assert plistlib.loads((other / "Info.plist").read_bytes())["CFBundleIdentifier"] == "com.someone.else"


def test_windows_shortcut_uses_the_windowless_python_and_quotes_paths(tmp_path):
    python = tmp_path / "envs" / "doe" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (python.parent / "pythonw.exe").touch()
    root = Path("C:/Users/Fraser O'Neil/doe")
    script = windows_script(None, root=root, python=str(python))
    assert f"$link.TargetPath = '{python.parent / 'pythonw.exe'}'" in script
    assert "Fraser O''Neil" in script  # PowerShell doubles a single quote inside quotes
    assert "[Environment]::GetFolderPath('Desktop')" in script  # finds a Desktop moved into OneDrive
    assert "[Environment]::GetFolderPath('Programs')" in script  # and the Start menu
    assert "doekit.ico,0" in script


@pytest.mark.skipif(sys.platform == "win32", reason="pretends to be Linux, which Windows paths can't")
def test_linux_icon_goes_in_a_folder_that_does_not_exist_yet(tmp_path, monkeypatch):
    from doekit import shortcut

    monkeypatch.setattr(shortcut.sys, "platform", "linux")
    monkeypatch.setattr(shortcut.os, "name", "posix")
    made = shortcut.make_shortcut(tmp_path / "new" / "Desktop")
    assert made == [tmp_path / "new" / "Desktop" / "doekit.desktop"] and made[0].exists()


def test_windows_shortcut_script_makes_its_folder_and_stops_on_failure():
    script = windows_script(["C:/Users/me/Desktop"], root=Path("C:/doe"), python="C:/doe/env/python.exe")
    assert script.startswith("$ErrorActionPreference = 'Stop'")
    assert "New-Item -ItemType Directory -Force -Path $folder" in script


def test_linux_entry_runs_the_launcher():
    entry = desktop_entry(root=PurePosixPath("/home/me/doe"), python="/home/me/conda/envs/doe/bin/python")
    assert "Exec=/home/me/conda/envs/doe/bin/python -m doekit.launcher --root /home/me/doe" in entry
    assert "Terminal=false" in entry


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("lsof"), reason="runs the icon's commands on macOS or Linux")
def test_the_icons_commands_really_start_and_stop_doekit(tmp_path):
    """Run the exact commands the Mac icon runs when it's opened and when it's quit."""
    port = free_port()
    settings = {"DOEKIT_NO_BROWSER": "1", "DOEKIT_LOG": str(tmp_path / "app.log"), "DOEKIT_SAVE_DIR": str(tmp_path / "studies")}
    start = launch_command(ROOT, port=port, environment=settings)
    try:
        first = subprocess.run(["/bin/sh", "-c", start], timeout=150, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr + (tmp_path / "app.log").read_text()
        assert launcher.is_ready(port)

        # Opening it again finds it running, rather than starting a second copy.
        assert subprocess.run(["/bin/sh", "-c", start], timeout=20).returncode == 0
        assert len(launcher.server_processes(port)) == 1

        assert subprocess.run(["/bin/sh", "-c", stop_command(port=port)], timeout=30).returncode == 0
        assert not launcher.is_listening(port)
    finally:
        for pid in launcher.server_processes(port):
            os.kill(pid, 15)


@pytest.mark.skipif(sys.platform == "win32", reason="Linux and macOS only")
def test_doekit_is_found_on_linux_without_lsof(monkeypatch):
    """Some Linux systems have no lsof, but ss comes with almost every one."""
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/ss" if name == "ss" else None)
    monkeypatch.setattr(launcher.os.path, "exists", lambda path: False)
    listing = 'LISTEN 0 128 127.0.0.1:8510 0.0.0.0:* users:(("python",pid=4321,fd=7))\n'
    calls = []

    def fake_run(command, **options):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.server_processes(8510) == [4321]
    assert calls[0][0] == "ss" and "sport = :8510" in calls[0]


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("lsof"), reason="uses lsof")
def test_stopping_leaves_other_programs_on_the_port_alone():
    port = free_port()
    other = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not launcher.is_listening(port) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert launcher.stop_server(port) is False
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait()


@pytest.mark.skipif(sys.platform != "win32", reason="makes a real Windows shortcut")
def test_windows_shortcut_is_made_and_points_at_the_launcher(tmp_path):
    from doekit.shortcut import _powershell_text, make_windows_shortcut

    made = make_windows_shortcut(tmp_path)
    assert made == [tmp_path / "doekit.lnk"] and made[0].exists()
    read_back = (
        "$link = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _powershell_text(str(made[0])) + "); "
        "Write-Output $link.TargetPath; Write-Output $link.Arguments; Write-Output $link.IconLocation"
    )
    target, arguments, icon = subprocess.run(["powershell", "-NoProfile", "-Command", read_back], capture_output=True, text=True).stdout.splitlines()[:3]
    assert target.lower().endswith("pythonw.exe")
    assert arguments.startswith("-m doekit.launcher --root")
    assert icon.lower().startswith(str(ROOT / "app" / "assets" / "doekit.ico").lower())


@pytest.mark.skipif(sys.platform != "win32", reason="starts doekit the way the Windows shortcut does")
def test_windows_shortcut_command_really_starts_and_stops_doekit(tmp_path):
    port = free_port()
    windowless = Path(sys.executable).with_name("pythonw.exe")
    python = str(windowless if windowless.exists() else sys.executable)
    settings = {**os.environ, "DOEKIT_NO_BROWSER": "1", "DOEKIT_LOG": str(tmp_path / "app.log"), "DOEKIT_SAVE_DIR": str(tmp_path / "studies")}
    start = [python, "-m", "doekit.launcher", "--root", str(ROOT), "--port", str(port)]
    try:
        assert subprocess.run(start, env=settings, cwd=ROOT, timeout=150).returncode == 0, (tmp_path / "app.log").read_text(errors="replace")
        assert launcher.is_ready(port)
        assert subprocess.run(start, env=settings, cwd=ROOT, timeout=30).returncode == 0
        assert len(launcher.server_processes(port)) == 1

        assert launcher.stop_server(port) is True
        assert not launcher.is_listening(port)
    finally:
        for pid in launcher.server_processes(port):
            os.kill(pid, 15)


# ---------------------------------------------------------------------- stopping the app
def test_close_button_only_appears_when_started_from_the_icon(monkeypatch):
    monkeypatch.delenv("DOEKIT_LAUNCHER", raising=False)
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    assert not [button for button in test.button if button.key == "close_app"]

    monkeypatch.setenv("DOEKIT_LAUNCHER", "1")
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    assert [button for button in test.button if button.key == "close_app"]


def test_close_button_saves_then_stops_the_app(monkeypatch):
    monkeypatch.setenv("DOEKIT_LAUNCHER", "1")
    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, signal: signals.append((pid, signal)))
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    test.button(key="close_app").click()
    test.run()
    assert not test.exception
    assert any("doekit has stopped" in message.value for message in test.success)
    deadline = time.monotonic() + 5
    while not signals and time.monotonic() < deadline:
        time.sleep(0.1)
    assert signals == [(os.getpid(), 15)]
