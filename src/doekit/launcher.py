"""Start the doekit app from an icon, without a terminal.

Clicking the icon runs this. If the app is already running, it just opens it in the
browser. Otherwise it starts the app in the background, waits until it's ready, and then
opens it. The app keeps running until someone chooses 'Close doekit' in its sidebar.

This module only uses the standard library, so it starts quickly, and it works even on
Windows when the conda environment hasn't been activated.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8510  # not Streamlit's usual 8501, so it won't open some other app by mistake
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
STARTUP_SECONDS = 120


class LaunchError(RuntimeError):
    """Something stopped the app from starting, explained in plain language."""


def app_url(port: int) -> str:
    return f"http://localhost:{port}"


def is_listening(port: int) -> bool:
    """Check whether anything is using the port, such as an app that's still starting."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def is_ready(port: int) -> bool:
    """Ask the app whether it's ready to use."""
    try:
        with urllib.request.urlopen(f"{app_url(port)}/_stcore/health", timeout=2) as reply:
            return reply.read().strip() == b"ok"
    except OSError:
        return False


def server_python() -> str:
    """Use python.exe for the app on Windows, even if this was started with pythonw.exe."""
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        console = executable.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(executable)


def server_command(root: Path, port: int) -> list[str]:
    return [
        server_python(), "-m", "streamlit", "run", str(root / "app" / "streamlit_app.py"),
        "--server.port", str(port),
        "--server.address", "localhost",  # only this computer can open it
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]  # fmt: skip


def server_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["DOEKIT_LAUNCHER"] = "1"  # shows the 'Close doekit' button
    if os.name == "nt":
        # An environment that hasn't been activated may not find its own libraries.
        prefix = Path(sys.executable).parent
        folders = [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin", prefix / "Library" / "bin", prefix / "Scripts"]
        environment["PATH"] = os.pathsep.join([*map(str, folders), environment.get("PATH", "")])
    return environment


def log_path(root: Path) -> Path:
    return Path(os.environ.get("DOEKIT_LOG") or root / "data" / "doekit_app.log")


def start_server(root: Path, port: int) -> subprocess.Popen:
    """Start the app in the background, writing its messages to a log file."""
    log = log_path(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    options: dict = {"cwd": root, "env": server_environment(), "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True  # keeps running after the launcher finishes
    with open(log, "ab") as output:
        return subprocess.Popen(server_command(root, port), stdout=output, **options)


def notify(message: str) -> None:
    """Show a brief notification, where that's easy to do."""
    if sys.platform == "darwin":
        script = f"display notification {_applescript_text(message)} with title \"doekit\""
        subprocess.run(["osascript", "-e", script], capture_output=True, check=False)


def show_error(message: str) -> None:
    """Show a problem in a dialog, since there's no terminal to print it in."""
    if sys.platform == "darwin":
        script = f"display alert \"doekit couldn't start\" message {_applescript_text(message)} as critical"
        subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    elif os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "doekit couldn't start", 0x10)
    print(f"doekit couldn't start: {message}", file=sys.stderr)


def _applescript_text(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _last_lines(path: Path, count: int = 6) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-count:])


def launch(root: Path = DEFAULT_ROOT, port: int = DEFAULT_PORT, open_browser: bool = True, timeout: float = STARTUP_SECONDS) -> str:
    """Make sure the app is running, then open it. Returns its address."""
    root = Path(root)
    if not (root / "app" / "streamlit_app.py").exists():
        raise LaunchError(f"The app isn't in {root}. If you moved the doe folder, make the icon again from its new place.")

    if not is_ready(port):
        process = None
        if not is_listening(port):  # if it's listening, a click a moment ago is already starting it
            notify("Starting doekit. Your browser will open in a few seconds.")
            process = start_server(root, port)
        deadline = time.monotonic() + timeout
        while not is_ready(port):
            if process is not None and process.poll() is not None:
                details = _last_lines(log_path(root))
                raise LaunchError(f"The app stopped while starting. Details are in {log_path(root)}.\n\n{details}".strip())
            if time.monotonic() > deadline:
                raise LaunchError(f"The app didn't start within {timeout:.0f} seconds. Details are in {log_path(root)}.")
            time.sleep(0.5)

    url = app_url(port)
    if open_browser and not os.environ.get("DOEKIT_NO_BROWSER"):
        webbrowser.open(url)
    return url


def server_processes(port: int) -> list[int]:
    """Find the processes listening on the port."""
    if os.name == "nt":
        script = f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue).OwningProcess"
        command = ["powershell", "-NoProfile", "-Command", script]
        result = subprocess.run(command, capture_output=True, text=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    elif shutil.which("lsof") or os.path.exists("/usr/sbin/lsof"):
        lsof = shutil.which("lsof") or "/usr/sbin/lsof"
        result = subprocess.run([lsof, "-t", "-i", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, check=False)
    elif shutil.which("ss"):
        # Linux without lsof. ss lists listening sockets with their processes, as "pid=1234".
        result = subprocess.run(["ss", "-H", "-l", "-t", "-n", "-p", f"sport = :{port}"], capture_output=True, text=True, check=False)
        return sorted({int(pid) for pid in re.findall(r"pid=(\d+)", result.stdout)})
    else:
        return []
    return sorted({int(pid) for pid in result.stdout.split() if pid.isdigit() and int(pid) > 0})


def stop_server(port: int, timeout: float = 10) -> bool:
    """Stop the app on this port, as quitting the Mac icon does. Returns whether it was running.

    Only something that answers like doekit does is stopped, so another program that
    happens to use the port is left alone.
    """
    if not is_ready(port):
        return False
    for pid in server_processes(port):
        os.kill(pid, signal.SIGTERM)  # the app saves as it goes, so this loses nothing
    deadline = time.monotonic() + timeout
    while is_listening(port) and time.monotonic() < deadline:
        time.sleep(0.2)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the doekit app and open it in the browser.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the doe folder")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DOEKIT_PORT") or DEFAULT_PORT))
    parser.add_argument("--stop", action="store_true", help="stop the app instead of starting it")
    args = parser.parse_args(argv)
    if args.stop:
        stop_server(args.port)
        return
    try:
        launch(args.root, args.port)
    except LaunchError as error:
        show_error(str(error))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
