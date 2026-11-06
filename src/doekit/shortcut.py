"""Make an icon that starts the doekit app with a double-click.

The icon runs doekit.launcher with this computer's own Python and doe folder, so it's
made on each computer rather than downloaded. That also means macOS doesn't treat it as
an app from an unknown developer. On a Mac, the icon stays in the Dock while doekit runs.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__
from .launcher import DEFAULT_PORT, _applescript_text, app_url

NAME = "doekit"
DESCRIPTION = "Design, analyse and optimise experiments"
ROOT = Path(__file__).resolve().parents[2]
ICON_PNG = ROOT / "app" / "assets" / "doekit_icon.png"
ICON_ICO = ROOT / "app" / "assets" / "doekit.ico"


class ShortcutError(RuntimeError):
    """A problem making the icon, explained in plain language."""


def check_environment() -> None:
    """Make sure this Python can run the app, so the icon doesn't fail later."""
    import importlib.util

    if importlib.util.find_spec("streamlit") is None:
        raise ShortcutError("This Python can't run the app. Run 'conda activate doe' first, then try again.")


def desktop_folder() -> Path:
    if os.name == "nt":
        script = "[Environment]::GetFolderPath('Desktop')"
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())  # may be inside OneDrive
    return Path.home() / "Desktop"


# ---------------------------------------------------------------------- macOS
APP_ID = "local.doekit.launcher"


def launch_command(root: Path = ROOT, python: str | None = None, port: int = DEFAULT_PORT, environment: dict | None = None) -> str:
    """The shell command the Mac icon runs to start doekit and open it."""
    settings = "".join(f"{name}={shlex.quote(str(value))} " for name, value in (environment or {}).items())
    return (
        f"cd {shlex.quote(str(root))} && {settings}"
        f"{shlex.quote(python or sys.executable)} -m doekit.launcher --root {shlex.quote(str(root))} --port {port}"
    )


def stop_command(python: str | None = None, port: int = DEFAULT_PORT) -> str:
    return f"{shlex.quote(python or sys.executable)} -m doekit.launcher --stop --port {port}"


def mac_script(root: Path = ROOT, python: str | None = None, port: int = DEFAULT_PORT, environment: dict | None = None) -> str:
    """Write the AppleScript behind the Mac icon.

    It's saved as an app that stays open, so its icon stays in the Dock while doekit runs.
    Clicking the icon opens doekit again, and quitting it stops doekit. If doekit is closed
    from its sidebar instead, the icon notices within a few seconds and quits too.
    """
    address = _applescript_text(app_url(port))
    return f"""-- doekit: starts the app, stays in the Dock while it runs, and stops it on Quit.
-- Made by scripts/make_shortcut.py.
property doekitAddress : {address}

on run
\ttry
\t\tdo shell script {_applescript_text(launch_command(root, python, port, environment))}
\ton error
\t\tquit -- the launcher has already said what went wrong
\tend try
end run

on reopen
\topen location doekitAddress
end reopen

on idle
\ttry
\t\tdo shell script "/usr/bin/curl --silent --fail --max-time 3 " & quoted form of (doekitAddress & "/_stcore/health")
\ton error
\t\tquit -- doekit was closed from its sidebar
\tend try
\treturn 5
end idle

on quit
\ttry
\t\tdo shell script {_applescript_text(stop_command(python, port))}
\tend try
\tcontinue quit
end quit
"""


def _is_ours(app: Path) -> bool:
    try:
        with open(app / "Contents" / "Info.plist", "rb") as handle:
            return plistlib.load(handle).get("CFBundleIdentifier") == APP_ID
    except (OSError, plistlib.InvalidFileException):
        return False


def make_mac_app(
    folder: Path, root: Path = ROOT, python: str | None = None, port: int = DEFAULT_PORT, environment: dict | None = None
) -> Path:
    """Build the Mac icon with the AppleScript tools that come with macOS."""
    from PIL import Image

    if sys.platform != "darwin" or not shutil.which("osacompile"):
        raise ShortcutError("The Mac icon can only be made on a Mac.")
    folder = Path(folder)
    app = folder / f"{NAME}.app"
    if app.exists():
        if not _is_ours(app):
            raise ShortcutError(f"There's already a different {app.name} in {folder}. Move it, or choose another folder with --folder.")
        shutil.rmtree(app)  # an older doekit icon, made by this script
    folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / f"{NAME}.applescript"
        source.write_text(mac_script(root, python, port, environment), encoding="utf-8")
        result = subprocess.run(["osacompile", "-s", "-o", str(app), str(source)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ShortcutError(f"macOS couldn't build the icon: {result.stderr.strip()}")

    # Swap in doekit's icon, and describe the app as doekit rather than a generic script.
    resources = app / "Contents" / "Resources"
    (resources / "Assets.car").unlink(missing_ok=True)  # holds the standard script icon
    Image.open(ICON_PNG).save(resources / "applet.icns")
    with open(app / "Contents" / "Info.plist", "rb") as handle:
        details = plistlib.load(handle)
    details.pop("CFBundleIconName", None)
    for key in [key for key in details if key.startswith("NS") and key.endswith("UsageDescription")]:
        del details[key]  # requests for the camera, contacts and so on, which doekit never makes
    details.update(
        {
            "CFBundleName": NAME,
            "CFBundleDisplayName": NAME,
            "CFBundleIdentifier": APP_ID,
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
        }
    )
    with open(app / "Contents" / "Info.plist", "wb") as handle:
        plistlib.dump(details, handle)

    # Changing the app means signing it again, as macOS expects, for this computer only.
    result = subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ShortcutError(f"macOS couldn't sign the icon: {result.stderr.strip()}")
    os.utime(app)  # so Finder notices the new icon
    return app


# ---------------------------------------------------------------------- Windows
def _powershell_text(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def windows_script(folders: list[str] | None, root: Path = ROOT, python: str | None = None) -> str:
    """Write the PowerShell that makes a shortcut on the Desktop and in the Start menu."""
    python = Path(python or sys.executable)
    windowless = python.with_name("pythonw.exe")
    target = windowless if windowless.exists() else python  # pythonw.exe opens no console window
    arguments = f'-m doekit.launcher --root "{root}"'
    icon = f"{ICON_ICO},0"
    places = (
        ", ".join(_powershell_text(folder) for folder in folders)
        if folders
        else "[Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs')"
    )
    return "\n".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f"foreach ($folder in @({places})) {{",
            f"  $link = $shell.CreateShortcut((Join-Path $folder {_powershell_text(NAME + '.lnk')}))",
            f"  $link.TargetPath = {_powershell_text(str(target))}",
            f"  $link.Arguments = {_powershell_text(arguments)}",
            f"  $link.WorkingDirectory = {_powershell_text(str(root))}",
            f"  $link.IconLocation = {_powershell_text(icon)}",
            f"  $link.Description = {_powershell_text(DESCRIPTION)}",
            "  $link.Save()",
            "  Write-Output (Join-Path $folder " + _powershell_text(NAME + ".lnk") + ")",
            "}",
        ]
    )


def make_windows_shortcut(folder: Path | None = None) -> list[Path]:
    script = windows_script([str(folder)] if folder else None)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ShortcutError(f"Windows couldn't make the shortcut: {result.stderr.strip()}")
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------- Linux
def desktop_entry(root: Path = ROOT, python: str | None = None) -> str:
    python = python or sys.executable
    command = f"{shlex.quote(python)} -m doekit.launcher --root {shlex.quote(str(root))}"
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={NAME}\n"
        f"Comment={DESCRIPTION}\n"
        f"Exec={command}\n"
        f"Path={root}\n"
        f"Icon={ICON_PNG}\n"
        "Terminal=false\n"
        "Categories=Science;Education;\n"
    )


def make_linux_entry(folder: Path | None = None) -> list[Path]:
    folders = [folder] if folder else [Path.home() / ".local" / "share" / "applications", desktop_folder()]
    made = []
    for place in folders:
        if not place.exists() and place.name == "Desktop":
            continue
        place.mkdir(parents=True, exist_ok=True)
        path = place / f"{NAME}.desktop"
        path.write_text(desktop_entry(), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        made.append(path)
    return made


def make_shortcut(folder: Path | None = None) -> list[Path]:
    """Make the icon in the usual place for this computer, or in the folder given."""
    check_environment()
    if sys.platform == "darwin":
        return [make_mac_app(folder or desktop_folder())]
    if os.name == "nt":
        return make_windows_shortcut(folder)
    return make_linux_entry(folder)
