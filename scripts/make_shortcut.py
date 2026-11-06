"""Make a doekit icon, so the app starts with a double-click instead of from a terminal.

Run this once, from the doe folder, with the doe environment active:
    conda activate doe
    python scripts/make_shortcut.py

On a Mac the icon goes on the Desktop. On Windows it goes on the Desktop and in the Start
menu. Double-click it to start doekit, and choose 'Close doekit' in the app's sidebar to
stop it. If you move the doe folder, run this again.
"""

import argparse
import sys
from pathlib import Path

from doekit.shortcut import ShortcutError, make_shortcut


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, help="put the icon in this folder instead")
    args = parser.parse_args()
    try:
        made = make_shortcut(args.folder)
    except ShortcutError as error:
        print(f"\nProblem: {error}\n", file=sys.stderr)
        raise SystemExit(1) from None
    for path in made:
        print(f"Made: {path}")
    print("Double-click it to start doekit. To stop it, choose 'Close doekit' in the app's sidebar.")


if __name__ == "__main__":
    main()
