#!/usr/bin/env bash
# Installs doekit for you, or updates it to the latest version. No administrator rights
# are needed. On a Mac, paste this into Terminal:
#
#   curl -fsSL https://raw.githubusercontent.com/hagneeptatt/doe/main/install.sh | bash
#
# Everything goes in one folder, doekit in your home folder: a private copy of conda
# (Miniforge), the packages doekit needs, and doekit itself. Nothing else on your
# computer is changed. Running it again updates doekit and keeps your saved studies.
# To remove doekit, delete that folder and the doekit icon.
#
# Settings, mainly for testing: DOEKIT_HOME (where to install), DOEKIT_ARCHIVE (a .tar.gz
# of the code to use instead of the latest release), DOEKIT_ICON_FOLDER (where to put the
# icon) and DOEKIT_NO_START (don't open doekit at the end).

set -euo pipefail

work=""

say() { printf '\n==> %s\n' "$1"; }
fail() {
  printf '\nProblem: %s\n' "$1" >&2
  exit 1
}

main() {
  local repo="hagneeptatt/doe"
  local home="${DOEKIT_HOME:-$HOME/doekit}"
  local code="$home/doe" conda="$home/miniforge" env="$home/env"
  local system machine archive tag icons

  case "$(uname -s)" in
    Darwin) system="MacOSX" ;;
    Linux) system="Linux" ;;
    *) fail "This installer is for Macs and Linux. On Windows, follow the instructions for Windows." ;;
  esac
  machine="$(uname -m)"
  icons="${DOEKIT_ICON_FOLDER:-$HOME/Desktop}"
  mkdir -p "$home"
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  say "Installing doekit in $home"

  # 1. A private copy of conda, so nothing else on this computer is affected.
  if [ ! -x "$conda/bin/conda" ]; then
    say "Downloading Miniforge (about 100 MB)"
    curl -fL --progress-bar -o "$work/miniforge.sh" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$system-$machine.sh" \
      || fail "Couldn't download Miniforge. Check your internet connection, then try again."
    bash "$work/miniforge.sh" -b -p "$conda" </dev/null >"$work/miniforge.log" 2>&1 \
      || { tail -n 20 "$work/miniforge.log" >&2; fail "Miniforge didn't install."; }
  fi

  # 2. The latest released version of doekit.
  say "Downloading doekit"
  archive="${DOEKIT_ARCHIVE:-}"
  if [ -z "$archive" ]; then
    tag="$(curl -fsSL "https://api.github.com/repos/$repo/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)" || true
    [ -n "$tag" ] || fail "Couldn't find the latest version of doekit. Check your internet connection, then try again."
    archive="https://github.com/$repo/archive/refs/tags/$tag.tar.gz"
  fi
  mkdir "$work/new"
  if [ -f "$archive" ]; then
    tar -xzf "$archive" -C "$work/new" --strip-components 1
  else
    curl -fsSL "$archive" | tar -xz -C "$work/new" --strip-components 1 || fail "Couldn't download doekit."
  fi
  [ -f "$work/new/app/streamlit_app.py" ] || fail "The download didn't contain doekit."

  # 3. Close doekit if it's running, then swap in the new version. Saved studies and
  #    results come along, and the previous version is kept, just in case.
  if [ -d "$code" ]; then
    say "Updating your copy of doekit. Your saved studies are kept"
    if [ -x "$env/bin/python" ]; then
      (cd "$code" && "$env/bin/python" -m doekit.launcher --stop </dev/null >/dev/null 2>&1) || true
    fi
    for keep in data results; do
      if [ -d "$code/$keep" ]; then
        rm -rf "${work:?}/new/$keep"
        cp -R "$code/$keep" "$work/new/$keep"
      fi
    done
    rm -rf "$home/doe-previous"
    mv "$code" "$home/doe-previous"
  fi
  mv "$work/new" "$code"

  # 4. The packages doekit needs, in an environment of their own.
  say "Installing the packages doekit needs. The first time, this takes about 5 to 10 minutes"
  cd "$code"
  if [ -x "$env/bin/python" ]; then
    "$conda/bin/conda" env update -p "$env" -f environment.yml --prune </dev/null \
      || fail "Couldn't update the packages doekit needs. Check your internet connection, then try again."
  else
    "$conda/bin/conda" env create -p "$env" -f environment.yml </dev/null \
      || fail "Couldn't install the packages doekit needs. Check your internet connection, then try again."
  fi

  # 5. The icon, then start doekit.
  say "Making the doekit icon"
  if [ -n "${DOEKIT_ICON_FOLDER:-}" ]; then
    "$env/bin/python" scripts/make_shortcut.py --folder "$DOEKIT_ICON_FOLDER" </dev/null
  else
    "$env/bin/python" scripts/make_shortcut.py </dev/null  # on Linux, also adds it to the applications menu
  fi || fail "Couldn't make the icon. You can still start doekit with: $env/bin/python -m doekit.launcher --root $code"

  say "doekit is installed"
  echo "From now on, double-click the doekit icon on your Desktop to start it."
  echo "To update doekit later, run the same command again. Your saved studies are kept."
  if [ -z "${DOEKIT_NO_START:-}" ]; then
    if [ "$system" = "MacOSX" ] && [ -d "$icons/doekit.app" ]; then
      open "$icons/doekit.app"
    else
      (nohup "$env/bin/python" -m doekit.launcher --root "$code" </dev/null >/dev/null 2>&1 &)
    fi
    echo "doekit is opening in your browser now. You can close this window."
  fi
}

# Everything is inside main, so nothing runs until the whole script has downloaded.
main "$@"
