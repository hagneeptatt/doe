"""Fill a run sheet with made-up results from the config's simulation section, for practice."""

import argparse

from doekit.workflow import run_script, simulate_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the study's config file, which must have a simulation section")
    parser.add_argument("--force", action="store_true", help="replace any results already in the run sheet")
    args = parser.parse_args()
    run_script(simulate_step, args.config, force=args.force)


if __name__ == "__main__":
    main()
