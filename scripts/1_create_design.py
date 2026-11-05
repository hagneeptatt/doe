"""Step 1: generate a design and write its run sheet to data/<study name>/runsheet.csv."""

import argparse

from doekit.designs import DESIGNS
from doekit.workflow import create_design_step, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="the study's config file, for example configs/my_study.yaml")
    parser.add_argument("--force", action="store_true", help="replace an existing run sheet, losing any results in it")
    parser.add_argument("--list", action="store_true", help="list the available design types and stop")
    args = parser.parse_args()
    if args.list:
        for entry in DESIGNS.values():
            print(f"{entry.name:<22} {entry.purpose:<20} {entry.summary}")
        return
    if not args.config:
        parser.error("give a config file, or use --list")
    run_script(create_design_step, args.config, force=args.force)


if __name__ == "__main__":
    main()
