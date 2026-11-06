"""Add the most useful new runs to a study you've already started.

Your existing runs and results stay as they are, and a copy of the old run sheet is kept.
The runs are chosen for the model in the config, so to add curvature to a screening study,
change its model to quadratic first.

Examples:
    python scripts/add_runs.py configs/my_study.yaml --runs 8
    python scripts/add_runs.py configs/my_study.yaml --runs 8 --repeats 2 --region pressure=60:90
"""

import argparse

from doekit.study import StudyError
from doekit.workflow import add_runs_step, run_script


def parse_region(items):
    region = {}
    for item in items or []:
        try:
            name, limits = item.split("=")
            low, high = limits.split(":")
            region[name.strip()] = (float(low), float(high))
        except ValueError:
            raise StudyError(f"Couldn't read the region '{item}'. Write it like pressure=60:90.") from None
    return region or None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the study's config file, for example configs/my_study.yaml")
    parser.add_argument("--runs", type=int, required=True, help="how many new runs to add, including any repeats")
    parser.add_argument("--goal", choices=["predictions", "effects"], default="predictions", help="what the new runs should improve most")
    parser.add_argument("--repeats", type=int, default=0, help="how many of the new runs repeat earlier ones")
    parser.add_argument("--same-session", action="store_true", help="the new runs won't be done separately, so don't add a new block")
    parser.add_argument("--region", action="append", metavar="FACTOR=LOW:HIGH", help="limit the new runs to part of a factor's range (can be repeated)")
    args = parser.parse_args()
    criterion = "I" if args.goal == "predictions" else "D"

    def add():
        return add_runs_step(args.config, args.runs, criterion, args.repeats, not args.same_session, parse_region(args.region))

    run_script(add)


if __name__ == "__main__":
    main()
