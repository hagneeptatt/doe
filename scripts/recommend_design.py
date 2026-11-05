"""Suggest designs for a study, with the reasons and real run counts."""

import argparse

from doekit.advisor import OBJECTIVES
from doekit.workflow import recommend_step, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the study's config file; its design section is ignored")
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="optimise", help="what you want to learn")
    parser.add_argument("--max-runs", type=int, help="the most runs you can afford")
    parser.add_argument("--avoid-extremes", action="store_true", help="prefer designs without every factor at an extreme")
    parser.add_argument("--signal-to-noise", type=float, default=2.0, help="the effect to detect, as a multiple of the noise")
    args = parser.parse_args()
    run_script(recommend_step, args.config, args.objective, args.max_runs, args.avoid_extremes, args.signal_to_noise)


if __name__ == "__main__":
    main()
