"""Compare designs side by side before committing to one.

Example:
    python scripts/compare_designs.py configs/my_study.yaml box_behnken central_composite i_optimal:runs=30
"""

import argparse

from doekit.workflow import compare_step, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the study's config file")
    parser.add_argument("designs", nargs="+", help="designs to compare, like box_behnken or i_optimal:runs=30")
    parser.add_argument("--signal-to-noise", type=float, default=2.0, help="the effect to detect, as a multiple of the noise")
    args = parser.parse_args()
    run_script(compare_step, args.config, args.designs, args.signal_to_noise)


if __name__ == "__main__":
    main()
