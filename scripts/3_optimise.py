"""Step 3: find the settings that best meet every response's goal."""

import argparse

from doekit.workflow import optimise_step, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the study's config file, for example configs/my_study.yaml")
    args = parser.parse_args()
    run_script(optimise_step, args.config)


if __name__ == "__main__":
    main()
