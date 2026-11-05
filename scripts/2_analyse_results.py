"""Step 2: fit and check a model for each response, once results are in the run sheet."""

import argparse

from doekit.workflow import analyse_step, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="the study's config file, for example configs/my_study.yaml")
    args = parser.parse_args()
    run_script(analyse_step, args.config)


if __name__ == "__main__":
    main()
