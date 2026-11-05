# doekit

Design, analyse and optimise experiments for any process, using a web app or from a single config file.

doekit takes you through a whole design of experiments (DOE) study. It helps you choose a design, gives you a randomised run sheet, fits and checks a model for each response, and finds the settings that best meet your goals. It explains its results in plain language, so you don't need a statistics background to use it.

**New to DOE, or choosing a design? Start with the [manual](docs/manual.md).** It also lists [the methods behind each step](docs/manual.md#11-methods-behind-each-step), with their sources.

## Quick start

```bash
git clone https://github.com/hagneeptatt/doe.git
cd doe
conda env create -f environment.yml
conda activate doe
streamlit run app/streamlit_app.py --server.address localhost
```

The app opens in your browser and stays private to your computer. Your work is saved automatically as you go, and you can carry on with a saved study from step 1. In step 1, load the worked example to see a complete study with simulated results.

Or run the same example with scripts:

```bash
python scripts/1_create_design.py   configs/examples/hydrogel_bioink_simulated.yaml
python scripts/simulate_results.py  configs/examples/hydrogel_bioink_simulated.yaml
python scripts/2_analyse_results.py configs/examples/hydrogel_bioink_simulated.yaml
python scripts/3_optimise.py        configs/examples/hydrogel_bioink_simulated.yaml
```

Reports and figures appear in `results/hydrogel_bioink_simulated/`.

## What it can do

| Purpose | Designs |
|---|---|
| **Screening** | Full factorial, fractional factorial, Plackett-Burman, Taguchi, generalized subset |
| **Optimising** | Central composite, Box-Behnken, Doehlert |
| **Blends** | Simplex lattice, simplex centroid, extreme vertices, crossed mixture-process |
| **Any factors or budget** | D-optimal, I-optimal |
| **Exploring** | Latin hypercube, Sobol |

It handles continuous, discrete, categorical and mixture factors, constraints, blocks and several responses at once. Analysis includes ANOVA, lack-of-fit testing, predicted R², model reduction, Lenth's method and residual checks, and optimisation uses desirability functions. Predictions come with intervals, so you can check them with confirmation runs.

## Your own study

Use the app, or copy the closest file from `configs/templates/` and edit it:

```bash
cp configs/templates/mixture_process_i_optimal.yaml configs/my_study.yaml
python scripts/recommend_design.py configs/my_study.yaml --objective optimise --max-runs 30
python scripts/1_create_design.py  configs/my_study.yaml
```

Run sheets go to `data/my_study/`, and they hold your results. Git ignores that folder to keep them off GitHub, so back them up yourself. `1_create_design.py` won't overwrite one that already exists.

## Project layout

```
app/                The web app
configs/
  templates/        Starting points to copy, one for each kind of study
  examples/         Worked examples with simulated results
data/               Run sheets, one folder per study
docs/               The manual, and the hosting guide
results/            Generated reports and figures, one folder per study
scripts/            The command-line steps
src/doekit/         The toolkit
  study.py            Factors, responses and config files
  designs/            Every design type
  model.py            Turns a study into a regression model
  evaluate.py         Judges a design before any runs are done
  advisor.py          Recommends designs
  runsheet.py         Reads and writes run sheets
  analysis.py         Fits and checks models
  optimise.py         Finds the best settings
  plots.py, report.py Figures and written reports
  simulate.py         Made-up results, for practice
tests/              Automatic checks, including published textbook results
archive/            The original analysis script
```

## Testing

```bash
pytest
```

The tests compare results with published examples, check the design algorithms against brute force, run every template from start to finish, and click through the app. See [how doekit was checked](docs/manual.md#12-how-doekit-was-checked).

## Sharing it

To host the app for other people, see the [hosting guide](docs/hosting.md).

## Licence

doekit is released under the [MIT License](LICENSE). Copyright (c) 2026 Fraser Shields.
