"""Check that every design has the structure it should have."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from doekit.designs import DESIGNS, create_design
from doekit.designs.factorial import FRACTIONAL_GENERATORS, resolution_of
from doekit.designs.mixture import centroid_points, lattice_points
from doekit.designs.optimal import _swap_gains, candidate_points, design_score, exchange_search
from doekit.designs.response_surface import doehlert_matrix
from doekit.evaluate import evaluate_design, term_power
from doekit.model import model_matrix
from doekit.study import StudyError, mixture_vertices, study_from_config


def numeric(count):
    return [{"name": f"x{i + 1}", "low": 0, "high": 10} for i in range(count)]


def build(factors, seed=1, model=None, **design):
    config = {"name": "t", "seed": seed, "factors": factors, "design": design}
    if model:
        config["model"] = model
    study = study_from_config(config)
    return study, create_design(study)


MIXTURE = [{"name": n, "type": "mixture"} for n in "abc"]
BOUNDED = [
    {"name": "a", "type": "mixture", "low": 0.2, "high": 0.7},
    {"name": "b", "type": "mixture", "low": 0.1, "high": 0.5},
    {"name": "c", "type": "mixture", "low": 0.1, "high": 0.4},
]
PROCESS = [{"name": "p", "low": 1, "high": 5}, {"name": "q", "low": 0, "high": 2}]


@pytest.mark.parametrize(
    "factors, settings, runs",
    [
        (numeric(3), {"type": "full_factorial"}, 8),
        (numeric(3), {"type": "full_factorial", "levels": 3}, 27),
        (numeric(3), {"type": "full_factorial", "center_points": 3}, 11),
        (numeric(5), {"type": "fractional_factorial", "resolution": 5}, 16),
        (numeric(7), {"type": "fractional_factorial", "resolution": 3}, 8),
        (numeric(7), {"type": "plackett_burman"}, 8),
        (numeric(11), {"type": "plackett_burman"}, 12),
        (numeric(3), {"type": "box_behnken"}, 15),
        (numeric(4), {"type": "box_behnken", "center_points": 3}, 27),
        (numeric(5), {"type": "box_behnken", "center_points": 6}, 46),
        (numeric(3), {"type": "central_composite"}, 18),
        (numeric(2), {"type": "central_composite", "center_points": 5}, 13),
        (numeric(3), {"type": "doehlert", "center_points": 2}, 15),
        (numeric(3), {"type": "taguchi"}, 9),
        (numeric(3), {"type": "gsd", "levels": 3, "reduction": 3}, 9),
        (numeric(3), {"type": "latin_hypercube", "runs": 20}, 20),
        (numeric(3), {"type": "sobol", "runs": 16}, 16),
        (MIXTURE, {"type": "simplex_lattice", "degree": 2}, 6),
        (MIXTURE, {"type": "simplex_lattice", "degree": 3}, 10),
        (MIXTURE, {"type": "simplex_lattice", "degree": 2, "check_points": True}, 10),
        (MIXTURE, {"type": "simplex_centroid"}, 7),
    ],
)
def test_run_counts(factors, settings, runs):
    _, design = build(factors, **settings)
    assert len(design.frame) == runs


def test_every_registered_design_is_covered_here():
    tested = {
        "full_factorial", "fractional_factorial", "plackett_burman", "box_behnken", "central_composite",
        "doehlert", "taguchi", "gsd", "latin_hypercube", "sobol", "simplex_lattice", "simplex_centroid",
        "extreme_vertices", "crossed", "d_optimal", "i_optimal",
    }  # fmt: skip
    assert tested == set(DESIGNS)


@pytest.mark.parametrize("count, table", FRACTIONAL_GENERATORS.items())
def test_tabulated_resolutions_are_correct(count, table):
    for claimed, generators in table:
        assert len(generators.split()) == count
        assert resolution_of(generators) == claimed


def test_two_level_designs_are_orthogonal():
    for settings, count in [
        ({"type": "full_factorial"}, 4),
        ({"type": "fractional_factorial", "resolution": 4}, 6),
        ({"type": "plackett_burman"}, 7),
    ]:
        study, design = build(numeric(count), **settings)
        coded = np.column_stack([study.factor(n).code(design.frame[n]) for n in study.factor_names])
        product = coded.T @ coded
        assert product == pytest.approx(len(coded) * np.eye(count)), settings


def test_box_behnken_never_uses_every_factor_at_an_extreme():
    study, design = build(numeric(3), type="box_behnken")
    coded = np.column_stack([study.factor(n).code(design.frame[n]) for n in study.factor_names])
    assert not np.any(np.all(np.abs(coded) == 1, axis=1))


def test_central_composite_variants_stay_where_they_should():
    for variant, inside in [("face_centred", True), ("inscribed", True), ("circumscribed", False)]:
        study, design = build(numeric(3), type="central_composite", variant=variant)
        within = design.frame[study.factor_names].to_numpy()
        assert ((within >= -1e-6).all() and (within <= 10 + 1e-6).all()) == inside, variant


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_doehlert_is_a_uniform_shell(count):
    points = doehlert_matrix(count)
    assert len(points) == count**2 + count + 1
    assert np.linalg.norm(points[1:], axis=1) == pytest.approx(np.ones(len(points) - 1))


def test_latin_hypercube_uses_each_slice_once():
    study, design = build(numeric(3), type="latin_hypercube", runs=20)
    for name in study.factor_names:
        slices = np.floor(design.frame[name].to_numpy() / 10 * 20).astype(int)
        # Rounding can only nudge a value across a slice boundary, not by a whole slice.
        assert len(np.unique(slices)) >= 19


def test_lattice_and_centroid_blends():
    assert len(lattice_points(3, 2)) == 6
    assert len(lattice_points(4, 3)) == 20
    assert lattice_points(3, 4).sum(axis=1) == pytest.approx(np.ones(15))
    assert len(centroid_points(3)) == 7
    assert len(centroid_points(4)) == 15


def test_mixture_vertices_for_a_known_region():
    vertices = mixture_vertices([0.2, 0.1, 0.1], [0.7, 0.5, 0.4], 1.0)
    assert vertices.sum(axis=1) == pytest.approx(np.ones(len(vertices)))
    expected = {(0.7, 0.2, 0.1), (0.7, 0.1, 0.2), (0.5, 0.1, 0.4), (0.2, 0.4, 0.4), (0.2, 0.5, 0.3), (0.4, 0.5, 0.1)}
    assert {tuple(float(x) for x in np.round(v, 6)) for v in vertices} == expected


@pytest.mark.parametrize("kind", ["extreme_vertices", "d_optimal", "i_optimal"])
def test_bounded_mixtures_stay_within_limits_and_add_up(kind):
    study, design = build(BOUNDED, type=kind)
    frame = design.frame
    assert frame[list("abc")].sum(axis=1).to_numpy() == pytest.approx(np.ones(len(frame)), abs=1e-6)
    assert study.is_feasible(frame).all()


def test_mixture_lattice_is_refused_when_limits_cut_the_simplex():
    with pytest.raises(StudyError, match="extreme_vertices"):
        build(BOUNDED, type="simplex_lattice")


def test_crossed_design_runs_every_blend_at_every_setting():
    settings = {"mixture": {"type": "simplex_lattice", "degree": 2}, "process": {"type": "full_factorial"}}
    _, design = build(MIXTURE + PROCESS, type="crossed", **settings)
    assert len(design.frame) == 6 * 4


def test_classical_designs_refuse_factors_they_cannot_handle():
    categorical = {"name": "c", "levels": ["x", "y", "z"]}
    with pytest.raises(StudyError, match="optimal"):
        build(numeric(3) + [categorical], type="box_behnken")
    with pytest.raises(StudyError, match="optimal"):
        build(MIXTURE, type="central_composite")
    with pytest.raises(StudyError, match="two settings"):
        build(numeric(3) + [categorical], type="plackett_burman")


def test_classical_design_refused_when_it_breaks_a_constraint():
    config = {
        "name": "t", "seed": 1, "factors": numeric(3), "design": {"type": "box_behnken"},
        "constraints": [{"expr": "x1 + x2", "max": 12}],
    }  # fmt: skip
    with pytest.raises(StudyError, match="constraints"):
        create_design(study_from_config(config))


def test_optimal_design_respects_constraints():
    config = {
        "name": "t", "seed": 1, "factors": numeric(3), "design": {"type": "i_optimal", "runs": 16},
        "constraints": [{"expr": "x1 + x2", "max": 12}],
    }  # fmt: skip
    study = study_from_config(config)
    design = create_design(study)
    assert (design.frame["x1"] + design.frame["x2"] <= 12 + 1e-9).all()


def test_too_large_designs_get_a_clear_message():
    with pytest.raises(StudyError, match="fractional_factorial"):
        build(numeric(20), type="full_factorial")
    with pytest.raises(StudyError, match="i_optimal"):
        build(numeric(12), type="central_composite")


def test_unknown_settings_are_caught_rather_than_ignored():
    with pytest.raises(StudyError, match="center_point"):
        build(numeric(3), type="box_behnken", center_point=3)
    with pytest.raises(StudyError, match="Unknown design type"):
        build(numeric(3), type="bbd")


def test_same_seed_recreates_the_same_design():
    _, first = build(BOUNDED + PROCESS, seed=11, type="i_optimal", runs=26, blocks=2)
    _, second = build(BOUNDED + PROCESS, seed=11, type="i_optimal", runs=26, blocks=2)
    pd.testing.assert_frame_equal(first.frame, second.frame)
    _, other = build(BOUNDED + PROCESS, seed=12, type="i_optimal", runs=26, blocks=2)
    assert not first.frame.equals(other.frame)


def test_replicate_runs_are_exact_repeats():
    study, design = build(BOUNDED + PROCESS, type="d_optimal", runs=26, replicate_runs=4)
    assert len(design.frame) == 26
    assert design.frame.duplicated(subset=study.factor_names).sum() >= 4


def test_blocks_are_balanced_and_grouped_in_run_order():
    _, design = build(numeric(3), type="central_composite", blocks=3)
    counts = design.frame["block"].value_counts()
    assert counts.max() - counts.min() <= 1
    assert design.frame["block"].is_monotonic_increasing


def test_runs_are_randomised_unless_told_otherwise():
    _, randomised = build(numeric(4), type="full_factorial")
    assert not randomised.frame["std_order"].is_monotonic_increasing
    assert list(randomised.frame["run"]) == list(range(1, 17))
    _, ordered = build(numeric(4), type="full_factorial", randomise=False)
    assert ordered.frame["std_order"].is_monotonic_increasing
    assert any("NOT randomised" in note for note in ordered.notes)


def test_rounding_keeps_blends_on_their_total():
    _, design = build(MIXTURE, type="simplex_lattice", degree=3)
    assert design.frame[list("abc")].sum(axis=1).to_numpy() == pytest.approx(np.ones(10), abs=1e-9)


# ------------------------------------------------------------------ the exchange algorithm
def _small_problem():
    study = study_from_config({"name": "t", "factors": numeric(2), "model": "quadratic"})
    grid = pd.DataFrame(list(itertools.product([0.0, 5.0, 10.0], repeat=2)), columns=["x1", "x2"])
    terms = [("x1",), ("x2",), ("x1", "x2"), ("x1", "x1"), ("x2", "x2")]
    return model_matrix(study, grid, terms, include_blocks=False).values


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_exchange_finds_the_true_optimum_of_a_small_problem(criterion):
    """Compare with brute force over every possible 7-run subset of a 3 x 3 grid."""
    C = _small_problem()
    moments = C.T @ C / len(C)
    brute = min(
        design_score(C, np.array(subset), criterion, moments)
        for subset in itertools.combinations(range(9), 7)
    )
    chosen = exchange_search(C, 7, criterion, np.random.default_rng(0), starts=30)
    assert design_score(C, chosen, criterion, moments) == pytest.approx(brute, rel=1e-6)


@pytest.mark.parametrize("criterion", ["D", "I"])
def test_swap_gain_formula_matches_direct_calculation(criterion):
    """The fast update formulas must agree with refitting the design for each swap."""
    C = _small_problem()
    moments = C.T @ C / len(C)
    chosen = np.array([0, 1, 2, 3, 4, 5, 6, 8])
    gains = _swap_gains(C, chosen, criterion, moments)
    before = design_score(C, chosen, criterion, moments)
    for i, j in [(0, 7), (3, 7), (7, 7), (2, 5)]:
        trial = chosen.copy()
        trial[i] = j
        after = design_score(C, trial, criterion, moments)
        if criterion == "D":
            assert 1 + gains[i, j] == pytest.approx(np.exp(before - after), rel=1e-5)
        else:
            assert gains[i, j] == pytest.approx(before - after, rel=1e-4, abs=1e-6)


def test_d_optimal_quadratic_mixture_is_the_simplex_lattice():
    """A known theoretical result (Kiefer, 1961) for 6 runs on a full simplex."""
    study = study_from_config(
        {
            "name": "t",
            "seed": 2,
            "factors": MIXTURE,
            "design": {"type": "d_optimal", "runs": 6, "replicate_runs": 0},
            "model": {"mixture": "quadratic"},
        }
    )
    frame = create_design(study).frame[list("abc")].round(3)
    points = {tuple(float(x) for x in row) for row in frame.to_numpy()}
    assert points == {tuple(float(x) for x in point) for point in lattice_points(3, 2).round(3)}


def test_i_optimal_predicts_better_and_d_optimal_estimates_better():
    config = {"name": "t", "seed": 5, "factors": numeric(3), "model": "quadratic"}
    d_study = study_from_config({**config, "design": {"type": "d_optimal", "runs": 16, "replicate_runs": 0}})
    i_study = study_from_config({**config, "design": {"type": "i_optimal", "runs": 16, "replicate_runs": 0}})
    d_quality = evaluate_design(d_study, create_design(d_study).frame)
    i_quality = evaluate_design(i_study, create_design(i_study).frame)
    # I-optimality minimises the average prediction variance, so compare exactly that.
    assert np.mean(i_quality.prediction_se**2) <= np.mean(d_quality.prediction_se**2) + 1e-9
    assert d_quality.d_efficiency >= i_quality.d_efficiency - 1e-9


def test_optimal_design_refuses_too_few_runs():
    with pytest.raises(StudyError, match="at least"):
        build(numeric(3), type="d_optimal", runs=8, model="quadratic")


def test_candidates_handle_many_factors_without_running_out_of_memory():
    study = study_from_config({"name": "t", "seed": 1, "factors": numeric(25), "model": "linear", "design": {"type": "d_optimal"}})
    frame = candidate_points(study, {}, np.random.default_rng(0))
    assert 0 < len(frame) <= 10000


# ------------------------------------------------------------------ evaluating designs
def test_design_quality_counts_degrees_of_freedom():
    study, design = build(numeric(3), type="box_behnken", center_points=3)
    quality = evaluate_design(study, design.frame, design.terms)
    assert quality.terms == 10
    assert quality.residual_df == 5
    assert quality.pure_error_df == 2
    assert quality.lack_of_fit_df == 3


def test_replicates_split_across_blocks_still_count_as_replicates():
    study, design = build(BOUNDED + PROCESS, seed=42, type="i_optimal", runs=30, replicate_runs=4, blocks=2)
    quality = evaluate_design(study, design.frame, design.terms)
    assert quality.distinct_runs <= 26
    assert quality.pure_error_df >= 3
    assert not any("No runs are replicated" in warning for warning in quality.warnings)


def test_design_quality_flags_a_model_the_design_cannot_estimate():
    study, design = build(numeric(5), type="fractional_factorial", resolution=3, model="interaction")
    quality = evaluate_design(study, design.frame, design.terms)
    assert not quality.estimable
    assert quality.aliases
    assert "can't be told apart" in quality.warnings[0]


def test_power_behaves_sensibly():
    low = term_power(np.array([0.5]), 10, 1.0)[0]
    high = term_power(np.array([0.5]), 10, 4.0)[0]
    assert 0.05 < low < high < 1
    assert np.isnan(term_power(np.array([0.5]), 0, 2.0)[0])


def test_power_matches_exact_value_for_a_factorial():
    # 2^3 replicated twice (16 runs, 8 terms): se = 1/4, df = 8. Detecting a change of
    # 2 noise SDs means a coefficient of 1, so the noncentrality is 4.
    terms = ["x1", "x2", "x3", "x1*x2", "x1*x3", "x2*x3", "x1*x2*x3"]
    study, design = build(numeric(3), type="full_factorial", repeats=2, model={"terms": terms})
    quality = evaluate_design(study, design.frame, design.terms)
    table = quality.term_table.set_index("term")
    assert table.loc["x1", "standard_error"] == pytest.approx(0.25)
    assert quality.residual_df == 8
    critical = stats.t.ppf(0.975, 8)
    expected = 1 - stats.nct.cdf(critical, 8, 4) + stats.nct.cdf(-critical, 8, 4)
    assert table.loc["x1", "power"] == pytest.approx(expected)


def test_power_is_not_reported_for_mixture_terms():
    study, design = build(BOUNDED + PROCESS, type="i_optimal", runs=30)
    quality = evaluate_design(study, design.frame, design.terms)
    table = quality.term_table.set_index("term")
    assert np.isnan(table.loc["a", "power"])
    assert np.isnan(table.loc["a:b", "power"])
    assert np.isnan(table.loc["a:p", "power"])
    assert np.isfinite(table.loc["p:q", "power"])
