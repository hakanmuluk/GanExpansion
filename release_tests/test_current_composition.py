"""Current public interface: immutable checkpoints and single-start replanning."""
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.compose import LoadedPolicy, configuration


@pytest.mark.parametrize("environment", ["tworoom", "cube"])
def test_only_initialization_and_budget_change(environment):
    original = configuration(environment, "grid", 20)
    for n in (0, 5, 10, 20, 40):
        single = configuration(environment, "single-gaussian", n)
        assert single["starts"] == 1 and single["steps"] == n
        assert {k: v for k, v in single.items() if k not in ("starts", "steps")} == {
            k: v for k, v in original.items() if k not in ("starts", "steps")}


def test_grid_budget_cannot_silently_change_the_original_reference():
    with pytest.raises(ValueError, match="original reference"):
        configuration("cube", "grid", 5)


@pytest.fixture(params=["tworoom", "cube"])
def policy(request):
    root = Path(__file__).resolve().parents[1]
    if not (root / "research_assets/composition" / request.param / "base.pt").exists():
        pytest.skip("Download --manifest configs/research_assets.json --bundle composition")
    return LoadedPolicy(request.param, steps=5, seed=241)


def observation(policy):
    dim, context = (10, 12) if policy.environment == "tworoom" else (47, 20)
    state = np.zeros((2, dim), np.float32)
    if policy.environment == "cube":
        state += policy.bank.k.base.obs_mean.detach().cpu().numpy()
    return {"state": state, "base_z": np.array([[1., .2], [-1., -.3]], np.float32),
            "keepout_context": np.zeros((2, context), np.float32),
            "waypoint_context": np.zeros((2, context), np.float32)}


def test_all_four_cases_preserve_frozen_models(policy):
    x = observation(policy)
    expected = (2, 16, 2 if policy.environment == "tworoom" else 5)
    for case in ("base", "keepout", "waypoint", "compose"):
        actions = policy.predict(x, case)
        assert tuple(actions.shape) == expected and torch.isfinite(actions).all()
        assert all(not p.requires_grad and p.grad is None for p in policy.bank.parameters())
        policy.bank.assert_frozen()


def test_one_gaussian_draw_is_retained_across_replans(policy):
    x = observation(policy)
    policy.predict(x)
    initial = policy.planner.gaussian.clone()
    previous = policy.planner.previous.clone()
    x["state"] = x["state"] + .001
    policy.predict(x)
    assert policy.planner.gaussian_draw_calls == 1
    assert policy.planner.replans == 2
    assert torch.equal(policy.planner.gaussian, initial)
    assert torch.equal(policy.planner._warm_input, previous)


def test_same_initial_gaussian_for_zero_and_five_steps(policy):
    x = observation(policy)
    zero = LoadedPolicy(policy.environment, steps=0, seed=241)
    policy.predict(x)
    zero.predict(x)
    assert torch.equal(policy.planner.gaussian, zero.planner.gaussian)
    assert torch.equal(policy.planner.noise, zero.planner.noise)
    assert torch.allclose(zero.planner.previous, zero.planner.decode(
        zero.planner.parameterize(zero.planner.noise[:, 0]),
        zero.planner.noise[:, 0, :2].sign()), atol=0, rtol=0)


def test_nonfinite_input_is_rejected(policy):
    x = observation(policy)
    x["state"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.predict(x)


def test_changed_base_code_cannot_reuse_previous_episode_inputs(policy):
    x = observation(policy)
    policy.predict(x)
    x["base_z"][0, 1] += .1
    with pytest.raises(ValueError, match="original base code"):
        policy.predict(x)


def test_published_csv_budgets_are_distinct_from_episode_lengths():
    import csv
    root = Path(__file__).resolve().parents[1]
    for name, count in (("results.csv", 30), ("seed_results.csv", 90)):
        rows = list(csv.DictReader((root / "benchmarks/single_gaussian/aggregate" / name).open()))
        assert len(rows) == count
        assert {int(r["steps"]) for r in rows} == {0, 5, 10, 20, 40}
        assert all(float(r["episode_steps"]) > 0 for r in rows)
