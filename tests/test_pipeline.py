import numpy as np
import torch
import pytest

from tworoom_gan.env import verify_reference, expert_action, observation, transition, TASK
from tworoom_gan.data import collect, quality_stats, sample_pairs
from tworoom_gan.models import ModelConfig, Generator, TrajectoryEncoder, Discriminator, masked_l1, gaussian_prior_match
from tworoom_gan.rollout_training import differentiable_rollout
from tworoom_gan.evaluate import rollout
from tworoom_gan.style import encode_style, decode_style


def test_transition_matches_original_environment():
    report = verify_reference(samples=128)
    assert report["max_position_error_pixels"] < 3e-5
    assert len(report["expert_rollouts"]) == 4


def test_successful_balanced_experts_and_no_split_leakage():
    a = collect(8, 100, "cpu")
    b = collect(8, 200, "cpu")
    report = quality_stats(a)
    assert report["success_rate"] == 1
    assert report["actual_route_labels_verified"]
    assert report["mode_counts"] == [16, 16]
    assert not set(map(tuple, np.concatenate([a["start"], a["goal"]], -1))) & set(map(tuple, np.concatenate([b["start"], b["goal"]], -1)))


@pytest.mark.parametrize("moments", [False, True])
def test_bicycle_components_backpropagate_and_mask_padding(moments):
    torch.manual_seed(3)
    config = ModelConfig(hidden=64, style_dim=32, encoder_hidden=32, blocks=2, encoder_moments=moments)
    g, e, d = Generator(config), TrajectoryEncoder(config), Discriminator(config)
    data = collect(2, 10, "cpu")
    states = torch.from_numpy(data["states"][:, :-1])
    actions = torch.from_numpy(data["actions"])
    lengths = torch.from_numpy(data["lengths"])
    mu, logvar = e(states, actions, lengths)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    predicted = g(states[:, 0], z)
    assert predicted.shape == (8, 16, 2)
    assert predicted.abs().max() <= 1
    mask = torch.ones(8, 16, dtype=torch.bool)
    score = d(states[:, 0], predicted, mask)
    loss = predicted.square().mean() + score.mean() + logvar.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in g.parameters() if p.grad is not None)
    assert g.mapping.net[0].weight.grad.abs().sum() > 0
    mask[:, 8:] = False
    modified = predicted.detach().clone()
    modified[:, 8:] = 99
    assert masked_l1(predicted.detach(), modified, mask) == 0
    # Padded observations/actions must never tell E anything about episode style.
    valid = torch.arange(states.shape[1])[None] < lengths[:, None]
    changed_states = torch.where(valid[..., None], states, torch.randn_like(states) * 100)
    changed_actions = torch.where(valid[..., None], actions, torch.randn_like(actions) * 100)
    altered_mu, _ = e(changed_states, changed_actions, lengths)
    assert torch.allclose(mu, altered_mu, atol=1e-6)


def test_fixed_latent_and_state_produce_identical_chunks():
    torch.manual_seed(9)
    g = Generator(ModelConfig(hidden=64, blocks=2)).eval()
    start, goal = sample_pairs(4, 4)
    obs = observation(start, goal)
    z = torch.randn(4, 2)
    assert torch.equal(g(obs, z), g(obs, z))
    assert not torch.equal(g(obs, z), g(obs, z + 1))


def test_continuous_geometric_variation_keeps_experts_successful():
    data = collect(16, 41, "cpu", route_offset=12)
    assert quality_stats(data)["actual_route_labels_verified"]
    assert data["offset"].std() > 5
    assert np.all(np.abs(data["offset"]) <= 12)


def test_differentiable_rollout_matches_execution_and_reaches_generator():
    torch.manual_seed(12)
    config = ModelConfig(hidden=32, style_dim=16, blocks=1, encoder_hidden=16)
    g = Generator(config)
    start, goal = sample_pairs(2, 5)
    z = torch.randn(2, 2)
    states, actions, lengths = differentiable_rollout(g, start, goal, z, max_steps=16)
    actual = rollout(g, start, goal, z)
    assert torch.allclose(states, actual["states"][:, :16])
    assert torch.allclose(actions, actual["actions"][:, :16])
    (states.square().mean() + actions.square().mean()).backward()
    grad = g.mapping.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_auxiliary_gaussian_style_coordinates_roundtrip():
    z = torch.tensor([[-2.5, -2.0], [-0.01, 1.0], [0.0, 0.0], [0.5, -0.5], [2.0, 2.5]])
    mode, speed, offset = decode_style(z)
    restored = encode_style(mode, speed, offset)
    assert torch.allclose(restored, z, atol=2e-4)
    assert ((speed >= 0.55) & (speed <= 1)).all()
    assert (offset.abs() <= 12).all()


def test_reflection_equivariance_of_generator_and_encoder():
    config = ModelConfig(hidden=32, style_dim=16, blocks=1, encoder_hidden=16, canonical=True, route_feature=True, encoder_moments=True)
    g, e = Generator(config).eval(), TrajectoryEncoder(config).eval()
    data = collect(2, 7, "cpu", route_offset=12)
    states = torch.tensor(data["states"][:, :-1])
    actions = torch.tensor(data["actions"])
    lengths = torch.tensor(data["lengths"])
    reflected_states, reflected_actions = states.clone(), actions.clone()
    reflected_states[..., 0] *= -1
    reflected_states[..., 2] *= -1
    reflected_actions[..., 0] *= -1
    z = torch.randn(len(states), 2)
    a, b = g(states[:, 0], z), g(reflected_states[:, 0], z)
    assert torch.allclose(a[..., 0], -b[..., 0])
    assert torch.allclose(a[..., 1], b[..., 1])
    mu, lv = e(states, actions, lengths)
    reflected_mu, reflected_lv = e(reflected_states, reflected_actions, lengths)
    assert torch.allclose(mu, reflected_mu)
    assert torch.allclose(lv, reflected_lv)


def test_aggregate_prior_regularizer_detects_collapse():
    torch.manual_seed(81)
    gaussian = torch.randn(1024, 2, requires_grad=True)
    good = gaussian_prior_match(gaussian)
    collapsed = gaussian_prior_match(torch.zeros_like(gaussian))
    assert good < 0.05 and collapsed > 0.9
    good.backward()
    assert torch.isfinite(gaussian.grad).all() and gaussian.grad.abs().sum() > 0
