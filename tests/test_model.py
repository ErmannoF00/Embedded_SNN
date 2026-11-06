import torch

from snn_pipeline.model import AdaptiveLIF, SurrogateSpike


def test_output_shapes_and_finite_gradient():
    model = AdaptiveLIF()
    features = torch.randn(7, 32)
    score, spikes = model(features)
    assert score.shape == (7, 2)
    assert spikes.shape == (7, 48)
    assert torch.isfinite(score).all()
    loss = torch.nn.functional.cross_entropy(score, torch.arange(7) % 2)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.input_proj.weight.grad.abs().sum().item() > 0


def test_one_window_is_not_affected_by_previous_batch():
    model = AdaptiveLIF().eval()
    first, second = torch.randn(1, 32), torch.randn(1, 32)
    with torch.no_grad():
        alone = model(first)[0]
        model(second)
        again = model(first)[0]
        together = model(torch.cat([first, second]))[0]
    torch.testing.assert_close(alone, again, rtol=0, atol=0)
    torch.testing.assert_close(alone, together[:1], rtol=1e-5, atol=1e-6)


def test_surrogate_derivative_exists_with_hard_forward():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0], requires_grad=True)
    spike = SurrogateSpike.apply(x)
    assert spike.tolist() == [0., 0., 1., 1., 1.]
    spike.sum().backward()
    assert torch.all(x.grad > 0)
