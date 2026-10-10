import torch

from novalm.model.config import ModelConfig
from novalm.model.layers.rmsnorm import RMSNorm
from novalm.model.layers.rope import RotaryEmbedding
from novalm.model.layers.swiglu import SwiGLU
from novalm.model.transformer import NovaLM, causal_lm_loss


def tiny_config(**overrides):
    values = {"vocab_size": 64, "hidden_size": 32, "num_layers": 2,
              "num_attention_heads": 4, "num_kv_heads": 2, "intermediate_size": 48,
              "max_seq_length": 16}
    values.update(overrides)
    return ModelConfig(**values)


def test_rmsnorm_matches_reference_and_has_gradients():
    x = torch.randn(2, 3, 8, requires_grad=True)
    layer = RMSNorm(8, eps=1e-5)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(layer(x), expected)
    layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rope_extends_cache_and_preserves_shape():
    rope = RotaryEmbedding(8, max_seq_length=2)
    out = rope(torch.randn(1, 2, 5, 8))
    assert out.shape == (1, 2, 5, 8)
    assert torch.isfinite(out).all()
    half_rope = RotaryEmbedding(8, max_seq_length=2).half()
    half_out = half_rope(torch.randn(1, 1, 5, 8).half())
    assert half_out.dtype == torch.float16 and torch.isfinite(half_out).all()


def test_swiglu_shape_and_gradients():
    module = SwiGLU(8, 12)
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_model_forward_loss_backward_and_weight_tying():
    model = NovaLM(tiny_config())
    ids = torch.randint(0, 64, (2, 12))
    logits = model(ids)
    assert logits.shape == (2, 12, 64)
    assert model.lm_head.weight is model.token_embedding.weight
    loss = causal_lm_loss(logits, ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None


def test_future_tokens_do_not_change_earlier_logits():
    torch.manual_seed(0)
    model = NovaLM(tiny_config()).eval()
    ids = torch.randint(0, 64, (1, 10))
    changed = ids.clone()
    changed[:, 7:] = torch.randint(0, 64, (1, 3))
    with torch.no_grad():
        a, b = model(ids), model(changed)
    torch.testing.assert_close(a[:, :7], b[:, :7])


def test_sdpa_matches_reference():
    torch.manual_seed(3)
    ref = NovaLM(tiny_config(attention_backend="naive"))
    opt = NovaLM(tiny_config(attention_backend="sdpa"))
    opt.load_state_dict(ref.state_dict())
    ids = torch.randint(0, 64, (2, 11))
    torch.testing.assert_close(ref(ids), opt(ids), atol=1e-5, rtol=1e-5)
    auto_kernel = NovaLM(tiny_config(attention_backend="flash"))
    auto_kernel.load_state_dict(ref.state_dict())
    torch.testing.assert_close(ref(ids), auto_kernel(ids), atol=1e-5, rtol=1e-5)


def test_invalid_config_rejected():
    try:
        tiny_config(hidden_size=30)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid head dimensions should fail")
