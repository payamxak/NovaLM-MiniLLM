import json
import tempfile
from pathlib import Path

import torch

from novalm.inference.quantization import dynamic_int8
from novalm.inference.sampler import sample_next_token
from novalm.model.attention_mla import LatentAttention
from novalm.model.config import ModelConfig
from novalm.model.moe.layer import MoELayer
from novalm.model.mtp.head import MultiTokenPredictionHead
from novalm.model.mtp.loss import multi_token_loss
from novalm.model.transformer import NovaLM
from novalm.post_training.dpo import completion_ids, dpo_loss, sequence_log_probability
from novalm.training.trainer import Trainer, TrainerConfig


def test_mtp_horizon_loss_and_gradients():
    hidden = torch.randn(2, 8, 12, requires_grad=True)
    ids = torch.randint(0, 20, (2, 8))
    heads = MultiTokenPredictionHead(12, 20, 2)
    loss, metrics = multi_token_loss(heads(hidden), ids, [1.0, 0.5])
    loss.backward()
    assert set(metrics) == {2, 3}
    assert torch.isfinite(loss) and hidden.grad is not None


def test_moe_shapes_routing_and_gradients():
    layer = MoELayer(12, 24, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 12, requires_grad=True)
    y, aux, metrics = layer(x)
    assert y.shape == x.shape
    assert metrics["expert_token_counts"].sum() == 20
    (y.mean() + aux).backward()
    assert x.grad is not None


def test_dpo_loss_is_finite_and_preference_accurate_when_ordered():
    chosen = torch.tensor([-1.0, -2.0])
    rejected = torch.tensor([-3.0, -4.0])
    loss, metrics = dpo_loss(chosen, rejected, torch.zeros(2), torch.zeros(2))
    assert torch.isfinite(loss)
    assert metrics["accuracy"] == 1


def test_dpo_completion_truncation_preserves_supervised_targets():
    class FakeProcessor:
        @staticmethod
        def bos_id():
            return 1

    class FakeTokenizer:
        processor = FakeProcessor()

        @staticmethod
        def encode(text, add_eos=False):
            ids = [4, 5, 6, 7] if text == "long prompt" else [8, 9, 10]
            return ids + ([2] if add_eos else [])

    ids, labels = completion_ids(FakeTokenizer(), "long prompt", "long answer", 5)
    assert ids.shape == labels.shape == (1, 5)
    assert labels[0, 0].item() == -100
    assert labels[:, 1:].ne(-100).all()
    logits = torch.randn(1, 5, 16, requires_grad=True)
    logps = sequence_log_probability(logits, labels)
    logps.sum().backward()
    assert torch.isfinite(logps).all() and logits.grad is not None
    try:
        completion_ids(FakeTokenizer(), "long prompt", "long answer", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("DPO must reject contexts too short for prompt and completion")


def test_latent_attention_cache_equivalence():
    torch.manual_seed(4)
    module = LatentAttention(16, num_heads=2, latent_dim=6, max_seq_length=8).eval()
    x = torch.randn(1, 6, 16)
    with torch.inference_mode():
        full = module(x)
        _, cache = module(x[:, :4], use_cache=True)
        cached, _ = module(x[:, 4:5], cache, use_cache=True)
    torch.testing.assert_close(full[:, 4], cached[:, 0], atol=2e-5, rtol=2e-5)


def test_dynamic_int8_model_forward():
    config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                         num_attention_heads=2, num_kv_heads=1,
                         intermediate_size=24, max_seq_length=8)
    model = dynamic_int8(NovaLM(config).eval())
    logits = model(torch.randint(0, 32, (1, 5)))
    assert logits.shape == (1, 5, 32) and torch.isfinite(logits).all()


def test_batched_repetition_penalty_is_row_local():
    scores = torch.tensor([[3.0, 2.0, 0.0], [3.0, 2.0, 0.0]])
    tokens = torch.tensor([[0, 0], [1, 1]])
    sampled = sample_next_token(scores, temperature=0, repetition_tokens=tokens,
                                repetition_penalty=2.0)
    assert sampled.flatten().tolist() == [1, 0]


def test_mtp_moe_and_mla_are_integrated_and_trainable():
    base = {"vocab_size": 32, "hidden_size": 16, "num_layers": 1,
            "num_attention_heads": 2, "num_kv_heads": 1,
            "intermediate_size": 24, "max_seq_length": 8}
    variants = [
        ModelConfig(**base, mtp_enabled=True, mtp_horizons=2),
        ModelConfig(**base, moe_enabled=True, num_experts=4, moe_top_k=2),
        ModelConfig(**base, attention_type="mla", mla_latent_dim=6),
    ]
    batch = torch.randint(0, 32, (2, 8))
    for config in variants:
        model = NovaLM(config)
        assert model.parameter_count == config.parameter_count()
        trainer = Trainer(model, TrainerConfig(learning_rate=1e-3, warmup_steps=1,
                                               total_steps=2), "cpu")
        assert torch.isfinite(torch.tensor(trainer.train_step(batch)))
        if config.mtp_enabled:
            assert "mtp_loss_h2" in trainer.last_metrics
            assert "mtp_loss_h3" in trainer.last_metrics
        if config.moe_enabled:
            assert len(trainer.last_metrics["expert_token_counts"]) == config.num_experts
            assert trainer.last_metrics["router_entropy"] >= 0
        if config.attention_type == "mla":
            model.eval()
            ids = batch[:1, :6]
            with torch.inference_mode():
                full = model(ids)
                _, cache = model(ids[:, :4], use_cache=True)
                cached, _ = model(ids[:, 4:5], cache, use_cache=True)
            torch.testing.assert_close(full[:, 4], cached[:, 0], atol=2e-5, rtol=2e-5)


def test_architecture_ablation_switches_and_parameter_counts():
    base = {"vocab_size": 32, "hidden_size": 16, "num_layers": 1,
            "num_attention_heads": 2, "num_kv_heads": 1,
            "intermediate_size": 24, "max_seq_length": 8}
    variants = [
        ModelConfig(**base, norm_type="layernorm"),
        ModelConfig(**base, mlp_type="gelu"),
        ModelConfig(**base, position_type="absolute"),
        ModelConfig(**base, norm_type="layernorm", mlp_type="gelu",
                    position_type="absolute"),
        ModelConfig(vocab_size=32, hidden_size=15, num_layers=1,
                    num_attention_heads=3, num_kv_heads=1, intermediate_size=20,
                    max_seq_length=8, position_type="absolute"),
    ]
    ids = torch.randint(0, 32, (1, 6))
    for config in variants:
        model = NovaLM(config).eval()
        assert model.parameter_count == config.parameter_count()
        with torch.inference_mode():
            full = model(ids)
            _, cache = model(ids[:, :4], use_cache=True)
            cached, _ = model(ids[:, 4:5], cache, use_cache=True)
        torch.testing.assert_close(full[:, 4], cached[:, 0], atol=2e-5, rtol=2e-5)


def test_activation_checkpointing_and_validation_logging():
    config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=2,
                         num_attention_heads=2, num_kv_heads=1,
                         intermediate_size=24, max_seq_length=8,
                         moe_enabled=True, num_experts=4, moe_top_k=2)
    torch.manual_seed(17)
    reference_model = NovaLM(config)
    checkpointed_model = NovaLM(config)
    checkpointed_model.load_state_dict(reference_model.state_dict())
    reference_trainer = Trainer(reference_model, TrainerConfig(
        learning_rate=1e-3, warmup_steps=1, total_steps=3), "cpu")
    checkpointed_trainer = Trainer(checkpointed_model, TrainerConfig(
        learning_rate=1e-3, warmup_steps=1, total_steps=3,
        activation_checkpointing=True), "cpu")
    batch = torch.randint(0, 32, (2, 8))
    reference_loss = reference_trainer.train_step(batch)
    checkpointed_loss = checkpointed_trainer.train_step(batch)
    assert abs(reference_loss - checkpointed_loss) < 1e-6
    for reference, checkpointed in zip(reference_model.parameters(),
                                       checkpointed_model.parameters()):
        torch.testing.assert_close(reference, checkpointed, atol=1e-6, rtol=1e-6)

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "metrics.jsonl"
        checkpointed_trainer.fit([batch], max_steps=2, log_path=log_path,
                                 validation_batches=[batch], eval_every_steps=2)
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        validated = [row for row in rows if "validation_loss" in row]
        assert len(validated) == 2
        assert [row["step"] for row in validated] == [2, 3]
        assert all(row["validation_tokens"] > 0 for row in validated)
