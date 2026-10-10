"""Focused regression checks for configuration and data component contracts."""
import tempfile
import unittest
from pathlib import Path

import torch

from novalm.config.schema import InferenceConfig, MoEConfig, TokenizerConfig, TrainingConfig
from novalm.data.collator import causal_lm_collate, shift_labels
from novalm.data.ingest import iter_text_documents
from novalm.data.tokenize import tokenizer_digest
from novalm.inference.quantization import QuantizedLinear4bit, quantize_int4
from novalm.model.attention import GroupedQueryAttention, MultiHeadAttention
from novalm.model.attention.causal import causal_mask, naive_causal_attention
from novalm.model.attention.kv_cache import cache_bytes, cache_length
from novalm.model.attention.sdpa import sdpa_causal_attention
from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM
from novalm.post_training.grpo import group_relative_policy_loss
from novalm.serving.batching import batch_generate
from novalm.tokenizer.train import train_tokenizer
from novalm.training.checkpoint import atomic_torch_save, load_torch_checkpoint
from novalm.training.sampler import DistributedResumableSampler
from novalm.training.scheduler import warmup_cosine_lr


class ComponentContracts(unittest.TestCase):
    def test_validated_configs_reject_invalid_values(self):
        self.assertEqual(TrainingConfig().precision, "fp32")
        self.assertEqual(MoEConfig().top_k, 2)
        with self.assertRaises(ValueError):
            InferenceConfig(top_p=0)

    def test_collator_and_causal_shift_mask_padding(self):
        inputs, labels = causal_lm_collate([[11, 12, 13], [21, 22]], pad_id=0)
        shifted_inputs, targets = shift_labels(inputs, labels)
        self.assertEqual(shifted_inputs.tolist(), [[11, 12], [21, 22]])
        self.assertEqual(targets.tolist(), [[12, 13], [22, -100]])
        self.assertEqual(inputs.dtype, torch.long)

    def test_ingestion_normalizes_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("  A  text\r\n\r\n  line  \n", encoding="utf-8")
            (root / "ignored.md").write_text("ignored", encoding="utf-8")
            rows = list(iter_text_documents([root], source_name="fixture",
                                            license_name="CC0", language="en"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "A text\n\nline")
        self.assertEqual(rows[0]["source"], "fixture")
        self.assertEqual(rows[0]["license"], "CC0")

    def test_tokenizer_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny.model"
            path.write_bytes(b"tokenizer fixture")
            self.assertEqual(tokenizer_digest(path), tokenizer_digest(path))

    def test_project_tokenizer_training_from_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                '{"text":"Hello world, Persian سلام دنیا, code x = 42"}\n' * 80,
                encoding="utf-8")
            tokenizer = train_tokenizer([corpus], root / "tokenizer",
                                        TokenizerConfig(vocab_size=64,
                                                        character_coverage=1.0))
            sample = "Hello world سلام دنیا"
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_attention_backends_and_mha_shapes(self):
        torch.manual_seed(11)
        query = torch.randn(2, 3, 5, 8)
        key = torch.randn(2, 3, 5, 8)
        value = torch.randn(2, 3, 5, 8)
        allowed = causal_mask(5, 5)
        reference = naive_causal_attention(query, key, value, allowed)
        optimized = sdpa_causal_attention(query, key, value, allowed)
        torch.testing.assert_close(reference, optimized, atol=1e-6, rtol=1e-6)
        gqa = GroupedQueryAttention(24, 3, 3, 8, 10000.0, 1e-6)
        mha = MultiHeadAttention(24, 3, 8)
        mha.load_state_dict(gqa.state_dict())
        x = torch.randn(2, 5, 24)
        self.assertEqual(mha.num_kv_heads, mha.num_heads)
        self.assertEqual(mha(x).shape, (2, 5, 24))
        torch.testing.assert_close(gqa(x), mha(x), atol=1e-6, rtol=1e-6)

    def test_cache_accounting_and_atomic_checkpoint(self):
        cache = ((torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),)
        self.assertEqual(cache_length(cache), 3)
        self.assertEqual(cache_bytes(cache), 1 * 2 * 3 * 4 * 4 * 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.pt"
            atomic_torch_save({"model": {"weight": torch.ones(2)}}, path)
            state = load_torch_checkpoint(path)
            torch.testing.assert_close(state["model"]["weight"], torch.ones(2))

    def test_int4_linear_packing_and_model_forward(self):
        linear = torch.nn.Linear(5, 3)
        quantized = QuantizedLinear4bit(linear)
        self.assertEqual(quantized.packed_weight.numel(), 3 * 3)
        packed_bytes = (quantized.packed_weight.numel() * quantized.packed_weight.element_size()
                        + quantized.scale.numel() * quantized.scale.element_size()
                        + quantized.bias.numel() * quantized.bias.element_size())
        self.assertLess(packed_bytes, linear.weight.numel() * 4 + linear.bias.numel() * 4)
        x = torch.randn(2, 5)
        output = quantized(x)
        self.assertEqual(output.shape, (2, 3))
        config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                             num_attention_heads=2, num_kv_heads=1,
                             intermediate_size=24, max_seq_length=8)
        model = quantize_int4(NovaLM(config)).eval()
        with torch.inference_mode():
            logits = model(torch.randint(0, 32, (1, 5)))
        self.assertEqual(logits.shape, (1, 5, 32))
        self.assertTrue(torch.isfinite(logits).all())

    def test_static_batch_generation_matches_separate_greedy(self):
        torch.manual_seed(3)
        config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                             num_attention_heads=2, num_kv_heads=1,
                             intermediate_size=24, max_seq_length=8)
        model = NovaLM(config).eval()
        prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
        batch_result = batch_generate(model, prompts, 2, temperature=0)
        separate = torch.cat([
            batch_generate(model, prompts[index:index + 1], 2, temperature=0)
            for index in range(2)
        ])
        self.assertTrue(torch.equal(batch_result, separate))

    def test_cosine_scheduler_endpoints(self):
        self.assertAlmostEqual(warmup_cosine_lr(0, 1.0, 2, 10), 0.5)
        self.assertAlmostEqual(warmup_cosine_lr(10, 1.0, 2, 10), 0.1)

    def test_grpo_policy_loss_is_finite_and_differentiable(self):
        logps = torch.tensor([-0.2, -0.8], requires_grad=True)
        old_logps = torch.tensor([-0.3, -0.7])
        advantages = torch.tensor([1.0, -1.0])
        loss = group_relative_policy_loss(logps, old_logps, advantages)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logps.grad).all())

    def test_distributed_resumable_sampler_shards_and_restores(self):
        data = list(range(11))
        samplers = [DistributedResumableSampler(data, 2, rank, seed=17) for rank in (0, 1)]
        shards = [list(iter(sampler)) for sampler in samplers]
        self.assertTrue(set(shards[0]).isdisjoint(shards[1]))
        self.assertEqual(sorted(shards[0] + shards[1]), data)

        sampler = DistributedResumableSampler(data, 2, 0, seed=17)
        iterator = iter(sampler)
        consumed = [next(iterator), next(iterator)]
        state = sampler.state_dict()
        restored = DistributedResumableSampler(data, 2, 0, seed=999)
        restored.load_state_dict(state)
        self.assertEqual(consumed + list(iter(restored)), shards[0])


if __name__ == "__main__":
    unittest.main()
