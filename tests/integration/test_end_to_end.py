"""Minimal CPU integration: forward, optimization, save/load, and cached decode."""
import tempfile
import unittest
from pathlib import Path

import torch

from novalm.inference.generate import generate
from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM
from novalm.training.trainer import Trainer, TrainerConfig


class EndToEndSmoke(unittest.TestCase):
    def test_train_checkpoint_reload_and_cached_generation(self):
        torch.manual_seed(29)
        config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                             num_attention_heads=2, num_kv_heads=1,
                             intermediate_size=24, max_seq_length=8)
        model = NovaLM(config)
        trainer = Trainer(model, TrainerConfig(learning_rate=1e-3, warmup_steps=1,
                                              total_steps=2), "cpu")
        batch = torch.randint(0, config.vocab_size, (2, 6))
        self.assertTrue(torch.isfinite(torch.tensor(trainer.train_step(batch))))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            trainer.save_checkpoint(path)
            restored = Trainer(NovaLM(config), TrainerConfig(learning_rate=1e-3,
                                warmup_steps=1, total_steps=2), "cpu")
            restored.load_checkpoint(path)
            self.assertEqual(restored.step, 1)
            ids = torch.tensor([[1, 2, 3]])
            full = generate(restored.model, ids, max_new_tokens=2, temperature=0)
            self.assertEqual(tuple(full.shape), (1, 5))
            with torch.inference_mode():
                _, cache = restored.model(ids, use_cache=True)
                cached_logits, _ = restored.model(full[:, 3:4], cache, use_cache=True)
                full_logits = restored.model(full[:, :4])
            torch.testing.assert_close(cached_logits[:, 0], full_logits[:, 3],
                                       atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
