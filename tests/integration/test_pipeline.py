import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from novalm.data.cleaning import clean_document
from novalm.data.dedup import exact_deduplicate
from novalm.data.language_filter import detect_language
from novalm.data.minhash import near_deduplicate
from novalm.data.mixture import sample_mixture, weighted_stream
from novalm.data.packing import pack_sequences
from novalm.data.quality import QualityConfig, quality_rejection
from novalm.data.shards import TokenShardDataset, verify_shard_manifest, write_token_shards
from novalm.evaluation.perplexity import evaluate_loss
from novalm.inference.generate import generate
from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM
from novalm.post_training.formatting import assistant_only_example
from novalm.serving.app import create_app
from novalm.tokenizer.chat_template import format_chat
from novalm.tokenizer.tokenizer import NovaTokenizer
from novalm.training.trainer import Trainer, TrainerConfig


def test_clean_quality_deduplicate_pack_and_shard():
    assert clean_document("Ａ\r\n  test  ") == "A\ntest"
    assert quality_rejection("meaningful English text " * 5, QualityConfig(min_chars=1)) is None
    docs = list(exact_deduplicate([{"text": "one  two"}, {"text": "one two"}]))
    assert len(docs) == 1
    blocks = list(pack_sequences([[4, 5], [6, 7]], 3, 2))
    assert blocks == [[4, 5, 2], [6, 7, 2]]
    with tempfile.TemporaryDirectory() as tmp:
        manifest = write_token_shards([[1, 2, 3, 4, 5]], tmp, shard_tokens=5)
        dataset = TokenShardDataset(Path(tmp) / manifest["shards"][0]["path"], 3)
        assert dataset[0].tolist() == [1, 2, 3, 4]
        assert len(dataset) == 1
        dataset.close()
        verify_shard_manifest(Path(tmp) / "manifest.json")
        with (Path(tmp) / manifest["shards"][0]["path"]).open("ab") as stream:
            stream.write(b"x")
        try:
            verify_shard_manifest(Path(tmp) / "manifest.json")
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt shards should fail verification")
        windows_dir = Path(tmp) / "windows"
        windows_manifest = write_token_shards([list(range(10))], windows_dir, shard_tokens=10)
        windows = TokenShardDataset(windows_dir / windows_manifest["shards"][0]["path"], 3)
        assert len(windows) == 3
        assert [windows[i].tolist() for i in range(len(windows))] == [
            [0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
        windows.close()


def test_sentencepiece_training_roundtrip_and_chat_template():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text(("سلام دنیا hello world python 123\n" * 100), encoding="utf-8")
        tokenizer = NovaTokenizer.train([corpus], Path(tmp) / "novalm", vocab_size=64,
                                        character_coverage=1.0)
        sample = "سلام hello world"
        assert tokenizer.decode(tokenizer.encode(sample)) == sample
        assert tokenizer.token_id("<|assistant|>") >= 0
        ids, labels = assistant_only_example([
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}], tokenizer)
        assert len(ids) == len(labels)
        assert all(x == -100 for x in labels[:labels.index(next(y for y in labels if y != -100))])
        assert any(x != -100 for x in labels)
    prompt = format_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert prompt.endswith("<|assistant|>\n")


def test_training_checkpoint_and_generation():
    config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                         num_attention_heads=2, num_kv_heads=1,
                         intermediate_size=24, max_seq_length=8)
    model = NovaLM(config)
    trainer = Trainer(model, TrainerConfig(total_steps=4, warmup_steps=1), "cpu")
    batch = torch.randint(0, 32, (2, 8))
    before = trainer.train_step(batch)
    assert before > 0 and trainer.step == 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        trainer.save_checkpoint(path)
        clone = Trainer(NovaLM(config), TrainerConfig(total_steps=4, warmup_steps=1), "cpu")
        clone.load_checkpoint(path)
        assert clone.step == 1
        torch.testing.assert_close(model(batch[:, :-1]), clone.model(batch[:, :-1]))
    result = generate(model, batch[:1, :3], max_new_tokens=2, temperature=0)
    assert result.shape == (1, 5)


def test_kv_cache_matches_full_prefix_logits():
    config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=2,
                         num_attention_heads=2, num_kv_heads=1,
                         intermediate_size=24, max_seq_length=8)
    model = NovaLM(config).eval()
    ids = torch.randint(0, 32, (1, 6))
    with torch.inference_mode():
        full = model(ids)
        _first, cache = model(ids[:, :4], use_cache=True)
        last, cache = model(ids[:, 4:5], cache, use_cache=True)
        final, cache = model(ids[:, 5:], cache, use_cache=True)
    torch.testing.assert_close(full[:, 4], last[:, 0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(full[:, 5], final[:, 0], atol=2e-5, rtol=2e-5)
    assert cache[0][0].shape[1] == config.num_kv_heads


def test_language_near_dedup_and_mixture():
    assert detect_language("سلام دنیا") == "fa"
    assert detect_language("hello world") == "en"
    rows = [{"text": "alpha bravo charlie delta echo foxtrot golf hotel"},
            {"text": "alpha bravo charlie delta echo foxtrot golf hotel"},
            {"text": "completely unrelated material for checking duplicate removal"}]
    assert len(list(near_deduplicate(rows, threshold=0.8, num_perm=32, bands=8))) == 2
    sources = {"en": [{"text": "hello"}], "fa": [{"text": "سلام"}]}
    a = list(sample_mixture(sources, {"en": 0.5, "fa": 0.5}, 10, seed=4))
    b = list(sample_mixture(sources, {"en": 0.5, "fa": 0.5}, 10, seed=4))
    assert a == b
    stream_sources = {"en": iter(sources["en"]), "fa": iter(sources["fa"])}
    first = list(weighted_stream(stream_sources, {"en": 0.5, "fa": 0.5}, seed=4))
    second = list(weighted_stream({"en": iter(sources["en"]), "fa": iter(sources["fa"])},
                                  {"en": 0.5, "fa": 0.5}, seed=4))
    assert first == second and len(first) == 2
    try:
        near_deduplicate(rows, bands=0)
        next(iter(near_deduplicate(rows, bands=0)))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid MinHash band count must fail clearly")
    try:
        QualityConfig(max_chars=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid quality bounds must fail clearly")


def test_perplexity_evaluation_returns_loss_and_token_count():
    config = ModelConfig(vocab_size=24, hidden_size=16, num_layers=1,
                         num_attention_heads=2, num_kv_heads=1,
                         intermediate_size=24, max_seq_length=8)
    model = NovaLM(config)
    metrics = evaluate_loss(model, [torch.randint(0, 24, (2, 7))])
    assert metrics["tokens"] == 12
    assert metrics["loss"] > 0 and metrics["perplexity"] > 1


def test_fastapi_health_info_completion_and_chat():
    from fastapi.testclient import TestClient
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text(("hello world سلام python\n" * 100), encoding="utf-8")
        tokenizer = NovaTokenizer.train([corpus], Path(tmp) / "tok", 64, 1.0)
        config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, num_layers=1,
                             num_attention_heads=2, num_kv_heads=1,
                             intermediate_size=24, max_seq_length=64)
        model = NovaLM(config).eval()
        client = TestClient(create_app(model, tokenizer))
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/model/info").json()["parameters"] == model.parameter_count
        response = client.post("/v1/completions", json={"prompt": "hello", "max_new_tokens": 1,
                                                         "temperature": 0, "min_p": 0.05,
                                                         "repetition_penalty": 1.1,
                                                         "stop": [" world"]})
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["completion_tokens"] == 1
        assert response.json()["choices"][0]["finish_reason"] in {"stop", "length"}
        invalid = client.post("/v1/completions", json={"prompt": "hello", "top_p": 0})
        assert invalid.status_code == 422
        streamed = client.post("/v1/completions", json={"prompt": "hello", "max_new_tokens": 1,
                                                         "temperature": 0, "stream": True})
        assert streamed.status_code == 200, streamed.text
        assert streamed.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]" in streamed.text
        chat = client.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "hello"}], "max_new_tokens": 1, "temperature": 0})
        assert chat.status_code == 200, chat.text


def test_training_cli_validation_and_activation_checkpointing():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        train_dir, valid_dir = work / "train", work / "valid"
        tokens = [i % 32 for i in range(96)]
        write_token_shards([tokens], train_dir, shard_tokens=96)
        write_token_shards([tokens[::-1]], valid_dir, shard_tokens=96)
        model_config = work / "model.yaml"
        model_config.write_text(
            "model:\n  vocab_size: 32\n  hidden_size: 16\n  num_layers: 1\n"
            "  num_attention_heads: 2\n  num_kv_heads: 1\n  intermediate_size: 24\n"
            "  max_seq_length: 8\n", encoding="utf-8")
        training_config = work / "training.yaml"
        training_config.write_text(
            "learning_rate: 0.001\nweight_decay: 0.0\ntotal_steps: 2\n"
            "warmup_steps: 1\ngradient_accumulation_steps: 1\n"
            "activation_checkpointing: true\n", encoding="utf-8")
        def invoke(checkpoint, steps, resume=False):
            command = [
                sys.executable, str(root / "scripts" / "train.py"),
                "--config", str(model_config), "--training-config", str(training_config),
                "--shards", str(train_dir), "--validation-shards", str(valid_dir),
                "--eval-every", "1", "--eval-batches", "1", "--steps", str(steps),
                "--batch-size", "1", "--checkpoint", str(checkpoint), "--device", "cpu",
                "--best-checkpoint", str(checkpoint.parent / "best.pt"),
            ]
            if resume:
                command.append("--resume")
            return subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)

        full_checkpoint = work / "full" / "latest.pt"
        assert "finished 3 optimizer steps" in invoke(full_checkpoint, 3).stdout
        resumed_checkpoint = work / "resumed" / "latest.pt"
        invoke(resumed_checkpoint, 2)
        assert "finished 3 optimizer steps" in invoke(resumed_checkpoint, 1, resume=True).stdout
        rows = [json.loads(line) for line in (resumed_checkpoint.parent / "metrics.jsonl").read_text().splitlines()]
        assert len([row for row in rows if "validation_loss" in row]) == 3
        full_state = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
        resumed_state = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
        best_state = torch.load(full_checkpoint.parent / "best.pt", map_location="cpu",
                                weights_only=False)
        assert best_state["best_validation_step"] in {1, 2, 3}
        assert best_state["metrics"]["validation_tokens"] > 0
        for name, tensor in full_state["model"].items():
            torch.testing.assert_close(tensor, resumed_state["model"][name], atol=0, rtol=0)


def test_prepare_data_cli_weighted_sources_and_provenance():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        english_dir = work / "english"
        persian_dir = work / "persian"
        english_dir.mkdir()
        persian_dir.mkdir()
        shared = "A shared document that appears in both licensed training sources."
        (english_dir / "unique.txt").write_text(
            "An English-only document with enough meaningful words.", encoding="utf-8")
        (english_dir / "shared.txt").write_text(shared, encoding="utf-8")
        (persian_dir / "unique.txt").write_text(
            "یک متن فارسی برای آزمایش خط پردازش داده و ثبت مجوز.", encoding="utf-8")
        (persian_dir / "shared.txt").write_text(shared, encoding="utf-8")
        config = work / "prepare.yaml"
        config.write_text(yaml.safe_dump({
            "output_file": str(work / "prepared.jsonl"),
            "quality": {"min_chars": 10},
            "sources": {
                "english": {"input_files": [str(english_dir)], "license": "CC-BY-4.0"},
                "persian": {"input_files": [str(persian_dir)], "license": "CC-BY-4.0"},
            },
            "mixture_weights": {"english": 0.4, "persian": 0.6},
            "mixture_seed": 14,
        }, allow_unicode=True), encoding="utf-8")
        subprocess.run([sys.executable, str(root / "scripts" / "prepare_data.py"),
                        "--config", str(config)], cwd=root, capture_output=True,
                       text=True, check=True)
        rows = [json.loads(line) for line in (work / "prepared.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3
        assert len({row["content_hash"] for row in rows}) == 3
        assert {row["mixture_source"] for row in rows} == {"english", "persian"}
        assert all(row["license"] == "CC-BY-4.0" for row in rows)
