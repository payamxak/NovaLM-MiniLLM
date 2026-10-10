import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NovaLM local benchmark script")
    parser.add_argument(
        "--suite", choices=("attention", "kv-cache", "inference", "architecture",
                            "training", "memory", "moe", "quantization"), required=True)
    parsed, forwarded = parser.parse_known_args()
    scripts = {
        "attention": "attention.py",
        "kv-cache": "kv_cache.py",
        "inference": "inference.py",
        "architecture": "architecture.py",
        "training": "training.py",
        "memory": "memory.py",
        "moe": "moe.py",
        "quantization": "quantization.py",
    }
    script = Path(__file__).resolve().parents[1] / "benchmarks" / scripts[parsed.suite]
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
