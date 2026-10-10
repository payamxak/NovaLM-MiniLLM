# Contributing

Keep reference math readable, add focused tests for behavior and gradients, and state explicitly when an experiment has not been run. Do not commit raw corpora, tokenizer models trained from non-redistributable data, checkpoints, logs, or generated data. Run `python -m unittest discover -s tests -v` and `python -m compileall -q novalm scripts tests` before a change. When available, also run `python -m pytest` and `ruff check novalm tests scripts`.
