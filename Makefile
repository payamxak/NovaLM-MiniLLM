.PHONY: test lint
test:
	python -m unittest discover -s tests -v
pytest:
	python -m pytest
lint:
	python -m ruff check novalm tests
