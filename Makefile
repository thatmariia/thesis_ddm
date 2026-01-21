.PHONY: docs docs-open

docs:
	uv run --group docs sphinx-build -M html docs/source docs/build

docs-open:
	uv run python -m http.server 8000 --bind 127.0.0.1 --directory docs/build/html