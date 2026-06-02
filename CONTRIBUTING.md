# Contributing

Contributions are welcome. This project bridges two fast-moving upstreams (Genesis and LeRobot), so
the guiding principle is a **small, well-tested surface that doesn't assume either library's APIs are
stable** — the GPU-bound code is kept behind a `Protocol` and verified against the live libraries.

## Development setup

```bash
git clone https://github.com/arun-prasath2005/lerobot-genesis
cd lerobot-genesis
pip install -e ".[dev]"
```

The default test suite runs on a CPU with neither `genesis-world` nor `lerobot` installed — the
simulator is faked behind the `SceneDriver` seam. Integration tests that need a real GPU are marked
`@pytest.mark.gpu` and are skipped unless those extras are installed.

## Checks (run before opening a PR)

```bash
ruff check .
ruff format --check .
mypy
pytest
```

CI runs the same four on Python 3.10–3.12. Please add a test for any behaviour change and a
`CHANGELOG.md` entry under *Unreleased*.

## Scope

Keep the package generic and dependency-light. Anything that ties it to a specific downstream
application, scene catalog, or asset pipeline belongs in *that* application, not here.
