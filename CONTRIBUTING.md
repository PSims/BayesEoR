# Contributing

Contributions to BayesEoR are welcome in the form of issues and pull requests.

## Development Setup

Create the project environment from the repository root:

```bash
mamba env create -f environment.yaml
```

If the `bayeseor` environment already exists, update it with:

```bash
mamba env update -n bayeseor -f environment.yaml --prune
```

If you prefer `conda`, replace `mamba` with `conda` in the commands above.

Activate the environment:

```bash
conda activate bayeseor
```

Install BayesEoR in editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
```

Install the local git hooks with:

```bash
pre-commit install
```

## Testing

Run the test suite with:

```bash
python -m pytest
```

You can run lint checks with:

```bash
ruff check bayeseor/model tests
```

You can also run the configured pre-commit hooks across the repository with:

```bash
pre-commit run --all-files
```

For CI or other non-interactive shells, `conda run -n bayeseor ...` is still a
good alternative to activating the environment first.

## Questions

For questions about contributing, please open an issue or contact one of the
project managers listed in the README.
