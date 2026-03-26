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

Install BayesEoR in editable mode with development tools:

```bash
conda run -n bayeseor python -m pip install -e ".[dev]"
```

## Testing

Run the test suite with:

```bash
conda run --no-capture-output -n bayeseor python -m pytest
```

You can run lint checks with:

```bash
conda run --no-capture-output -n bayeseor ruff check bayeseor/model tests
```

## Questions

For questions about contributing, please open an issue or contact one of the
project managers listed in the README.
