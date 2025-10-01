# Publishing to PyPI

This guide explains how to publish the `cosine-plateau-scheduler` package to PyPI.

## Prerequisites

1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. Install uv if you haven't:
   ```bash
   pip install uv
   ```

## Building the Package

1. Make sure all tests pass:
   ```bash
   uv pip install -e ".[dev]"
   pytest tests/
   ```

2. Build the distribution:
   ```bash
   uv build
   ```

   This creates:
   - `dist/cosine_plateau_scheduler-X.Y.Z.tar.gz` (source distribution)
   - `dist/cosine_plateau_scheduler-X.Y.Z-py3-none-any.whl` (wheel)

## Testing on TestPyPI

Before publishing to the real PyPI, test on TestPyPI:

```bash
# Install twine for uploading
uv pip install twine

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

Test the installation:
```bash
pip install --index-url https://test.pypi.org/simple/ cosine-plateau-scheduler
```

## Publishing to PyPI

Once everything works on TestPyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for your PyPI username and password.

## Using API Tokens (Recommended)

For better security, use API tokens:

1. Go to PyPI Account Settings → API tokens
2. Create a token with appropriate scope
3. Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
```

Now you can upload without entering credentials:
```bash
python -m twine upload dist/*
```

## Version Management

Update version in `pyproject.toml` before each release:

```toml
[project]
name = "cosine-plateau-scheduler"
version = "0.1.0"  # Update this
```

Follow [Semantic Versioning](https://semver.org/):
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes

## Release Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] CHANGELOG updated (if you create one)
- [ ] Version bumped in `pyproject.toml`
- [ ] Built distribution (`uv build`)
- [ ] Tested on TestPyPI
- [ ] Published to PyPI
- [ ] Tagged release in git:
  ```bash
  git tag -a v0.1.0 -m "Release version 0.1.0"
  git push origin v0.1.0
  ```

## Continuous Integration (Optional)

Consider setting up GitHub Actions to automate:
- Running tests on pull requests
- Publishing to PyPI on tagged releases

Example workflow (`.github/workflows/publish.yml`):

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install uv
        run: pip install uv
      - name: Build
        run: uv build
      - name: Publish
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          pip install twine
          twine upload dist/*
```

## Troubleshooting

**"File already exists"**: You cannot overwrite existing versions on PyPI. Bump the version number.

**"Invalid package name"**: Ensure the name in `pyproject.toml` matches PyPI naming conventions (lowercase, hyphens allowed).

**Import errors after install**: Make sure `src/cosine_plateau_scheduler/__init__.py` properly exports the main classes.

