# Development Guide

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- [ruff](https://github.com/astral-sh/ruff) - Fast Python linter and formatter

## Install Dev Dependencies

```bash
make dev
```

Or manually:

```bash
uv sync --extra dev
```

## Run Tests

```bash
make test
```

With coverage report:

```bash
make test-cov
```

## Lint

Check code style:

```bash
make lint
```

## Format

Auto-format code:

```bash
make format
```

## Build Package

```bash
make dist
```

Output will be in `dist/`.

## Publish to PyPI

```bash
make release
```

## Project Structure

```
hanzi_char_featurizer/
├── hanzi_char_featurizer/      # Main source package
│   ├── __init__.py             # Featurizer class and protocol
│   └── featurizers/            # Feature extractors
│       ├── four_corner.py      # Four-corner encoding
│       ├── pinyin.py           # Pinyin extraction
│       ├── pinyin_parts.py     # Pinyin parts (initial, final, tone)
│       └── chaizi.py           # Character decomposition
├── tests/                      # Test suite
├── usage/                      # Usage examples
├── pyproject.toml              # Project configuration
└── Makefile                    # Development commands
```

## API Overview

The library provides a unified API:

- `extract(char_seq, as_numpy=False)` - Extract features, returns dict or NumPy arrays
- `vocabulary` - Property that returns the vocabulary dict

Example:

```python
from hanzi_char_featurizer import Featurizer

f = Featurizer()

# Extract as dict
result = f.extract('明天')
# {'pinyin': {...}, 'four_corner': {...}}

# Extract as NumPy arrays
result = f.extract('明天', as_numpy=True)

# Get vocabulary
vocab = f.vocabulary
```

## Running Individual Featurizers

```python
from hanzi_char_featurizer.featurizers.four_corner import FourCorner
from hanzi_char_featurizer.featurizers.pinyin_parts import PinYinParts

fc = FourCorner()
print(fc.extract('明'))
# {'upper_left': ['6'], 'upper_right': ['7'], ...}

pp = PinYinParts()
print(pp.extract('明'))
# {'initial': [['m']], 'final': [['ing']], 'tone': [['2']]}
```
