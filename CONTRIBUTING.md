# Contributing to OpenEvals

Thank you for your interest in contributing to OpenEvals!

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/openevals.git
cd openevals
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pre-commit && pre-commit install
```

## Code Style

- **Python**: PEP 8, Black formatting, 88 char line length
- **TypeScript**: ESLint + Prettier

```bash
black . && isort .
```

## Testing

```bash
pytest tests/
pytest --cov=openevals tests/
```

## Pull Request Process

1. Fork and create branch from `main`
2. Make changes with tests
3. Ensure all tests pass
4. Submit PR with clear description

## Commit Messages

```
<type>: <description>
```

Types: `Add`, `Fix`, `Docs`, `Refactor`, `Test`, `Chore`

## Adding New Features

- **New benchmarks**: Add to `openevals/tasks/`, register in `__init__.py`
- **New models**: Add loader to `openevals/core/model_loader.py`

See [full contributing guide](https://heilcheng.github.io/openevals/contributing/) for details.

## License

MIT License - contributions licensed under the same.
