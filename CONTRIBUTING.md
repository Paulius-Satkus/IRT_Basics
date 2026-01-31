## Contributing

Thanks for your interest in contributing. This project accepts issues and pull
requests.

### Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Tests

```bash
python -m pytest -vv
```

### Style

- Prefer small, focused changes.
- Add tests for new functionality.
- Keep API changes backwards compatible when possible.
