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

**Full test suite (runs all tests and generates QA_REPORT.md):**

```bash
make test
# or
python scripts/run_tests_with_qa_report.py
```

**Quick pytest run (no QA report):**

```bash
python -m pytest -vv
```

### Style

- Prefer small, focused changes.
- Add tests for new functionality.
- Keep API changes backwards compatible when possible.
