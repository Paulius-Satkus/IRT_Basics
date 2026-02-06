.PHONY: test pytest qa

# Full test suite: runs all tests and generates QA_REPORT.md at the end.
# Use this when you want a complete QA run.
test: qa

qa:
	python scripts/run_tests_with_qa_report.py

# Quick pytest run (no QA report)
pytest:
	python -m pytest -vv --tb=short
