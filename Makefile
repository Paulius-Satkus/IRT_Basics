.PHONY: test pytest qa refs

# Full test suite: runs all tests and generates QA_REPORT.md at the end.
# Use this when you want a complete QA run.
test: qa

qa:
	python scripts/run_tests_with_qa_report.py

# Generate polytomous R reference files (requires R + mirt)
refs:
	Rscript tests/generate_polytomous_reference.R

# Quick pytest run (no QA report)
pytest:
	python -m pytest -vv --tb=short
