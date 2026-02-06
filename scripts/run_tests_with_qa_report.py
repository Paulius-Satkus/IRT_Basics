#!/usr/bin/env python3
"""
Run the full test suite and generate QA_REPORT.md at the end.

Usage:
    python scripts/run_tests_with_qa_report.py
    make test

This is the canonical way to run a "full suite of testing" - it executes
all tests and produces an up-to-date QA report.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_REPORT_PATH = PROJECT_ROOT / "QA_REPORT.md"


def main() -> int:
    """Run pytest, capture output, generate QA report, return pytest exit code."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-vv", "--tb=short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    # Parse summary line: "232 passed, 7 warnings in 19.15s" or "1 failed, 231 passed in 20s"
    passed = failed = warnings = 0
    duration_match = re.search(r"in ([\d.]+)s", output)
    duration_s = duration_match.group(1) if duration_match else "?"

    # Extract numbers from pytest summary (last occurrence before "in Xs")
    for m in re.finditer(r"(\d+)\s+passed", output):
        passed = int(m.group(1))
    for m in re.finditer(r"(\d+)\s+failed", output):
        failed = int(m.group(1))
    for m in re.finditer(r"(\d+)\s+warnings?", output):
        warnings = int(m.group(1))

    status = "PASS" if result.returncode == 0 else "FAIL"
    total = passed + failed

    report = f"""# QA Report

Date: {datetime.now().strftime("%Y-%m-%d")}

## Summary

- Status: {status}
- Total tests: {total}
- Passed: {passed}
- Failures: {failed}
- Warnings: {warnings}
- Duration: {duration_s}s

## Command

```
python -m pytest -vv --tb=short
```

## Full Test Log (Verbose)

```
{output}
```

## Results

- All tests {'passed' if status == 'PASS' else 'FAILED'}, including:
  - API integration tests (binary + polytomous)
  - Core math utilities (binary + polytomous)
  - Data handling
  - Quadrature
  - MML-EM and M-step solvers
  - Scoring
  - Plotting
  - External validation against R `mirt` and `ltm`

## Warnings (Expected in tests)

- JMLE extreme scores warning emitted during JMLE test
- Data filtering warnings emitted in tests that intentionally drop rows/cols
- Polytomous non-convergence warnings on some random-data tests (NRM, etc.)

## Notes

- `mirt` reference data generated via `tests/generate_mirt_reference.R`
- `ltm` reference data generated via `tests/generate_ltm_reference.R`
- Run `python scripts/run_tests_with_qa_report.py` or `make test` for full suite + QA report
"""

    QA_REPORT_PATH.write_text(report, encoding="utf-8")
    print(output)
    print(f"\n[QA report written to {QA_REPORT_PATH}]")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
