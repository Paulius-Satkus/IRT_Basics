"""
Pytest configuration and shared fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Absolute path to the tests directory."""
    return Path(__file__).resolve().parent / "tests"


@pytest.fixture(scope="session")
def mirt_reference_dir(tests_dir: Path) -> Path:
    """Absolute path to mirt reference data."""
    return tests_dir / "mirt_reference"


@pytest.fixture(scope="session")
def ltm_reference_dir(tests_dir: Path) -> Path:
    """Absolute path to ltm reference data."""
    return tests_dir / "ltm_reference"
