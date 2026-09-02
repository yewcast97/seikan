"""Shared test fixtures.

seikan is a stateless verifier — there is no database, no home directory, no shared state to wire.
Tests build their inputs (CSV files, thesis DSL dicts, synthetic summary dicts) under ``tmp_path``.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _scrub_seikan_env(monkeypatch):
    """Strip ambient ``SEIKAN_*`` vars so the suite is hermetic: ``GateThresholds`` reads the env
    at construction, and under ``extra="forbid"`` an unknown ``SEIKAN_*`` var is a hard error —
    a developer's shell must never bleed into (or break) a test. Tests that exercise env
    behavior set their own vars via ``monkeypatch`` afterwards."""
    for key in [k for k in os.environ if k.startswith("SEIKAN_")]:
        monkeypatch.delenv(key)
