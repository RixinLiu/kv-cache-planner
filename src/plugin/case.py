"""read_case shows the agent a case; case_record, the task's reference, pins the same case for the verifier."""

from __future__ import annotations

from pathlib import Path

from contract import CaseInput, CaseOutput
from plugin import cases


def read_case(given: CaseInput, workspace: Path) -> CaseOutput:
    return cases.case(given.case)


def case_record(given: CaseInput, workspace: Path) -> CaseOutput:
    return cases.case(given.case)
