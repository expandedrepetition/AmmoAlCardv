"""Offline tests for persistent explicit-correction memory; no HF credentials/network required."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from correction_memory import CorrectionMemory


def test_markers_and_non_corrections() -> None:
    assert CorrectionMemory.is_correction("هذا مو صحيح، لا تكرر نفس الخطأ")
    assert CorrectionMemory.is_correction("You repeated the same wrong answer")
    assert not CorrectionMemory.is_correction("Please explain recursion")


def test_add_deduplicates_and_persists() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "corrections.jsonl"
        memory = CorrectionMemory(path=str(path), repo="")
        assert memory.add("هذا غلط، لا تكرر الحل")
        assert not memory.add("  هذا غلط، لا تكرر الحل  ")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1 and rows[0]["text"] == "هذا غلط، لا تكرر الحل"
        assert CorrectionMemory(path=str(path), repo="").relevant("لا تكرر الحل")


def test_context_is_relevant_and_explains_conflict_rule() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = CorrectionMemory(path=str(Path(td) / "memory.jsonl"), repo="")
        memory.add("You repeated the same wrong coding approach; do not repeat it")
        prompt = memory.prompt("coding approach")
        assert "Persistent user corrections" in prompt
        assert "do not repeat it" in prompt
        assert "follow the current request" in prompt


def test_corrupt_lines_are_ignored() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.jsonl"
        path.write_text('{broken json\n{"ts": 1, "text": "that answer was wrong"}\n', encoding="utf-8")
        memory = CorrectionMemory(path=str(path), repo="")
        assert memory.relevant("wrong answer") == ["that answer was wrong"]


def test_record_limit_is_enforced() -> None:
    old_limit = CorrectionMemory.MAX_RECORDS
    try:
        CorrectionMemory.MAX_RECORDS = 3
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "memory.jsonl"
            memory = CorrectionMemory(path=str(path), repo="")
            for i in range(5):
                assert memory.add(f"this is wrong, please do not repeat lesson number {i}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 3
            assert [row["text"] for row in rows] == [
                "this is wrong, please do not repeat lesson number 2",
                "this is wrong, please do not repeat lesson number 3",
                "this is wrong, please do not repeat lesson number 4",
            ]
    finally:
        CorrectionMemory.MAX_RECORDS = old_limit


def test_remote_sync_is_disabled_by_default() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = CorrectionMemory(path=str(Path(td) / "memory.jsonl"), repo="")
        assert memory.sync_status == "disabled"
        assert memory.sync_now() is False


def main() -> None:
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} correction-memory tests passed.")


if __name__ == "__main__":
    main()
