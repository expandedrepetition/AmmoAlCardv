"""Check that the benchmark's bundled reference implementations satisfy their tests.

This is an offline data-integrity check, not a measure of the deployed model's accuracy.
"""
from __future__ import annotations

from eval_eim import PROBLEMS


def test_problem_ids_are_unique_and_complete() -> None:
    ids = [problem["id"] for problem in PROBLEMS]
    assert len(ids) == 25, f"Expected 25 benchmark problems, got {len(ids)}"
    assert len(ids) == len(set(ids)), "Benchmark problem IDs must be unique"


def test_each_problem_has_sufficient_assertions() -> None:
    for problem in PROBLEMS:
        assert problem.get("task", "").strip(), problem.get("id")
        assert len(problem.get("tests", [])) >= 5, problem.get("id")
        assert problem.get("reference", "").strip(), problem.get("id")


def test_all_reference_implementations_pass_their_assertions() -> None:
    for problem in PROBLEMS:
        namespace: dict = {"__name__": "__benchmark_reference__"}
        try:
            exec(compile(problem["reference"], f"<{problem['id']}-reference>", "exec"), namespace, namespace)
            for number, assertion in enumerate(problem["tests"], 1):
                exec(compile(assertion, f"<{problem['id']}-test-{number}>", "exec"), namespace, namespace)
        except Exception as exc:
            raise AssertionError(f"{problem['id']} failed its bundled reference test: {assertion!r}: {exc}") from exc


def main() -> None:
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} benchmark integrity tests passed ({len(PROBLEMS)} problems checked).")


if __name__ == "__main__":
    main()
