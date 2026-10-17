"""Regression tests for actual subprocess evidence and safety-policy gates."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from terminal_verify import run_command, run_python_file


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="terminal_verify_test_") as td:
        root = Path(td)
        passed = run_command([sys.executable, "-c", "print('hello')"], workspace=root,
                             expected_stdout="hello\n", timeout=5, cpu_seconds=4, memory_mb=1024)
        check("real child process captures stdout, exit code and evidence", passed["status"] == "PASS"
              and passed["actual"]["stdout"] == "hello\n" and passed["actual"]["exit_code"] == 0)

        failed = run_command([sys.executable, "-c", "print('actual')"], workspace=root,
                             expected_stdout="expected\n", timeout=5, cpu_seconds=4, memory_mb=1024)
        check("output mismatch is reported as failure", failed["status"] == "FAIL"
              and "stdout did not match" in " ".join(failed.get("mismatches", [])))

        timeout = run_command([sys.executable, "-c", "import time; time.sleep(2)"], workspace=root,
                              timeout=0.2, cpu_seconds=4, memory_mb=1024)
        check("deadline terminates an overlong child process", timeout["status"] == "TIMEOUT" and timeout["timed_out"])

        risky = run_command(["bash", "-lc", "echo should-not-run"], workspace=root, timeout=3,
                            cpu_seconds=3, memory_mb=1024)
        check("risky shell is blocked unless explicitly authorized", risky["status"] == "POLICY_BLOCKED")

        network = run_command([sys.executable, "-c", "import socket; print('not run')"], workspace=root,
                              timeout=3, cpu_seconds=3, memory_mb=1024)
        check("network-capable Python import is blocked by default", network["status"] == "POLICY_BLOCKED")

        (root / "sample.py").write_text("print('file-ok')\n", encoding="utf-8")
        file_run = run_python_file("sample.py", workspace=root, expected_stdout="file-ok\n", timeout=5,
                                   cpu_seconds=4, memory_mb=1024)
        check("run_python_file executes workspace files and captures output", file_run["status"] == "PASS")

        outside = run_python_file("../outside.py", workspace=root)
        check("script path cannot escape workspace", outside["status"] == "INVALID_SCRIPT")

    print("\nAll terminal verifier tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
