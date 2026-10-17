#!/usr/bin/env python3
"""Real subprocess execution and evidence-based verification for Python projects.

This is a constrained runner, NOT a security-grade OS sandbox. It uses argv (never a shell),
requires a workspace cwd, applies time/resource limits where supported, and records actual
stdout/stderr/exit status/duration. It does not claim network or filesystem isolation.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

MAX_OUTPUT_CHARS = 200_000

# Conservative policy gate. It is not a security boundary against hostile native code.
_RISKY_BINARIES = {"sudo", "su", "rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "poweroff", "mount",
                   "umount", "chmod", "chown", "useradd", "userdel", "iptables", "ssh", "scp", "sftp", "nft", "bash", "sh", "zsh",
                   "fish", "cmd", "powershell", "pwsh", "cp", "mv", "install", "truncate", "tee", "ln"}
_NETWORK_BINARIES = {"curl", "wget", "ssh", "scp", "sftp", "nc", "netcat", "socat", "ftp", "telnet", "ping"}
_NETWORK_MODULES = {"socket", "requests", "httpx", "urllib3", "aiohttp", "websockets", "ftplib", "smtplib",
                    "telnetlib", "paramiko", "openai", "anthropic", "grpc"}
_RISKY_MODULES = {"subprocess", "ctypes", "multiprocessing", "pty", "fcntl"}
_PATH_METHODS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "unlink", "rmdir", "mkdir",
                 "touch", "rename", "replace", "remove", "rmtree", "copyfile", "copy2", "move", "truncate"}


def _literal_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"Path", "PurePath"} and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _python_policy(source: str, workspace: pathlib.Path) -> list[str]:
    """Detect common unsafe imports/calls before execution; interpreter still executes permitted code normally."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []  # let the real interpreter report syntax errors
    issues: list[str] = []
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.Import):
            module = node.names[0].name.split(".")[0] if node.names else None
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module = node.module.split(".")[0]
        if module in _NETWORK_MODULES:
            issues.append(f"network import '{module}' requires --allow-network")
        if module in _RISKY_MODULES:
            issues.append(f"process/native-control import '{module}' requires --allow-risky")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else ""
            if owner == "os" and (name in {"system", "popen", "fork", "chmod", "chown"} or name.startswith("exec") or name.startswith("spawn")):
                issues.append(f"potentially destructive/process call 'os.{name}' requires --allow-risky")
            if owner == "subprocess" or (owner == "shutil" and name in {"rmtree", "move", "copyfile", "copy2"}):
                issues.append(f"process or file-management call '{owner}.{name}' requires --allow-risky")
            call_path = None
            if isinstance(func, ast.Name) and func.id == "open" and node.args:
                call_path = _literal_path(node.args[0])
            elif isinstance(func, ast.Attribute) and func.attr in _PATH_METHODS:
                call_path = _literal_path(func.value)
                if call_path is None and node.args and name in {"remove", "unlink", "rmtree", "move", "copyfile", "copy2", "rename", "replace"}:
                    call_path = _literal_path(node.args[0])
            if call_path is not None:
                path = pathlib.Path(call_path)
                try:
                    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
                    resolved.relative_to(workspace)
                except (ValueError, OSError):
                    issues.append(f"file path '{call_path}' escapes the workspace and requires --allow-outside-workspace")
    return sorted(set(issues))


def _policy_issues(argv: Sequence[str], workspace: pathlib.Path, allow_risky: bool,
                   allow_network: bool, allow_outside_workspace: bool) -> list[str]:
    if not argv:
        return ["empty command"]
    binary = pathlib.Path(argv[0]).name.lower()
    issues: list[str] = []
    if binary in _NETWORK_BINARIES and not allow_network:
        issues.append(f"network command '{binary}' requires --allow-network")
    if binary in _RISKY_BINARIES and not allow_risky:
        issues.append(f"privileged/destructive command '{binary}' requires --allow-risky")
    if binary in _RISKY_BINARIES or binary in {"curl", "wget", "tee", "cp", "mv", "install"}:
        for arg in argv[1:]:
            if not arg or arg.startswith("-"):
                continue
            candidate = pathlib.Path(arg)
            if not candidate.is_absolute() and ".." not in candidate.parts:
                continue
            try:
                resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
                resolved.relative_to(workspace)
            except (ValueError, OSError):
                if not allow_outside_workspace:
                    issues.append(f"argument path '{arg}' escapes the workspace and requires --allow-outside-workspace")
    source = None
    if binary.startswith(("python", "pypy")):
        if "-c" in argv:
            try: source = argv[list(argv).index("-c") + 1]
            except IndexError: source = None
        else:
            for arg in argv[1:]:
                path = pathlib.Path(arg)
                if path.suffix == ".py":
                    target = path if path.is_absolute() else workspace / path
                    try:
                        target = target.resolve()
                        target.relative_to(workspace)
                        if target.is_file(): source = target.read_text(encoding="utf-8", errors="replace")
                    except (OSError, ValueError):
                        pass
                    break
    elif pathlib.Path(argv[0]).suffix == ".py":
        try: source = pathlib.Path(argv[0]).read_text(encoding="utf-8", errors="replace")
        except OSError: source = None
    if source is not None:
        for issue in _python_policy(source, workspace):
            if "--allow-network" in issue and allow_network:
                continue
            if "--allow-risky" in issue and allow_risky:
                continue
            if "--allow-outside-workspace" in issue and allow_outside_workspace:
                continue
            issues.append(issue)
    return sorted(set(issues))


def _limits_preexec(cpu_seconds: int, memory_mb: int, file_mb: int, processes: int):
    """Build POSIX child limits. On platforms without resource, limits are unavailable."""
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:
        return None

    def apply_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_mb * 1024 * 1024, file_mb * 1024 * 1024))
        if hasattr(resource, "RLIMIT_AS"):
            memory = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
        if hasattr(resource, "RLIMIT_CORE"):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply_limits


def run_command(
    argv: Sequence[str], *, workspace: str | os.PathLike[str], timeout: float = 10.0,
    input_text: str | None = None, expected_stdout: str | None = None,
    expected_stderr: str | None = None, expected_exit_code: int = 0,
    cpu_seconds: int = 8, memory_mb: int = 512, file_mb: int = 16, processes: int = 32,
    env_overrides: dict[str, str] | None = None, allow_risky: bool = False,
    allow_network: bool = False, allow_outside_workspace: bool = False,
) -> dict[str, Any]:
    """Execute one real process and return an evidence record. argv is never interpreted by a shell."""
    start = time.monotonic()
    record: dict[str, Any] = {
        "command_argv": list(argv), "command_display": shlex.join(list(argv)),
        "workspace": str(pathlib.Path(workspace).resolve()), "input": input_text,
        "expected": {"stdout": expected_stdout, "stderr": expected_stderr,
                     "exit_code": expected_exit_code},
        "actual": {"stdout": "", "stderr": "", "exit_code": None},
        "duration_seconds": None, "timed_out": False, "status": "NOT_RUN", "error": None,
        "limits": {"timeout_seconds": timeout, "cpu_seconds": cpu_seconds,
                   "memory_mb": memory_mb, "file_mb": file_mb, "processes": processes},
        "isolation_note": "Temporary/workspace cwd and process limits only; filesystem and network are NOT fully sandboxed.",
    }
    if not argv or not all(isinstance(x, str) for x in argv):
        record["error"] = "argv must be a non-empty sequence of strings"
        record["status"] = "INVALID_COMMAND"
        record["duration_seconds"] = round(time.monotonic() - start, 6)
        return record
    if timeout <= 0 or cpu_seconds < 1 or memory_mb < 32 or file_mb < 1 or processes < 1:
        record["error"] = "invalid resource limit: timeout>0, cpu>=1, memory>=32MB, file>=1MB, processes>=1 required"
        record["status"] = "INVALID_LIMITS"
        record["duration_seconds"] = round(time.monotonic() - start, 6)
        return record
    root = pathlib.Path(workspace).resolve()
    if not root.is_dir():
        record["error"] = f"workspace is not a directory: {root}"
        record["status"] = "INVALID_WORKSPACE"
        record["duration_seconds"] = round(time.monotonic() - start, 6)
        return record
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(root), "TMPDIR": str(root),
           "TEMP": str(root), "TMP": str(root), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    issues = _policy_issues(argv, root, allow_risky, allow_network, allow_outside_workspace)
    if issues:
        record["status"] = "POLICY_BLOCKED"
        record["error"] = "Execution blocked before starting the process: " + "; ".join(issues)
        record["policy_issues"] = issues
        record["duration_seconds"] = round(time.monotonic() - start, 6)
        record["authorization"] = {"allow_risky": allow_risky, "allow_network": allow_network,
                                   "allow_outside_workspace": allow_outside_workspace}
        return record
    record["authorization"] = {"allow_risky": allow_risky, "allow_network": allow_network,
                               "allow_outside_workspace": allow_outside_workspace}
    try:
        # File-backed output plus RLIMIT_FSIZE avoids unbounded pipe buffering if a program prints forever.
        with tempfile.TemporaryFile(dir=str(root)) as out_file, tempfile.TemporaryFile(dir=str(root)) as err_file:
            proc = subprocess.Popen(
                list(argv), cwd=str(root), env=env, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=out_file, stderr=err_file, text=True, shell=False, close_fds=True,
                start_new_session=(os.name == "posix"),
                preexec_fn=_limits_preexec(cpu_seconds, memory_mb, file_mb, processes),
            )
            try:
                proc.communicate(input=input_text, timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:  # pragma: no cover - Windows only
                        proc.kill()
                except (OSError, ProcessLookupError):
                    try: proc.kill()
                    except OSError: pass
                try: proc.communicate(timeout=2)
                except subprocess.TimeoutExpired: pass
            out_file.flush(); err_file.flush()
            out_file.seek(0); err_file.seek(0)
            out_bytes = out_file.read(MAX_OUTPUT_CHARS + 1)
            err_bytes = err_file.read(MAX_OUTPUT_CHARS + 1)
            out = out_bytes.decode("utf-8", errors="replace")
            err = err_bytes.decode("utf-8", errors="replace")
            record["actual"].update({
                "stdout": out[:MAX_OUTPUT_CHARS], "stderr": err[:MAX_OUTPUT_CHARS],
                "exit_code": None if record["timed_out"] else proc.returncode,
                "stdout_truncated": len(out_bytes) > MAX_OUTPUT_CHARS,
                "stderr_truncated": len(err_bytes) > MAX_OUTPUT_CHARS,
            })
    except (OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["status"] = "EXECUTION_ERROR"
    record["duration_seconds"] = round(time.monotonic() - start, 6)
    if record["status"] == "EXECUTION_ERROR":
        return record
    if record["timed_out"]:
        record["status"] = "TIMEOUT"
        return record
    actual = record["actual"]
    mismatches = []
    if actual["exit_code"] != expected_exit_code:
        mismatches.append(f"exit code expected {expected_exit_code}, got {actual['exit_code']}")
    if expected_stdout is not None and actual["stdout"] != expected_stdout:
        mismatches.append("stdout did not match expected output")
    if expected_stderr is not None and actual["stderr"] != expected_stderr:
        mismatches.append("stderr did not match expected output")
    record["mismatches"] = mismatches
    record["status"] = "PASS" if not mismatches else "FAIL"
    return record


def run_python_file(script: str, *, workspace: str | os.PathLike[str], args: Sequence[str] = (), **kwargs) -> dict[str, Any]:
    """Run an actual Python file as a child process, with isolated mode enabled."""
    root = pathlib.Path(workspace).resolve()
    path = pathlib.Path(script)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return {"status": "INVALID_SCRIPT", "error": "script must be inside the workspace", "command_argv": [],
                "actual": {"stdout": "", "stderr": "", "exit_code": None}, "duration_seconds": 0.0,
                "timed_out": False}
    if not path.is_file():
        return {"status": "INVALID_SCRIPT", "error": f"script not found: {path}", "command_argv": [],
                "actual": {"stdout": "", "stderr": "", "exit_code": None}, "duration_seconds": 0.0,
                "timed_out": False}
    return run_command([sys.executable, "-I", str(path), *args], workspace=root, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real command and save its evidence report as JSON.")
    parser.add_argument("--workspace", default=".", help="existing working directory for the child process")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--expect-stdout", default=None, help="exact expected stdout, including newlines")
    parser.add_argument("--expect-stderr", default=None, help="exact expected stderr, including newlines")
    parser.add_argument("--expect-exit", type=int, default=0)
    parser.add_argument("--allow-risky", action="store_true", help="authorize commands that may change system/files/processes")
    parser.add_argument("--allow-network", action="store_true", help="authorize network-capable commands/imports")
    parser.add_argument("--allow-outside-workspace", action="store_true", help="authorize detected absolute/escaping file paths")
    parser.add_argument("--report", default="terminal_report.json", help="JSON report path, relative to workspace unless absolute")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command argv after --; shell syntax is not evaluated")
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    report = run_command(argv, workspace=args.workspace, timeout=args.timeout,
                         expected_stdout=args.expect_stdout, expected_stderr=args.expect_stderr,
                         expected_exit_code=args.expect_exit, allow_risky=args.allow_risky,
                         allow_network=args.allow_network, allow_outside_workspace=args.allow_outside_workspace)
    report_path = pathlib.Path(args.report)
    if not report_path.is_absolute(): report_path = pathlib.Path(args.workspace).resolve() / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
