"""EIM core engine: process-isolated verification, model adapters, memory, and repair strategies.

Backends:
  EIM_BACKEND=local  - Transformers model loaded in this process (GPU/CPU)
  EIM_BACKEND=hf     - Hugging Face Inference Providers (needs huggingface_hub and a token/provider)
  EIM_BACKEND=openai - any OpenAI-compatible chat-completions endpoint
  EIM_BACKEND=auto   - explicit compatible endpoint wins; on HF Spaces + HF_TOKEN uses HF inference;
                       otherwise local Transformers.

This verifier uses real child processes, deadlines and POSIX resource limits where available. It is NOT a
security-grade sandbox: Python code can still access host files and POSIX resource limits do not block all OS
interfaces. For untrusted public submissions run the entire app in a disposable container/VM with networking off.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

# Hugging Face ZeroGPU is optional. Import it before torch/transformers can initialise CUDA.
try:  # pragma: no cover - only exists on compatible Spaces
    import spaces as _spaces  # type: ignore
except Exception:  # ordinary local execution
    _spaces = None


def _identity_decorator(fn=None, **_kwargs):
    if fn is None:
        return lambda real_fn: real_fn
    return fn


def _is_zero_gpu_space() -> bool:
    hardware = os.environ.get("SPACE_HARDWARE", "").lower()
    return bool(os.environ.get("SPACES_ZERO_GPU") == "1" or
                (os.environ.get("SPACE_ID") and "zero" in hardware))


def _gpu_size() -> str:
    """ZeroGPU allocation size. `large` is the default 48 GB slice; `xlarge` requests 96 GB."""
    size = os.environ.get("EIM_GPU_SIZE", "large").strip().lower()
    return size if size in {"large", "xlarge"} else "large"


def _gpu_duration_seconds() -> int:
    """Clamp the duration to ZeroGPU's supported range and account for xlarge's doubled quota cost."""
    size = _gpu_size()
    default = "60" if size == "xlarge" else "120"
    try:
        requested = int(os.environ.get("EIM_GPU_SECONDS", default))
    except (TypeError, ValueError):
        requested = int(default)
    # xlarge consumes 2x quota, so keep the effective request at or below 120 seconds.
    ceiling = 60 if size == "xlarge" else 120
    return max(10, min(ceiling, requested))


if _spaces is not None and hasattr(_spaces, "GPU"):
    try:
        # Older spaces versions lacked `size`; keep `large` compatible and fail clearly for xlarge.
        try:
            gpu = _spaces.GPU(duration=_gpu_duration_seconds(), size=_gpu_size())
        except TypeError:
            if _gpu_size() == "xlarge":
                raise RuntimeError("EIM_GPU_SIZE=xlarge requires a newer `spaces` package; update requirements.txt")
            gpu = _spaces.GPU(duration=_gpu_duration_seconds())
    except Exception as exc:
        # Never silently strip the GPU decorator on a ZeroGPU Space: the Hub then
        # reports the opaque "No @spaces.GPU function detected" startup error.
        if _is_zero_gpu_space():
            raise RuntimeError(
                "This Space is configured for ZeroGPU, but spaces.GPU could not be initialized. "
                "Check the installed `spaces` version and EIM_GPU_SECONDS."
            ) from exc
        gpu = _identity_decorator
else:
    if _is_zero_gpu_space():
        raise RuntimeError(
            "This Space appears to use ZeroGPU, but the `spaces` package or spaces.GPU is missing. "
            "Install the `spaces` dependency from requirements.txt and rebuild the Space."
        )
    gpu = _identity_decorator


@dataclass
class Config:
    model_name: str = field(default_factory=lambda: os.environ.get("EIM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8"))
    per_test_seconds: float = field(default_factory=lambda: float(os.environ.get("EIM_PER_TEST_SECONDS", "3")))
    wall_seconds: float = field(default_factory=lambda: float(os.environ.get("EIM_WALL_SECONDS", "30")))
    cpu_seconds: int = field(default_factory=lambda: int(os.environ.get("EIM_CPU_SECONDS", "3")))
    memory_path: str = field(default_factory=lambda: os.environ.get("EIM_MEMORY_PATH", "eim_memory.json"))
    max_new_tokens: int = field(default_factory=lambda: int(os.environ.get("EIM_MAX_NEW_TOKENS", "1200")))
    min_gain: float = field(default_factory=lambda: float(os.environ.get("EIM_MIN_GAIN", "0.005")))
    memory_mb: int = field(default_factory=lambda: int(os.environ.get("EIM_TEST_MEMORY_MB", "768")))
    output_limit_bytes: int = field(default_factory=lambda: int(os.environ.get("EIM_TEST_OUTPUT_BYTES", str(1024 * 1024))))
    max_test_count: int = field(default_factory=lambda: int(os.environ.get("EIM_MAX_TESTS", "100")))
    model_revision: str = field(default_factory=lambda: os.environ.get("EIM_MODEL_REVISION", "main"))
    allow_network: bool = field(default_factory=lambda: os.environ.get("EIM_ALLOW_NETWORK", "0") == "1")
    allow_risky_code: bool = field(default_factory=lambda: os.environ.get("EIM_ALLOW_RISKY_CODE", "0") == "1")
    allow_outside_workspace: bool = field(default_factory=lambda: os.environ.get("EIM_ALLOW_OUTSIDE_WORKSPACE", "0") == "1")


CFG = Config()


DEFAULT_REPAIR_RULES = """Evidence-driven repair protocol:
1. Read the exact failing assertion, traceback, exit code, and timeout evidence before editing.
2. Identify the likely root cause, then make the smallest change that addresses it.
3. Preserve the requested function names and signatures; do not rewrite unrelated code.
4. Cover general inputs and boundaries; never hardcode test values, expected outputs, or lookup tables.
5. If an approach fails twice, change the algorithm or decomposition instead of renaming variables.
6. Prefer deterministic standard-library solutions; do not access network, secrets, unrelated files, or install packages.
7. Return one complete runnable implementation only. Never claim tests passed unless verifier results prove it."""


def _repair_rules() -> str:
    """Read an optional deployment repair policy, capped to prevent unbounded prompt growth."""
    return (os.environ.get("EIM_REPAIR_RULES", "").strip() or DEFAULT_REPAIR_RULES)[:5000]


class LanguageModel(Protocol):
    def generate(self, prompts: Sequence[str], temperature: float, max_new_tokens: int) -> list[str]: ...


@dataclass
class ExecResult:
    passed: int
    total: int
    failures: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    policy_blocked: bool = False

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class Score:
    reward: float
    correctness: float
    complexity_penalty: float = 0.0


@dataclass
class Mutation:
    strategy: str
    code: str


@dataclass
class Step:
    iteration: int
    label: str
    accepted: bool
    strategy: str
    code: str
    result: ExecResult
    score: Score


class ExperienceMemory:
    """Small, atomic persistent memory of failure-signature -> strategy outcomes."""
    def __init__(self, path: str = "eim_memory.json", *args, **kwargs):
        self.path = str(path or "")
        self._lock = threading.RLock()
        self._items: list[tuple[str, str]] = []
        if not self.path:
            return
        p = Path(self.path)
        try:
            raw = p.read_text(encoding="utf-8")
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = data.get("items", [])
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            sig, strat = item.get("signature"), item.get("strategy")
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            sig, strat = item[0], item[1]
                        else:
                            continue
                        if isinstance(sig, str) and isinstance(strat, str) and sig and strat:
                            self._items.append((sig, strat))
            except json.JSONDecodeError:
                # Gracefully accept older JSONL memory files.
                for line in raw.splitlines():
                    try:
                        item = json.loads(line)
                        sig, strat = item.get("signature"), item.get("strategy")
                        if isinstance(sig, str) and isinstance(strat, str):
                            self._items.append((sig, strat))
                    except Exception:
                        continue
        except (OSError, UnicodeError):
            pass
        self._items = self._items[-2000:]

    def add(self, signature: str, strategy: str) -> None:
        signature, strategy = str(signature).strip(), str(strategy).strip()
        if not signature or not strategy:
            return
        with self._lock:
            pair = (signature, strategy)
            if pair in self._items:
                self._items.remove(pair)
            self._items.append(pair)
            self._items = self._items[-2000:]
            if self.path:
                target = Path(self.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
                data = [{"signature": a, "strategy": b} for a, b in self._items]
                try:
                    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(tmp, target)
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass

    def hint(self, signature: str) -> str:
        if not signature:
            return ""
        kind = signature.split("|", 1)[0]
        with self._lock:
            items = list(self._items[-400:])
        votes: Counter[str] = Counter()
        q = _canonical_failure(signature)
        for rank, (stored, strategy) in enumerate(items):
            if stored.split("|", 1)[0] != kind:
                continue
            s = _canonical_failure(stored)
            sim = difflib.SequenceMatcher(None, q, s).ratio()
            if sim >= 0.60:
                votes[strategy] += sim * (1.0 + rank / max(1, len(items)))
        return votes.most_common(1)[0][0] if votes else ""


STRATEGIES = frozenset({"repair", "rewrite", "edge_cases", "simplify", "algorithm", "performance", "restart"})


class IFC:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg

    def signature(self, result: ExecResult) -> str:
        if result.timed_out:
            kind = "timeout"
        elif not result.total or result.passed == 0:
            kind = "zero"
        elif result.pass_rate < 1.0:
            kind = "partial"
        else:
            kind = "success"
        content = "\n".join(result.failures[:2]) or result.stderr or result.stdout
        return f"{kind}|{_first_error_line(content)[:260]}"

    def feedback(self, code: str, result: ExecResult, score: Score) -> str:
        lines = [f"Verified tests: {result.passed}/{result.total}; pass rate={result.pass_rate:.1%}; reward={score.reward:.3f}."]
        if result.timed_out:
            lines.append("At least one test timed out. Check algorithmic complexity and accidental infinite loops.")
        if result.failures:
            lines.append("ACTUAL TEST FAILURES (read the exact output; fix the root cause):")
            lines.extend(f"- {x[:1800]}" for x in result.failures[:6])
        if result.stderr:
            lines.append("Captured stderr:\n" + result.stderr[:2500])
        lines.append("Do not repeat the rejected implementation verbatim. Preserve the required function names, reason from the observed errors, and cover edge cases.")
        return "\n".join(lines)

    def converged(self, result: ExecResult, stale: int) -> bool:
        return result.pass_rate >= 1.0 or stale >= 4


class EUTV:
    """Run each candidate/test suite in a real OS child; assertions get separate namespaces and individual deadlines."""
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg

    def _safety_issues(self, code: str, tests: Sequence[str]) -> list[str]:
        try:
            from terminal_verify import _python_policy
        except ImportError:
            return ["the terminal safety policy module is missing; execution is refused"]
        issues: list[str] = []
        # A throwaway root only lets the analyzer distinguish relative paths from absolute escapes.
        with tempfile.TemporaryDirectory(prefix="eim_policy_") as policy_dir:
            root = Path(policy_dir).resolve()
            for label, source in [("candidate", code), *((f"test {i}", test) for i, test in enumerate(tests, 1))]:
                for issue in _python_policy(source, root):
                    if "--allow-network" in issue and self.cfg.allow_network:
                        continue
                    if "--allow-risky" in issue and self.cfg.allow_risky_code:
                        continue
                    if "--allow-outside-workspace" in issue and self.cfg.allow_outside_workspace:
                        continue
                    issues.append(f"{label}: {issue}")
        return sorted(set(issues))

    def _run_suite(self, code: str, tests: Sequence[str], deadline: float) -> ExecResult:
        """Run a whole assertion suite in one real child process; each assertion gets a fresh namespace and timer."""
        started = time.monotonic()
        clean_tests = [str(t).strip() for t in tests if str(t).strip()]
        total = len(clean_tests)
        remaining = deadline - started
        if remaining <= 0:
            return ExecResult(0, total, ["overall wall-clock deadline exhausted before execution"], 0.0, True, "", "", -1)
        with tempfile.TemporaryDirectory(prefix="eim_verify_") as td:
            root = Path(td)
            (root / "solution.py").write_text(code + "\n", encoding="utf-8")
            runner = root / "_eim_runner.py"
            runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
            tests_path = root / "_eim_tests.json"
            tests_path.write_text(json.dumps(clean_tests, ensure_ascii=False), encoding="utf-8")
            out_path, err_path = root / "stdout.log", root / "stderr.log"
            env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "WINDIR": os.environ.get("WINDIR", ""),
                "TEMP": td,
                "TMP": td,
                "TMPDIR": td,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": td,
                "LANG": "C.UTF-8",
                "EIM_ALLOW_NETWORK_TESTS": "1" if self.cfg.allow_network else "0",
                "EIM_PER_TEST_SECONDS": str(max(0.05, float(self.cfg.per_test_seconds))),
            }
            timed_out = False
            proc = None
            try:
                with open(out_path, "wb") as out_handle, open(err_path, "wb") as err_handle:
                    kwargs = {}
                    if os.name == "posix":
                        kwargs["start_new_session"] = True
                        kwargs["preexec_fn"] = lambda: _apply_limits(self.cfg, cpu_multiplier=len(clean_tests))
                    proc = subprocess.Popen(
                        [sys.executable, "-I", str(runner), str(tests_path)], cwd=td, env=env,
                        stdin=subprocess.DEVNULL, stdout=out_handle, stderr=err_handle, **kwargs
                    )
                    try:
                        proc.wait(timeout=max(0.05, min(float(self.cfg.wall_seconds), remaining)))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        _kill_process_tree(proc)
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
            except Exception as exc:
                return ExecResult(0, total, [f"runner error: {type(exc).__name__}: {exc}"],
                                  time.monotonic() - started, False, "", "", -1)

            stdout = _read_capped(out_path, self.cfg.output_limit_bytes)
            stderr = _read_capped(err_path, self.cfg.output_limit_bytes)
            elapsed = time.monotonic() - started
            exit_code = int(proc.returncode if proc is not None and proc.returncode is not None else -1)
            summaries = re.findall(r"(?m)^EIM_SUMMARY::(\{.*\})\s*$", stdout)
            payload = None
            if summaries:
                try:
                    candidate = json.loads(summaries[-1])
                    if isinstance(candidate, dict) and candidate.get("total") == total:
                        payload = candidate
                except (ValueError, TypeError):
                    payload = None
            if payload is not None:
                passed = max(0, min(total, int(payload.get("passed", 0))))
                rows = payload.get("failures", [])
                if any(isinstance(row, dict) and "assertion exceeded per-test timeout" in str(row.get("error", "")) for row in rows):
                    timed_out = True
                failures = []
                for row in rows[:max(1, total)]:
                    if not isinstance(row, dict):
                        continue
                    idx = int(row.get("index", 0))
                    detail = str(row.get("error", "assertion failed"))[:3500]
                    failures.append(f"TEST {idx}/{total} FAILED: {detail}")
                if timed_out and passed < total and not failures:
                    failures.append("TIMEOUT: suite deadline exhausted")
                return ExecResult(passed, total, failures, elapsed, timed_out, stdout, stderr, exit_code)

            # Missing summary is a failure even if candidate code called os._exit(0); never infer success.
            passed_markers = len(re.findall(r"(?m)^EIM_ASSERT_PASS \d+/\d+\s*$", stdout))
            passed = max(0, min(total, passed_markers))
            why = "TIMEOUT: suite deadline exhausted" if timed_out else \
                  f"runner exited without a valid result summary (exit code {exit_code})"
            if stderr:
                why += "\nstderr:\n" + stderr[-3500:]
            if stdout:
                why += "\nstdout:\n" + stdout[-1500:]
            return ExecResult(passed, total, [why], elapsed, timed_out, stdout, stderr, exit_code)

    def _run_one(self, code: str, test: str, index: int, deadline: float) -> tuple[bool, str, float, bool, int, str, str]:
        """Compatibility helper retained for callers/tests that need one assertion."""
        result = self._run_suite(code, [test], deadline)
        return (result.pass_rate == 1.0, "\n".join(result.failures), result.elapsed,
                result.timed_out, result.exit_code, result.stdout, result.stderr)

    def run(self, code: str, tests: Sequence[str]) -> ExecResult:
        clean_tests = [str(t).strip() for t in list(tests)[:max(0, self.cfg.max_test_count)] if str(t).strip()]
        started = time.monotonic()
        if not clean_tests:
            return ExecResult(0, 0, ["No executable assertions were supplied."], 0.0, False, "", "", 2)
        policy_issues = self._safety_issues(code, clean_tests)
        if policy_issues:
            detail = "NOT EXECUTED — SAFETY POLICY BLOCKED before starting any child process:\n" + "\n".join(policy_issues[:12])
            return ExecResult(0, len(clean_tests), [detail], time.monotonic() - started, False,
                              "", detail, 126, True)
        deadline = started + max(0.1, float(self.cfg.wall_seconds))
        return self._run_suite(code, clean_tests, deadline)

    def run_many(self, codes: Sequence[str], tests: Sequence[Sequence[str]]) -> list[ExecResult]:
        return [self.run(code, suite) for code, suite in zip(codes, tests)]


_RUNNER_SOURCE = r'''import json, os, signal, socket, sys, traceback
# Conservative Python-level network denial; explicit config authorizes network access.
def _blocked(*args, **kwargs):
    raise PermissionError("network access is disabled in EIM verification")
if os.environ.get("EIM_ALLOW_NETWORK_TESTS") != "1":
    socket.socket.connect = _blocked
    socket.create_connection = _blocked
sys.path.insert(0, os.getcwd())
try:
    import solution
except BaseException:
    traceback.print_exc()
    raise SystemExit(2)
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        tests = json.load(handle)
except BaseException:
    traceback.print_exc()
    raise SystemExit(2)
passed = 0
failures = []
try:
    per_test = max(0.05, float(os.environ.get("EIM_PER_TEST_SECONDS", "3")))
except Exception:
    per_test = 3.0
for i, source in enumerate(tests, 1):
    namespace = dict(vars(solution))
    namespace["__name__"] = f"__eim_test_{i}__"
    namespace["__file__"] = f"<eim_test_{i}>"
    old_handler = None
    timer_enabled = False
    try:
        if hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
            old_handler = signal.getsignal(signal.SIGALRM)
            def _test_timeout(_signum, _frame):
                raise TimeoutError(f"assertion exceeded per-test timeout ({per_test:.2f}s)")
            signal.signal(signal.SIGALRM, _test_timeout)
            signal.setitimer(signal.ITIMER_REAL, per_test)
            timer_enabled = True
        exec(compile(str(source), f"<eim_test_{i}>", "exec"), namespace, namespace)
        passed += 1
        print(f"EIM_ASSERT_PASS {i}/{len(tests)}")
    except BaseException:
        detail = traceback.format_exc()
        failures.append({"index": i, "error": detail[-4000:]})
        print(f"EIM_ASSERT_FAIL {i}/{len(tests)}", file=sys.stderr)
        print(detail, file=sys.stderr)
    finally:
        if timer_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
summary = {"passed": passed, "total": len(tests), "failures": failures}
print("EIM_SUMMARY::" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)
raise SystemExit(0 if passed == len(tests) else 1)
'''
def _apply_limits(cfg: Config, cpu_multiplier: int = 1) -> None:
    try:
        import resource
        cpu = max(1, min(int(cfg.cpu_seconds) * max(1, int(cpu_multiplier)), max(int(cfg.cpu_seconds), int(cfg.wall_seconds))))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        memory = max(128, int(cfg.memory_mb)) * 1024 * 1024
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        if hasattr(resource, "RLIMIT_FSIZE"):
            size = max(65536, int(cfg.output_limit_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        if hasattr(resource, "RLIMIT_NOFILE"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    except Exception:
        # Unsupported OS/runtime limits are documented and the timeout still applies.
        pass


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows only
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


def _read_capped(path: Path, limit: int) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(max(1, int(limit)) + 1)
        clipped = len(data) > limit
        data = data[:max(1, int(limit))]
        out = data.decode("utf-8", errors="replace")
        if clipped:
            out += "\n[output truncated at configured cap]"
        return out
    except OSError:
        return ""


def _first_error_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line and not line.startswith("File \"") and not line.startswith("During handling"):
            return line
    return str(text).strip()[:260]


def _canonical_failure(text: str) -> str:
    text = re.sub(r"\b\d+\b", "#", str(text).lower())
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'x'", text)
    return re.sub(r"\s+", " ", text).strip()


class DCME:
    """Diverse candidate generation; repeated code is rejected before verification."""
    def __init__(self, lm: LanguageModel, memory: ExperienceMemory, cfg: Config = CFG):
        self.lm, self.memory, self.cfg = lm, memory, cfg

    def propose(self, task: str, code: str, feedback: str, result: ExecResult, k: int,
                temperature: float, tried: set[str]) -> list[Mutation]:
        k = max(0, min(int(k), 12))
        if not k:
            return []
        hint = self.memory.hint(IFC(self.cfg).signature(result))
        strategies = [hint] if hint in STRATEGIES else []
        pool = ["repair", "edge_cases", "rewrite", "algorithm", "simplify", "performance", "restart"]
        for strategy in pool:
            if strategy not in strategies:
                strategies.append(strategy)
        prompts = []
        for i in range(k):
            strategy = strategies[i % len(strategies)]
            prompts.append(_mutation_prompt(task, code, feedback, strategy, i))
        try:
            outputs = self.lm.generate(prompts, float(temperature), min(4096, max(64, int(self.cfg.max_new_tokens))))
        except Exception as exc:
            return []
        out: list[Mutation] = []
        local_seen: set[str] = set()
        for i, raw in enumerate(outputs or []):
            candidate = normalise(extract_code(str(raw)))
            digest = hashlib.sha256(candidate.encode("utf-8", "ignore")).hexdigest()
            already = any(hashlib.sha256(normalise(prev).encode("utf-8", "ignore")).hexdigest() == digest for prev in tried)
            if not candidate or already or digest in local_seen or candidate == code:
                continue
            local_seen.add(digest)
            out.append(Mutation(strategies[i % len(strategies)], candidate))
        return out


def _mutation_prompt(task: str, code: str, feedback: str, strategy: str, variant: int) -> str:
    focus = {
        "repair": "Make the smallest correct fix to the root cause in the exact diagnostics.",
        "rewrite": "Re-derive the specification and rewrite the implementation cleanly without copying the faulty approach.",
        "edge_cases": "Prioritise empty inputs, one item, duplicates, negative numbers, boundaries and invalid types if relevant.",
        "algorithm": "Choose a different, well-understood algorithm and justify it internally against the specification.",
        "simplify": "Remove unnecessary branches and state; prefer the simplest correct implementation.",
        "performance": "Keep correctness first, then reduce time and space complexity for large inputs.",
        "restart": "Ignore the previous implementation and derive a general solution from scratch.",
    }.get(strategy, "Fix the root cause.")
    return (f"You are repairing Python code. Task:\n{task}\n\nCurrent implementation:\n```python\n{code[:12000]}\n```\n\n"
            f"Real verifier diagnostics:\n{feedback[:5000]}\n\nStrategy for this candidate: {focus}\n\n"
            "Rules: return the complete corrected Python code only; preserve required function names/signatures; do not hardcode test inputs or expected answers; "
            "do not use network; do not repeat the same failed implementation; use standard-library features unless task requires otherwise.\n\n"
            "REPAIR PROTOCOL:\n" + _repair_rules())


class _ScriptedLM:
    """Deterministic test double; it is used only by unit tests and never by production get_lm()."""
    def __init__(self, answers: Sequence[str]):
        self.answers = list(answers)
        self.i = 0
        self.calls: list[tuple[list[str], float, int]] = []

    def generate(self, prompts: Sequence[str], temperature: float, max_new_tokens: int) -> list[str]:
        batch = list(prompts)
        self.calls.append((batch, temperature, max_new_tokens))
        out = []
        for _ in batch:
            if not self.answers:
                out.append("")
            else:
                out.append(self.answers[min(self.i, len(self.answers) - 1)])
            self.i += 1
        return out


def normalise(code: str) -> str:
    code = str(code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    code = re.sub(r"^\s*```(?:python|py)?\s*\n", "", code, count=1, flags=re.I)
    code = re.sub(r"\n?```\s*$", "", code, count=1)
    return code.strip() + ("\n" if code.strip() else "")


def extract_code(text: str) -> str:
    text = str(text or "").strip()
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.I | re.S)
    if fenced:
        # Prefer a block containing Python definitions or imports; otherwise first block.
        for block in fenced:
            if re.search(r"(?m)^\s*(?:def |class |async def |import |from )", block):
                return normalise(block)
        return normalise(fenced[0])
    return normalise(text)


def draft_prompt(task: str, tests: Sequence[str]) -> str:
    shown = "\n".join(f"- {t}" for t in tests[:40])
    return ("You are a careful Python engineer. Implement the user's task as general-purpose Python code.\n"
            "Return the complete implementation only, with no Markdown fences or explanation. Do not hardcode expected outputs.\n\n"
            f"TASK:\n{task}\n\nASSERTIONS TO PASS:\n{shown}\n\n"
            "Respect names and signatures referenced by assertions. Consider edge cases and invalid inputs. Do not access network or unrelated files.\n\n"
            "REPAIR PROTOCOL:\n" + _repair_rules())


def score(code: str, result: ExecResult, cfg: Config = CFG) -> Score:
    correct = result.pass_rate
    # Correctness dominates. Penalise excessive code slightly to break ties without displacing a passing solution.
    complexity = min(0.04, max(0, len(code.strip()) - 1500) / 100000.0)
    timeout_penalty = 0.03 if result.timed_out else 0.0
    reward = 100.0 * correct - complexity - timeout_penalty
    return Score(reward, correct, complexity + timeout_penalty)


def render_model_prompt(tok, prompt: str) -> str:
    """Render one user message with the tokenizer's chat template when available."""
    prompt = str(prompt)
    messages = [{"role": "user", "content": prompt}]
    if tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "### Instruction:\n" + prompt + "\n\n### Response:\n"


class CodeLM:
    """Lazy local Transformers or remote API model. Remote mode avoids model weights on small free servers."""
    def __init__(self, name: str | None = None, backend: str | None = None):
        self.model_name = name or CFG.model_name
        self.backend = self._choose_backend((backend or os.environ.get("EIM_BACKEND", "auto")).lower())
        self.model = None
        self.tokenizer = None
        self.is_remote = self.backend in {"hf", "openai"}
        self._client = None
        self._lock = threading.RLock()

    @staticmethod
    def _choose_backend(backend: str) -> str:
        if backend in {"hf", "huggingface", "remote"}:
            return "hf"
        if backend in {"openai", "openai-compatible", "api"}:
            return "openai"
        if backend == "local":
            return "local"
        if backend != "auto":
            raise ValueError("EIM_BACKEND must be auto, local, hf, or openai")
        if os.environ.get("EIM_API_BASE") or os.environ.get("OPENAI_BASE_URL"):
            return "openai"
        # On ZeroGPU, prefer local weights. A token alone does not prove that a provider hosts the requested model.
        zero_gpu = os.environ.get("SPACES_ZERO_GPU") == "1" or (
            bool(os.environ.get("SPACE_ID")) and "zero" in os.environ.get("SPACE_HARDWARE", "").lower()
        )
        if zero_gpu:
            return "local"
        # A Hub token grants access; it does NOT mean this model is deployed on an
        # Inference Provider. Use local weights by default; explicitly set
        # EIM_BACKEND=hf only for a model/provider combination that is deployed.
        return "local"

    def _load_local(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Local backend needs torch and transformers; install requirements.txt or set EIM_BACKEND=hf with HF_TOKEN") from exc
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or None
        cache_dir = os.environ.get("HF_HOME") or None
        kwargs = {"token": token, "cache_dir": cache_dir, "revision": CFG.model_revision, "trust_remote_code": False}
        if token is None:
            kwargs.pop("token")
        if cache_dir is None:
            kwargs.pop("cache_dir")
        tok = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = "left"
        # ZeroGPU exposes CUDA through a startup-time emulation layer. Treat a detected ZeroGPU
        # Space as CUDA-capable here so weights are placed before the real GPU callback begins.
        cuda = torch.cuda.is_available() or _is_zero_gpu_space()
        mps = bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
        model_kwargs = dict(kwargs)
        # Keep the checkpoint's declared dtype by default (needed by FP8 deployments),
        # but allow Kaggle T4 users to select fp16 explicitly because T4 does not
        # support native BF16/FP8 execution like newer accelerators.
        dtype_name = os.environ.get("EIM_TORCH_DTYPE", "auto").strip().lower()
        if dtype_name in {"auto", ""}:
            model_kwargs["torch_dtype"] = "auto"
        elif dtype_name in {"fp16", "float16", "half"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype_name in {"bf16", "bfloat16"}:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype_name in {"fp32", "float32"}:
            model_kwargs["torch_dtype"] = torch.float32
        else:
            raise ValueError("EIM_TORCH_DTYPE must be auto, fp16, bf16, or fp32")
        model_kwargs["low_cpu_mem_usage"] = True
        if cuda:
            # `device_map` supports larger checkpoints without a second full copy in RAM.
            # accelerate is listed in requirements.txt. Keep a conservative fallback
            # for environments where dispatch is unavailable.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto", **model_kwargs
                )
            except (ImportError, ValueError) as exc:
                if "accelerate" not in str(exc).lower() and "device_map" not in str(exc).lower():
                    raise
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                model.to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            if mps:
                model.to("mps")
        model.eval()
        self.tokenizer, self.model = tok, model
        print("=== MODEL CHECK ===")
        print("Model:", self.model.config._name_or_path)
        print("Parameters:", f"{self.model.num_parameters():,}")
        print("Layers:", getattr(self.model.config, "num_hidden_layers", "unknown"))
        print("Device:", next(self.model.parameters()).device)
        print("=== END MODEL CHECK ===")
        
    def _hf_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError("Hugging Face backend requires `huggingface_hub`. Install requirements.txt.") from exc
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            if not token:
                raise RuntimeError("Set HF_TOKEN in your environment or Hugging Face Space Secrets to use the HF Inference API.")
            timeout = float(os.environ.get("EIM_API_TIMEOUT", "90"))
            self._client = InferenceClient(model=self.model_name, token=token, timeout=timeout)
        return self._client

    def _remote_one(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        if self.backend == "hf":
            client = self._hf_client()
            try:
                out = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max(16, min(4096, int(max_new_tokens))),
                    temperature=max(0.0, min(1.5, float(temperature))),
                )
                content = out.choices[0].message.content
                if isinstance(content, list):
                    return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
                return str(content or "")
            except Exception as first:
                # Some HF providers expose only text generation for a given model.
                try:
                    result = client.text_generation(prompt, max_new_tokens=max(16, min(4096, int(max_new_tokens))),
                                                    temperature=max(0.01, min(1.5, float(temperature))),
                                                    do_sample=temperature > 0, return_full_text=False)
                    return str(result)
                except Exception as second:
                    raise RuntimeError(f"Hugging Face inference failed ({type(first).__name__}: {first}); text-generation fallback failed ({type(second).__name__}: {second})") from second
        base = os.environ.get("EIM_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
        if not base:
            raise RuntimeError("Set EIM_API_BASE (or OPENAI_BASE_URL) for the OpenAI-compatible backend")
        api_key = os.environ.get("EIM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        url = base.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = json.dumps({"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                           "temperature": max(0.0, min(1.5, float(temperature))),
                           "max_tokens": max(16, min(4096, int(max_new_tokens)))}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(os.environ.get("EIM_API_TIMEOUT", "90"))) as response:
                payload = json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"OpenAI-compatible inference failed: {type(exc).__name__}: {exc}") from exc

    def generate(self, prompts: Sequence[str], temperature: float, max_new_tokens: int) -> list[str]:
        if self.is_remote:
            out = []
            for prompt in prompts:
                out.append(self._remote_one(str(prompt), temperature, max_new_tokens))
            return out
        with self._lock:
            self._load_local()
            import torch
            tok, model = self.tokenizer, self.model
            results: list[str] = []
            for prompt in prompts:
                rendered = render_model_prompt(tok, prompt)
                inputs = tok(rendered, return_tensors="pt", truncation=True, max_length=max(512, int(os.environ.get("EIM_INPUT_TOKENS", "8192"))))
                try:
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                options = {"max_new_tokens": max(16, min(4096, int(max_new_tokens))),
                           "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                           "use_cache": True, "repetition_penalty": 1.06}
                if float(temperature) > 0:
                    options.update(do_sample=True, temperature=max(0.01, min(1.5, float(temperature))), top_p=0.9)
                else:
                    options["do_sample"] = False
                with torch.inference_mode():
                    output = model.generate(**inputs, **options)
                prompt_len = inputs["input_ids"].shape[1]
                results.append(tok.decode(output[0][prompt_len:], skip_special_tokens=True).strip())
            return results


_LM_SINGLETON: CodeLM | None = None
_LM_LOCK = threading.RLock()


def get_lm() -> CodeLM:
    global _LM_SINGLETON
    with _LM_LOCK:
        wanted_name = os.environ.get("EIM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
        wanted_backend = os.environ.get("EIM_BACKEND", "auto")
        if _LM_SINGLETON is None or _LM_SINGLETON.model_name != wanted_name or _LM_SINGLETON.backend != CodeLM._choose_backend(wanted_backend):
            _LM_SINGLETON = CodeLM(wanted_name, wanted_backend)
        return _LM_SINGLETON


def _zero_gpu_decorator() -> bool:
    hardware = os.environ.get("SPACE_HARDWARE", "").lower()
    return bool(os.environ.get("SPACES_ZERO_GPU") == "1" or (os.environ.get("SPACE_ID") and "zero" in hardware))


def _preload_zerogpu_model() -> None:
    """Pre-place local weights before callbacks, as required by current ZeroGPU guidance.

    This is best-effort so a transient model-download error does not prevent the Gradio UI from booting.
    Set EIM_ZEROGPU_PRELOAD=0 only for debugging; lazy loading remains a recovery path if preloading fails.
    """
    if not _zero_gpu_decorator() or os.environ.get("EIM_ZEROGPU_PRELOAD", "1") == "0":
        return
    try:
        if CodeLM._choose_backend(os.environ.get("EIM_BACKEND", "auto")) != "local":
            return
        get_lm()._load_local()
    except Exception as exc:
        print(f"WARNING: ZeroGPU model pre-load failed; a request may retry: {type(exc).__name__}: {exc}", file=sys.stderr)


def _verify_backend() -> str:
    """Return a harmless backend diagnostic without loading/downloading model weights."""
    backend = CodeLM._choose_backend(os.environ.get("EIM_BACKEND", "auto"))
    print(f"Backend: {backend}")
    print(f"Model: {os.environ.get('EIM_MODEL', CFG.model_name)}")
    print(f"Hugging Face Space: {'yes' if os.environ.get('SPACE_ID') else 'no'}")
    print(f"ZeroGPU marker: {'yes' if _zero_gpu_decorator() else 'no'}")
    print(f"ZeroGPU size: {_gpu_size()}")
    print(f"GPU request duration: {_gpu_duration_seconds()} seconds")
    return backend


# Current Hugging Face ZeroGPU guidance recommends loading and placing model weights on CUDA
# at module startup (the runtime provides CUDA emulation outside the decorated callback).
# Keep this best-effort and skip it entirely for CPU/dedicated-GPU Spaces and remote backends.
_preload_zerogpu_model()


def run_selftest() -> int:
    """Core tests that run without model weights or network access."""
    checks = 0
    failures: list[str] = []

    def check(name: str, fn) -> None:
        nonlocal checks
        checks += 1
        try:
            result = fn() if callable(fn) else bool(fn)
            if not result:
                raise AssertionError("condition was false")
            print(f"PASS {name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    cfg = Config(memory_path="", per_test_seconds=1, wall_seconds=8, cpu_seconds=3, memory_mb=512)
    verifier = EUTV(cfg)
    check("real child process returns correct output", lambda: verifier.run("def f(x): return x+1", ["assert f(2) == 3"]).pass_rate == 1)
    check("failed assertion is captured", lambda: "AssertionError" in "\n".join(verifier.run("def f(x): return x", ["assert f(2) == 3"]).failures))
    check("syntax error is captured", lambda: "SyntaxError" in "\n".join(verifier.run("def f(: pass", ["assert True"]).failures))
    check("timeout is recorded", lambda: verifier.run("def f():\n while True: pass", ["f()" ]).timed_out)
    check("imports from solution module work", lambda: verifier.run("def f(x): return x*2", ["from solution import f\nassert f(4) == 8"]).pass_rate == 1)
    check("no-test input fails closed", lambda: verifier.run("x=1", []).pass_rate == 0)
    check("candidate generator rejects duplicate answers", lambda: len(DCME(_ScriptedLM(["def f(): return 1"]), ExperienceMemory(path="" ), cfg).propose("f", "", "", ExecResult(0,1,["AssertionError"]), 2, 0.2, {"def f(): return 1"})) == 0)
    check("code extraction strips fences", lambda: "def f" in extract_code("```python\ndef f(): return 1\n```"))
    check("correctness dominates reward", lambda: score("x", ExecResult(2,2), cfg).reward > score("x", ExecResult(1,2), cfg).reward)
    check("HF/local backend selection", lambda: CodeLM._choose_backend("local") == "local" and CodeLM._choose_backend("hf") == "hf")
    check("requested Qwen model is the default", _test_model_default)
    check("draft prompt contains evidence-driven repair rules", _test_repair_prompt)
    check("chat template is used when the tokenizer exposes one", lambda: _test_prompt_template())
    check("fallback prompt used if no chat template exists", lambda: render_model_prompt(None, "x") == "### Instruction:\nx\n\n### Response:\n")
    check("Space auto-backend uses the real SPACE_ID variable", _space_auto_backend_test)
    check("ZeroGPU Space is detected before silently falling back to a no-op decorator", _zero_gpu_environment_test)
    check("ZeroGPU request duration stays within the documented ceiling", _gpu_duration_test)
    check("network-capable source is blocked before child execution", _policy_denied(cfg))
    check("memory persistence and reload", lambda: _memory_roundtrip(cfg))
    print(f"\nSelf-test: {checks - len(failures)}/{checks} passed; {len(failures)} failed.")
    for failure in failures:
        print("FAIL DETAIL:", failure)
    return 0 if not failures else 1


def _policy_denied(cfg: Config) -> bool:
    result = EUTV(cfg).run("import socket\ndef f(): return 1", ["assert True"])
    return result.policy_blocked and result.passed == 0 and "NOT EXECUTED" in result.failures[0]


def _memory_roundtrip(cfg: Config) -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "memory.json")
        mem = ExperienceMemory(path)
        mem.add("partial|AssertionError", "repair")
        return ExperienceMemory(path).hint("partial|AssertionError") == "repair"


def _test_model_default() -> bool:
    old = os.environ.pop("EIM_MODEL", None)
    try:
        return Config().model_name == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    finally:
        if old is not None:
            os.environ["EIM_MODEL"] = old


def _test_repair_prompt() -> bool:
    old = os.environ.get("EIM_REPAIR_RULES")
    try:
        os.environ.pop("EIM_REPAIR_RULES", None)
        default_ok = "Evidence-driven repair protocol" in draft_prompt("add", ["assert add(1, 2) == 3"])
        os.environ["EIM_REPAIR_RULES"] = "CUSTOM_REPAIR_POLICY"
        custom_ok = "CUSTOM_REPAIR_POLICY" in draft_prompt("add", ["assert add(1, 2) == 3"])
        mutation_has_rules = "CUSTOM_REPAIR_POLICY" in _mutation_prompt("add", "def add(a,b): return a+b", "AssertionError", "repair", 0)
        os.environ["EIM_REPAIR_RULES"] = "X" * 9000
        capped = len(_repair_rules()) == 5000
        return default_ok and custom_ok and capped and mutation_has_rules
    finally:
        if old is None:
            os.environ.pop("EIM_REPAIR_RULES", None)
        else:
            os.environ["EIM_REPAIR_RULES"] = old


def _test_prompt_template() -> bool:
    class _Tokenizer:
        chat_template = "available"
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False and add_generation_prompt is True
            return "<user>hello<assistant>"
    return render_model_prompt(_Tokenizer(), "hello") == "<user>hello<assistant>"


def _zero_gpu_environment_test() -> bool:
    keys = ("SPACE_ID", "SPACE_HARDWARE", "SPACES_ZERO_GPU")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        normal = not _is_zero_gpu_space()
        os.environ["SPACE_ID"] = "owner/demo"
        os.environ["SPACE_HARDWARE"] = "zerogpu"
        space_hardware = _is_zero_gpu_space()
        os.environ.pop("SPACE_ID", None)
        os.environ.pop("SPACE_HARDWARE", None)
        os.environ["SPACES_ZERO_GPU"] = "1"
        runtime_marker = _is_zero_gpu_space()
        return normal and space_hardware and runtime_marker
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


def _gpu_duration_test() -> bool:
    old = os.environ.get("EIM_GPU_SECONDS")
    old_size = os.environ.get("EIM_GPU_SIZE")
    try:
        os.environ["EIM_GPU_SIZE"] = "large"
        os.environ["EIM_GPU_SECONDS"] = "180"
        too_long = _gpu_duration_seconds() == 120
        os.environ["EIM_GPU_SECONDS"] = "2"
        too_short = _gpu_duration_seconds() == 10
        os.environ["EIM_GPU_SECONDS"] = "60"
        normal = _gpu_duration_seconds() == 60
        os.environ["EIM_GPU_SIZE"] = "xlarge"
        os.environ["EIM_GPU_SECONDS"] = "120"
        xlarge_clamped = _gpu_duration_seconds() == 60
        os.environ["EIM_GPU_SECONDS"] = "not-a-number"
        malformed_fallback = _gpu_duration_seconds() == 60
        return too_long and too_short and normal and xlarge_clamped and malformed_fallback
    finally:
        if old is None:
            os.environ.pop("EIM_GPU_SECONDS", None)
        else:
            os.environ["EIM_GPU_SECONDS"] = old
        if old_size is None:
            os.environ.pop("EIM_GPU_SIZE", None)
        else:
            os.environ["EIM_GPU_SIZE"] = old_size


def _space_auto_backend_test() -> bool:
    keys = ("SPACE_ID", "SPACES_ID", "SPACE_HARDWARE", "SPACES_ZERO_GPU", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "EIM_API_BASE", "OPENAI_BASE_URL")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ["SPACE_ID"] = "owner/test-space"
        os.environ["HF_TOKEN"] = "hf_test_token"
        cpu_space_defaults_local = CodeLM._choose_backend("auto") == "local"
        explicit_hf_backend_is_respected = CodeLM._choose_backend("hf") == "hf"
        os.environ["SPACES_ZERO_GPU"] = "1"
        zero_gpu_uses_local_weights = CodeLM._choose_backend("auto") == "local"
        return cpu_space_defaults_local and explicit_hf_backend_is_respected and zero_gpu_uses_local_weights
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(run_selftest())
    if "--backend-check" in sys.argv:
        _verify_backend()
